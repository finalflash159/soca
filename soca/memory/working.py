"""Bounded, typed conversation state for answer prompts and compaction jobs.

This module owns *working* memory only.  It is intentionally separate from
approved core memory and from the searchable archive: none of its summaries is
eligible as an archival fact or retrieval evidence.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Literal

TurnStatus = Literal["pending", "complete", "interrupted", "failed"]
SummaryMode = Literal["trim_only", "background_summary"]


def _normalise(text: str) -> str:
    return " ".join(text.strip().split())


def approximate_tokens(text: str) -> int:
    """Conservative deterministic fallback until a local tokenizer is supplied."""
    if not text.strip():
        return 0
    return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass(frozen=True)
class ConversationTurn:
    sequence: int
    user_text: str
    assistant_text: str = ""
    status: TurnStatus = "pending"

    def __post_init__(self) -> None:
        if self.sequence < 1:
            raise ValueError("turn sequence must be positive")
        if self.status not in {"pending", "complete", "interrupted", "failed"}:
            raise ValueError("invalid conversation turn status")
        if not self.user_text.strip():
            raise ValueError("conversation turn needs user text")
        if self.status == "complete" and not self.assistant_text.strip():
            raise ValueError("complete conversation turn needs delivered assistant text")


@dataclass(frozen=True)
class WorkingSummaryArtifact:
    version: int
    generation: int
    source_through_sequence: int
    summary: str
    user_constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    corrections: tuple[str, ...] = ()
    open_items: tuple[str, ...] = ()
    continuity_refs: tuple[str, ...] = ()
    prompt_fingerprint: str = ""

    def __post_init__(self) -> None:
        if self.version != 1 or self.generation < 0 or self.source_through_sequence < 0:
            raise ValueError("invalid working summary version or sequence")
        if len(self.summary.split()) > 256:
            raise ValueError("working summary exceeds 256-token content budget")
        for values in (
            self.user_constraints,
            self.decisions,
            self.corrections,
            self.open_items,
            self.continuity_refs,
        ):
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError("working summary fields must contain non-empty strings")
        if approximate_tokens(self.render()) > 256:
            raise ValueError("working summary artifact exceeds 256-token content budget")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def render(self) -> str:
        sections: list[str] = []
        if self.summary:
            sections.append("Summary:\n" + self.summary)
        for label, values in (
            ("User constraints", self.user_constraints),
            ("Active decisions", self.decisions),
            ("Corrections", self.corrections),
            ("Open items", self.open_items),
            ("Continuity references", self.continuity_refs),
        ):
            if values:
                sections.append(label + ":\n" + "\n".join(f"- {value}" for value in values))
        return "\n".join(sections)


@dataclass(frozen=True)
class WorkingMemoryPolicy:
    hard_limit_tokens: int = 16_384
    high_watermark_tokens: int = 15_000
    target_tokens: int = 12_000
    summary_budget_tokens: int = 256
    recent_budget_tokens: int = 512
    minimum_recent_complete_turns: int = 2
    preferred_recent_complete_turns: int = 4
    mode: SummaryMode = "trim_only"

    def __post_init__(self) -> None:
        if (self.target_tokens, self.high_watermark_tokens, self.hard_limit_tokens) != (
            12_000,
            15_000,
            16_384,
        ):
            raise ValueError(
                "working_v2_16k requires target/high/hard = 12000/15000/16384"
            )
        if (self.summary_budget_tokens, self.recent_budget_tokens) != (256, 512):
            raise ValueError("working_v2_16k requires summary/recent = 256/512")
        if self.minimum_recent_complete_turns != 2 or self.preferred_recent_complete_turns != 4:
            raise ValueError("working_v2_16k requires minimum/preferred recent turns = 2/4")
        if self.mode not in {"trim_only", "background_summary"}:
            raise ValueError("unknown working summary mode")


@dataclass(frozen=True)
class CompactionJob:
    generation: int
    revision: int
    previous_summary: WorkingSummaryArtifact | None
    frozen_turns: tuple[ConversationTurn, ...]


@dataclass(frozen=True)
class WorkingMemorySnapshot:
    thread_id: str
    summary: WorkingSummaryArtifact | None
    turns: tuple[ConversationTurn, ...]
    generation: int
    revision: int
    token_count: int
    pending_compaction: bool


class WorkingMemory:
    """Conversation state with generation-CAS compaction publication.

    ``trim_only`` is a valid degraded mode: it preserves the hard prompt cap
    without silently pretending that an extractive/regex summary is truthful.
    """

    def __init__(
        self,
        *,
        thread_id: str = "default",
        policy: WorkingMemoryPolicy | None = None,
        token_counter: Callable[[str], int] = approximate_tokens,
    ) -> None:
        if not thread_id.strip():
            raise ValueError("thread_id must not be empty")
        self.thread_id = thread_id
        self.policy = policy or WorkingMemoryPolicy()
        self._token_counter = token_counter
        self._turns: list[ConversationTurn] = []
        self._summary: WorkingSummaryArtifact | None = None
        self._generation = 0
        self._revision = 0
        self._pending_compaction = False

    @property
    def snapshot(self) -> WorkingMemorySnapshot:
        return WorkingMemorySnapshot(
            thread_id=self.thread_id,
            summary=self._summary,
            turns=tuple(self._turns),
            generation=self._generation,
            revision=self._revision,
            token_count=self._token_count(),
            pending_compaction=self._pending_compaction,
        )

    def begin_turn(self, user_text: str) -> ConversationTurn:
        text = _normalise(user_text)
        if not text:
            raise ValueError("user turn must not be empty")
        sequence = self._turns[-1].sequence + 1 if self._turns else 1
        turn = ConversationTurn(sequence=sequence, user_text=text)
        self._turns.append(turn)
        self._revision += 1
        return turn

    def finish_turn(self, sequence: int, assistant_text: str, *, status: TurnStatus = "complete") -> None:
        if status not in {"complete", "interrupted", "failed"}:
            raise ValueError("finish_turn needs terminal status")
        for index in range(len(self._turns) - 1, -1, -1):
            current = self._turns[index]
            if current.sequence != sequence:
                continue
            if current.status != "pending":
                raise ValueError("conversation turn is already terminal")
            delivered = _normalise(assistant_text)
            if status == "complete" and not delivered:
                raise ValueError("complete turn needs delivered assistant text")
            self._turns[index] = ConversationTurn(
                sequence=current.sequence,
                user_text=current.user_text,
                assistant_text=delivered,
                status=status,
            )
            self._revision += 1
            return
        raise ValueError("unknown conversation turn sequence")

    def prepare_compaction(self, *, force: bool = False) -> CompactionJob | None:
        if self._pending_compaction:
            return None
        if not force and self._token_count() < self.policy.high_watermark_tokens:
            return None
        completed = [turn for turn in self._turns if turn.status == "complete"]
        keep = completed[-self.policy.preferred_recent_complete_turns :]
        frozen_sequences = {turn.sequence for turn in keep}
        frozen = tuple(
            turn for turn in self._turns if turn.status == "complete" and turn.sequence not in frozen_sequences
        )
        if not frozen:
            return None
        self._generation += 1
        self._pending_compaction = True
        return CompactionJob(
            generation=self._generation,
            revision=self._revision,
            previous_summary=self._summary,
            frozen_turns=frozen,
        )

    def publish_summary(self, job: CompactionJob, artifact: WorkingSummaryArtifact) -> bool:
        """Publish only the exact job generation; stale workers are discarded."""
        if not self._pending_compaction or job.generation != self._generation:
            return False
        if artifact.generation != job.generation:
            raise ValueError("summary artifact generation does not match job")
        if artifact.source_through_sequence != job.frozen_turns[-1].sequence:
            raise ValueError("summary source sequence does not cover frozen prefix")
        frozen = {turn.sequence for turn in job.frozen_turns}
        self._turns = [turn for turn in self._turns if turn.sequence not in frozen]
        self._summary = artifact
        self._pending_compaction = False
        self._revision += 1
        self._enforce_hard_limit()
        return True

    def cancel_compaction(self, generation: int | None = None) -> bool:
        if not self._pending_compaction or (generation is not None and generation != self._generation):
            return False
        self._pending_compaction = False
        return True

    def trim_only(self) -> None:
        """Deterministic fallback when no approved local summary model is ready."""
        if self._pending_compaction:
            self._pending_compaction = False
        self._enforce_hard_limit(target=self.policy.target_tokens)

    def render(self) -> str:
        parts: list[str] = []
        if self._summary is not None and (rendered_summary := self._summary.render()):
            parts.append("Earlier conversation state:\n" + rendered_summary)
        lines: list[str] = []
        for turn in self._turns:
            lines.append("User: " + turn.user_text)
            if turn.assistant_text:
                lines.append("Assistant: " + turn.assistant_text)
        if lines:
            parts.append("Recent conversation:\n" + "\n".join(lines))
        return "\n\n".join(parts)

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "thread_id": self.thread_id,
            "summary": self._summary.to_dict() if self._summary is not None else None,
            "turns": [asdict(turn) for turn in self._turns],
            "generation": self._generation,
            "revision": self._revision,
        }

    @classmethod
    def from_dict(
        cls,
        payload: object,
        *,
        policy: WorkingMemoryPolicy | None = None,
        token_counter: Callable[[str], int] = approximate_tokens,
    ) -> WorkingMemory:
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError("unsupported working memory checkpoint")
        thread_id = payload.get("thread_id")
        turns_data = payload.get("turns")
        if not isinstance(thread_id, str) or not isinstance(turns_data, list):
            raise ValueError("invalid working memory checkpoint")
        memory = cls(thread_id=thread_id, policy=policy, token_counter=token_counter)
        turns: list[ConversationTurn] = []
        for value in turns_data:
            if not isinstance(value, dict):
                raise ValueError("invalid checkpoint turn")
            turns.append(
                ConversationTurn(
                    sequence=int(value.get("sequence", 0)),
                    user_text=str(value.get("user_text", "")),
                    assistant_text=str(value.get("assistant_text", "")),
                    status=str(value.get("status", "pending")),  # type: ignore[arg-type]
                )
            )
        if any(left.sequence >= right.sequence for left, right in zip(turns, turns[1:], strict=False)):
            raise ValueError("checkpoint turn sequences must be monotonic")
        summary_data = payload.get("summary")
        if summary_data is not None:
            if not isinstance(summary_data, dict):
                raise ValueError("invalid checkpoint summary")
            memory._summary = WorkingSummaryArtifact(
                version=int(summary_data.get("version", 0)),
                generation=int(summary_data.get("generation", 0)),
                source_through_sequence=int(summary_data.get("source_through_sequence", 0)),
                summary=str(summary_data.get("summary", "")),
                user_constraints=tuple(summary_data.get("user_constraints", ())),
                decisions=tuple(summary_data.get("decisions", ())),
                corrections=tuple(summary_data.get("corrections", ())),
                open_items=tuple(summary_data.get("open_items", ())),
                continuity_refs=tuple(summary_data.get("continuity_refs", ())),
                prompt_fingerprint=str(summary_data.get("prompt_fingerprint", "")),
            )
        memory._turns = turns
        memory._generation = int(payload.get("generation", 0))
        memory._revision = int(payload.get("revision", 0))
        if memory._generation < 0 or memory._revision < 0:
            raise ValueError("invalid checkpoint generation")
        return memory

    def _token_count(self) -> int:
        return self._token_counter(self.render())

    def _enforce_hard_limit(self, *, target: int | None = None) -> None:
        limit = target if target is not None else self.policy.hard_limit_tokens
        while self._token_count() > limit and len(self._turns) > self.policy.minimum_recent_complete_turns:
            self._turns.pop(0)
            self._revision += 1


__all__ = [
    "CompactionJob",
    "ConversationTurn",
    "SummaryMode",
    "TurnStatus",
    "WorkingMemory",
    "WorkingMemoryPolicy",
    "WorkingMemorySnapshot",
    "WorkingSummaryArtifact",
    "approximate_tokens",
]
