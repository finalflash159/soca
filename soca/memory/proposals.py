from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

ProposalKind = Literal["preference", "stable_fact", "project", "correction"]
ProposalStatus = Literal["pending", "approved", "rejected"]
_SECRET_RE = re.compile(r"(api[_ -]?key|token|password|secret|sk-[A-Za-z0-9])", re.I)
_IMPERATIVE_RE = re.compile(r"\b(ignore|execute|run|delete|send|reveal|open|call)\b", re.I)


@dataclass(frozen=True)
class MemoryProposal:
    id: str
    kind: ProposalKind
    statement: str
    evidence_excerpt: str
    confidence: float
    source_episode_id: str
    created_at: datetime
    status: ProposalStatus = "pending"

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.id)
            uuid.UUID(self.source_episode_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("proposal IDs must be UUIDs") from exc
        if self.kind not in {"preference", "stable_fact", "project", "correction"}:
            raise ValueError("unknown proposal kind")
        if self.status not in {"pending", "approved", "rejected"}:
            raise ValueError("unknown proposal status")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("proposal confidence must be numeric")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("proposal confidence must be between 0 and 1")
        for value in (self.statement, self.evidence_excerpt):
            if not value.strip() or len(value) > 1_000:
                raise ValueError("proposal text is invalid")
        if _SECRET_RE.search(self.statement) or _SECRET_RE.search(self.evidence_excerpt):
            raise ValueError("proposal contains secret-like text")
        if _IMPERATIVE_RE.search(self.statement):
            raise ValueError("proposal statement must not contain instructions")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
            raise ValueError("proposal timestamp must be timezone-aware")
        object.__setattr__(self, "created_at", self.created_at.astimezone(UTC))


class ProposalStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser()
        self._ensure_root()

    def put(self, proposal: MemoryProposal) -> MemoryProposal:
        path = self._path(proposal.id)
        if path.exists() or path.is_symlink():
            existing = self.get(proposal.id)
            if existing is None:
                raise ValueError("proposal file is invalid")
            if existing != proposal:
                raise ValueError("proposal ID already exists with different content")
            return existing
        self._atomic_write(path, _encode(proposal))
        return proposal

    def get(self, proposal_id: str) -> MemoryProposal | None:
        path = self._path(proposal_id)
        if not path.exists():
            return None
        if path.is_symlink():
            raise ValueError("proposal path may not be a symlink")
        return _decode(json.loads(path.read_text(encoding="utf-8")))

    def list(self, *, status: ProposalStatus | None = None) -> tuple[MemoryProposal, ...]:
        values: list[MemoryProposal] = []
        for path in sorted(self.root.glob("*.json")):
            if path.is_symlink():
                raise ValueError("proposal store contains a symlink")
            item = _decode(json.loads(path.read_text(encoding="utf-8")))
            if status is None or item.status == status:
                values.append(item)
        return tuple(values)

    def transition(self, proposal_id: str, status: ProposalStatus) -> MemoryProposal:
        current = self.get(proposal_id)
        if current is None:
            raise KeyError(proposal_id)
        if current.status == status:
            return current
        if current.status != "pending":
            raise ValueError("proposal has already been resolved")
        updated = replace(current, status=status)
        self._atomic_write(self._path(proposal_id), _encode(updated))
        return updated

    def _path(self, proposal_id: str) -> Path:
        try:
            value = uuid.UUID(proposal_id)
        except (ValueError, AttributeError) as exc:
            raise ValueError("proposal ID must be a UUID") from exc
        return self.root / f"{value}.json"

    def _ensure_root(self) -> None:
        if self.root.exists() and self.root.is_symlink():
            raise ValueError("proposal directory may not be a symlink")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.root.chmod(0o700)

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


def _encode(proposal: MemoryProposal) -> bytes:
    payload = {
        "id": proposal.id,
        "kind": proposal.kind,
        "statement": proposal.statement,
        "evidence_excerpt": proposal.evidence_excerpt,
        "confidence": proposal.confidence,
        "source_episode_id": proposal.source_episode_id,
        "created_at": proposal.created_at.isoformat(),
        "status": proposal.status,
    }
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _decode(payload: object) -> MemoryProposal:
    if not isinstance(payload, dict):
        raise ValueError("proposal root must be an object")
    required = {"id", "kind", "statement", "evidence_excerpt", "confidence", "source_episode_id", "created_at", "status"}
    if set(payload) != required:
        raise ValueError("proposal schema is invalid")
    return MemoryProposal(
        id=payload["id"],
        kind=payload["kind"],
        statement=payload["statement"],
        evidence_excerpt=payload["evidence_excerpt"],
        confidence=payload["confidence"],
        source_episode_id=payload["source_episode_id"],
        created_at=datetime.fromisoformat(payload["created_at"]),
        status=payload["status"],
    )


__all__ = ["MemoryProposal", "ProposalStore", "ProposalKind", "ProposalStatus"]
