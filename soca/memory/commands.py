from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from soca.memory.proposals import MemoryProposal, ProposalStore


@dataclass(frozen=True)
class MemoryCommandResult:
    status: str
    proposal_id: str
    note_path: str = ""
    message: str = ""


class MemoryCommands:
    def __init__(self, vault: str | Path, proposals: ProposalStore) -> None:
        self.vault = Path(vault).expanduser().resolve()
        self.proposals = proposals

    def list_pending(self) -> tuple[MemoryProposal, ...]:
        return self.proposals.list(status="pending")

    def approve(self, proposal_id: str) -> MemoryCommandResult:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            return MemoryCommandResult("not_found", proposal_id, message="proposal not found")
        if proposal.status == "rejected":
            return MemoryCommandResult("conflict", proposal_id, message="proposal was rejected")
        note_path = self._note_path(proposal_id)
        if proposal.status == "pending":
            self._write_note(note_path, proposal)
            self.proposals.transition(proposal_id, "approved")
        elif not note_path.exists():
            self._write_note(note_path, proposal)
        return MemoryCommandResult(
            "approved",
            proposal_id,
            note_path=str(note_path.relative_to(self.vault)),
            message="proposal approved",
        )

    def reject(self, proposal_id: str) -> MemoryCommandResult:
        proposal = self.proposals.get(proposal_id)
        if proposal is None:
            return MemoryCommandResult("not_found", proposal_id, message="proposal not found")
        if proposal.status == "approved":
            return MemoryCommandResult("conflict", proposal_id, message="proposal was approved")
        if proposal.status == "pending":
            self.proposals.transition(proposal_id, "rejected")
        return MemoryCommandResult("rejected", proposal_id, message="proposal rejected")

    def _note_path(self, proposal_id: str) -> Path:
        captured = self.vault / "memory" / "captured"
        self._validate_components(captured)
        captured.mkdir(mode=0o700, parents=True, exist_ok=True)
        return captured / f"{proposal_id}.md"

    def _write_note(self, path: Path, proposal: MemoryProposal) -> None:
        if path.exists() and not path.is_symlink():
            return
        if path.is_symlink():
            raise ValueError("captured note may not be a symlink")
        body = (
            "---\n"
            f"created_at: {proposal.created_at.isoformat()}\n"
            "importance: 5\n"
            "---\n\n"
            f"# {proposal.kind.replace('_', ' ').title()}\n\n"
            f"{proposal.statement}\n\n"
            "Evidence:\n"
            f"> {proposal.evidence_excerpt}\n"
        ).encode()
        descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(body)
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

    def _validate_components(self, path: Path) -> None:
        try:
            path.relative_to(self.vault)
        except ValueError as exc:
            raise ValueError("captured note is outside the vault") from exc
        current = self.vault
        for part in path.relative_to(self.vault).parts:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ValueError("captured note path contains a symlink")


__all__ = ["MemoryCommandResult", "MemoryCommands"]
