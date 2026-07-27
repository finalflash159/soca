import { Box, Text, useInput } from "ink";
import { useEffect, useState } from "react";
import { COLOR, ICON, ROLE } from "../theme.js";

export interface MemoryProposalView { id: string; kind: "preference" | "stable_fact" | "project" | "correction"; statement: string; confidence: number; createdAt: string; }

export function MemoryProposalInbox({ proposals, error, onApprove, onReject, onClose }: { proposals: MemoryProposalView[]; error: string; onApprove: (id: string) => void; onReject: (id: string) => void; onClose: () => void }) {
  const [index, setIndex] = useState(0);
  const [confirm, setConfirm] = useState<"approve" | "reject" | null>(null);
  useEffect(() => {
    setIndex((value) => Math.min(value, Math.max(0, proposals.length - 1)));
  }, [proposals.length]);
  const selectedIndex = Math.min(index, Math.max(0, proposals.length - 1));
  const current = proposals[selectedIndex];
  useInput((input, key) => {
    if (key.escape) { if (confirm) setConfirm(null); else onClose(); return; }
    if (confirm) { if (key.return && current) { (confirm === "approve" ? onApprove : onReject)(current.id); setConfirm(null); } return; }
    if (key.upArrow) { setIndex((value) => Math.max(0, value - 1)); return; }
    if (key.downArrow) { setIndex((value) => Math.min(Math.max(0, proposals.length - 1), value + 1)); return; }
    if (!current) return;
    if (input === "a" || input === "r") setConfirm(input === "a" ? "approve" : "reject");
  });
  return <Box flexDirection="column" paddingX={1}><Text bold color={ROLE.focus}>Memory proposals</Text>{proposals.length === 0 ? <Text color={COLOR.muted}>{ICON.dot} No pending proposals. Press Esc to return.</Text> : proposals.map((proposal, row) => <Box key={proposal.id} flexDirection="column"><Text color={row === selectedIndex ? ROLE.focus : COLOR.muted} bold={row === selectedIndex}>{row === selectedIndex ? `${ICON.pointer} ` : "  "}{proposal.kind} · {proposal.statement}</Text>{row === selectedIndex ? <Text color={COLOR.muted}>{`    confidence ${(proposal.confidence * 100).toFixed(0)}% · ${proposal.id.slice(0, 8)}`}</Text> : null}</Box>)}{confirm && current ? <Box marginTop={1} flexDirection="column" borderStyle="round" borderColor={confirm === "reject" ? ROLE.danger : ROLE.focus} paddingX={1}><Text bold color={confirm === "reject" ? ROLE.danger : ROLE.focus}>{`Confirm ${confirm}`}</Text><Text>{current.statement}</Text><Text color={COLOR.muted}>Press Enter to confirm or Esc to cancel.</Text></Box> : <Text color={COLOR.muted}>↑/↓ select · a approve · r reject · Esc close</Text>}{error ? <Text color={ROLE.danger}>{ICON.err} {error}</Text> : null}</Box>;
}
