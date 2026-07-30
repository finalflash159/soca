import { Box, Text } from "ink";
import type { WorkflowEvent } from "../protocol.js";
import { COLOR, ICON, ROLE } from "../theme.js";
import { Panel } from "./Primitives.js";

function eventLabel(event: WorkflowEvent): string {
  return event.event.replaceAll("_", " ");
}
function eventColor(event: WorkflowEvent): string {
  if (event.event === "turn_terminal") {
    const status = String(event.payload["terminal_status"] ?? "");
    return ["safe_failure", "system_failure", "budget_exhausted"].includes(status)
      ? ROLE.danger
      : ROLE.ok;
  }
  if (event.status === "failed" || event.status === "cancelled") return ROLE.danger;
  if (event.event === "answer_delta" || event.event === "public_update") return ROLE.busy;
  return ROLE.info;
}

export function WorkflowInspector({
  events,
  pendingAnswer,
  terminalStatus,
  expanded,
  width,
  onToggle,
}: {
  events: WorkflowEvent[];
  pendingAnswer: string;
  terminalStatus: string | null;
  expanded: boolean;
  width: number;
  onToggle: () => void;
}) {
  const current = events.at(-1);
  if (!current) return null;
  return (
    <Box paddingX={1} marginTop={1}>
      <Panel
        title="workflow"
        subtitle={expanded ? "collapse" : "inspect"}
        width={width}
        variant={terminalStatus && terminalStatus !== "achieved" ? "danger" : "info"}
      >
        <Box flexDirection="column">
          <Text color={COLOR.text}>
            {current.run_id.slice(0, 10)} · {events.length} events ·{" "}
            {terminalStatus ?? "in progress"}
          </Text>
          {pendingAnswer ? (
            <Text color={ROLE.busy} wrap="truncate-end">
              {ICON.dot} draft · {pendingAnswer}
            </Text>
          ) : null}
          <Text color={COLOR.alt}>
            {ICON.pointer} {current.node} · {eventLabel(current)} · press /workflow to{" "}
            {expanded ? "collapse" : "expand"}
          </Text>
          {expanded ? (
            <Box marginTop={1} flexDirection="column">
              {events.map((event) => {
                const text =
                  typeof event.payload["text"] === "string"
                    ? ` · ${event.payload["text"]}`
                    : typeof event.payload["terminal_status"] === "string"
                      ? ` · ${event.payload["terminal_status"]}`
                      : "";
                return (
                  <Text key={`${event.run_id}:${event.sequence}`} color={eventColor(event)}>
                    {event.sequence.toString().padStart(2, "0")} {eventLabel(event)} · {event.node}
                    {text}
                  </Text>
                );
              })}
            </Box>
          ) : null}
          <Text color={COLOR.muted}>
            {expanded ? "" : "terminal trace is retained for this turn"}
          </Text>
        </Box>
      </Panel>
    </Box>
  );
}
