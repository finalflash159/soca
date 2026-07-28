import { Box } from "ink";
import type {
  TurnProgressEvent,
  TurnProgressPhase,
} from "../protocol.js";
import { COLOR, ROLE } from "../theme.js";
import { Spinner } from "./Primitives.js";

interface PhaseView {
  label: string;
  color: string;
}

const PHASE_VIEW: Record<TurnProgressPhase, PhaseView> = {
  preparing: { label: "Preparing…", color: ROLE.info },
  analyzing: { label: "Analyzing…", color: ROLE.focus },
  routing: { label: "Routing…", color: ROLE.info },
  memory: { label: "Searching memory…", color: ROLE.ok },
  retrieval: { label: "Searching knowledge…", color: COLOR.accentBright },
  tool: { label: "Running tool…", color: ROLE.busy },
  synthesis: { label: "Generating…", color: ROLE.busy },
  validation: { label: "Validating…", color: ROLE.ok },
  speech: { label: "Synthesizing speech…", color: ROLE.ok },
  complete: { label: "Done", color: ROLE.ok },
};

function progressLabel(event: TurnProgressEvent): string {
  if (event.operation.startsWith("tool:")) {
    return `Running ${event.operation.slice("tool:".length)}…`;
  }
  if (event.operation === "speech_recognition") {
    return "Transcribing…";
  }
  return PHASE_VIEW[event.phase].label;
}

export function TurnProgress({
  progress,
}: {
  progress: TurnProgressEvent;
}) {
  const view = PHASE_VIEW[progress.phase];
  return (
    <Box paddingX={1} marginBottom={1}>
      <Spinner
        label={progressLabel(progress)}
        color={view.color}
        labelColor={view.color}
      />
    </Box>
  );
}
