import { Box, Text } from "ink";
import {
  COLOR,
  ICON,
  RETRIEVAL_STYLE,
  ROLE,
  meterCells,
  styleOf,
} from "../theme.js";

export interface RetrievalHitView {
  path: string;
  score: number;
  sparse_score?: number;
  dense_score?: number;
  fusion_score?: number;
}
export interface RetrievalTraceView {
  query: string;
  tier: "deterministic" | "semantic" | "llm" | "none";
  latencyMs: number;
  columns: Array<{ source: string; hits: RetrievalHitView[] }>;
  fused: Array<{ path: string; picked: boolean; backend?: string; score?: number }>;
  rejectedCount?: number;
  evidence?: Record<string, unknown> | null;
  citation?: { path: string; lineStart?: number; lineEnd?: number };
}

function ScoreRow({ hit, color }: { hit: RetrievalHitView; color: string }) {
  const { filled } = meterCells(hit.score, 5);
  return (
    <Text>
      <Text color={color}>{ICON.bar.repeat(filled)}</Text>
      <Text color={COLOR.border}>{ICON.bar.repeat(5 - filled)}</Text>
      <Text color={COLOR.muted}>{` ${hit.path}`}</Text>
    </Text>
  );
}

export function RetrievalInspector({
  trace,
  width,
}: {
  trace: RetrievalTraceView;
  width: number;
}) {
  const columnWidth = Math.max(
    16,
    Math.floor((width - 6) / Math.max(1, trace.columns.length + 1)),
  );
  return (
    <Box flexDirection="column">
      <Text>
        <Text color={COLOR.muted}>query </Text>
        <Text color={COLOR.text}>{trace.query}</Text>
      </Text>
      <Text color={COLOR.muted}>
        backend {trace.columns.map((column) => column.source).join(" + ") || "none"}
        {` · rejected ${trace.rejectedCount ?? 0}`}
      </Text>
      <Text>
        <Text color={COLOR.muted}>router </Text>
        <Text color={ROLE.focus}>{trace.tier}</Text>
        <Text color={COLOR.muted}>{` · ${trace.latencyMs.toFixed(1)}ms`}</Text>
      </Text>
      <Box marginTop={1}>
        {trace.columns.map((column) => {
          const style = styleOf(RETRIEVAL_STYLE, column.source);
          return (
            <Box
              key={column.source}
              width={columnWidth}
              flexDirection="column"
              marginRight={1}
            >
              <Text bold color={style.color}>
                {style.tag === "·" ? column.source : style.tag}
              </Text>
              {column.hits.map((hit) => (
                <ScoreRow key={hit.path} hit={hit} color={style.color} />
              ))}
            </Box>
          );
        })}
        <Box width={columnWidth} flexDirection="column">
          <Text bold color={ROLE.focus}>
            {styleOf(RETRIEVAL_STYLE, "rrf").tag}
          </Text>
              {trace.fused.map((item) => (
            <Text
              key={item.path}
              color={item.picked ? ROLE.focus : COLOR.muted}
              bold={item.picked}
            >
              {item.picked ? `${ICON.pointer} ${item.path}` : `  ${item.path}`}
              {item.backend ? ` · ${item.backend}` : ""}
            </Text>
          ))}
        </Box>
      </Box>
      {trace.citation ? (
        <Text
          color={ROLE.info}
        >{`[K1] ${trace.citation.path}${trace.citation.lineStart ? `:${trace.citation.lineStart}-${trace.citation.lineEnd}` : ""}`}</Text>
      ) : null}
      {trace.evidence ? (
        <Text color={COLOR.muted}>
          evidence {String(trace.evidence["status"] ?? "unknown")} ·{" "}
          {String(trace.evidence["reason"] ?? "")}
        </Text>
      ) : null}
    </Box>
  );
}
