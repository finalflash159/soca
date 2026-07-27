import { useEffect, useState } from "react";
import { Box, Text } from "ink";
import { animationsEnabled } from "../capabilities.js";
import { COLOR, ICON, MEMORY_STYLE, styleOf } from "../theme.js";
import type { MemoryType } from "../theme.js";

export interface ChipState {
  type: MemoryType;
  label: string;
  detail: string;
  active: boolean;
}

export function MemoryChips({ chips }: { chips: ChipState[] }) {
  const [pulse, setPulse] = useState(true);
  useEffect(() => {
    if (!animationsEnabled()) return;
    const timer = setInterval(() => setPulse((value) => !value), 600);
    timer.unref?.();
    return () => clearInterval(timer);
  }, []);
  return (
    <Box>
      {chips.map((chip) => {
        const style = styleOf(MEMORY_STYLE, chip.type);
        const dot = chip.active ? (pulse ? ICON.on : ICON.half) : ICON.off;
        return (
          <Box
            key={chip.type}
            width={20}
            flexDirection="column"
            marginRight={1}
          >
            <Text color={chip.active ? style.color : COLOR.muted}>
              {dot} <Text bold>{chip.label}</Text>
            </Text>
            <Text color={COLOR.muted}>{`  ${chip.detail}`}</Text>
          </Box>
        );
      })}
    </Box>
  );
}
