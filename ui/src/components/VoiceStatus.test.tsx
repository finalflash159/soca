import React from "react";
import { render } from "ink-testing-library";
import { describe, expect, it } from "vitest";
import { splitSpeechText, VoiceStatus } from "./VoiceStatus.js";

describe("VoiceStatus speech caption", () => {
  it("splits Vietnamese text on grapheme boundaries", () => {
    expect(splitSpeechText("chào bạn", 0.5)).toEqual({
      spoken: "chào",
      pending: " bạn",
    });
  });

  it("keeps the speaking status while adding the speech caption", () => {
    const view = render(
      <VoiceStatus
        state="speaking"
        note=""
        turnIndex={2}
        latencyMs={null}
        caption={null}
        speechChunks={[
          {
            index: 0,
            text: "Mình đang trả lời bạn.",
            durationMs: 1400,
            status: "playing",
          },
        ]}
        bargeIn="armed"
      />,
    );

    const frame = view.lastFrame() ?? "";
    expect(frame).toContain("speaking");
    expect(frame).toContain("barge-in armed");
    expect(frame).toContain("Mình đang trả lời bạn.");
    view.unmount();
  });
});
