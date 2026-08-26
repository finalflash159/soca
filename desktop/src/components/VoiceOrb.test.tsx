import { describe, expect, it } from "vitest";

import { perceptualVoiceLevel } from "./VoiceOrb";

describe("perceptualVoiceLevel", () => {
  it("keeps recorder silence visually still", () => {
    expect(perceptualVoiceLevel(0)).toBe(0);
    expect(perceptualVoiceLevel(0.00035)).toBe(0);
  });

  it("makes an observed close-mic speech envelope visibly responsive", () => {
    // RMS measured from the real default MacBook microphone during the
    // packaged-sidecar probe was approximately 0.003.
    expect(perceptualVoiceLevel(0.003)).toBeGreaterThan(0.15);
    expect(perceptualVoiceLevel(0.003)).toBeLessThan(1);
  });

  it("is bounded and monotonic", () => {
    expect(perceptualVoiceLevel(0.01)).toBeGreaterThan(perceptualVoiceLevel(0.003));
    expect(perceptualVoiceLevel(1)).toBe(1);
  });
});
