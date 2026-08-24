/** Audio-reactive visual for the dedicated Voice surface. */

import type { CSSProperties } from "react";

import voiceOrbAsset from "@/assets/soca-voice-orb.png";
import type { VoicePhase, VoiceState } from "@/engine/voice";
import { peakLevel } from "@/engine/voice";
import { cn } from "@/lib/utils";

export type VoicePresentation = "compact" | "immersive";
export type VoiceOrbMode = "setup" | VoicePhase;

/** A large orb is reserved for live microphone capture, never a vague busy state. */
export function voicePresentationFor(voice: VoiceState, ready: boolean): VoicePresentation {
  return ready && voice.phase === "listening" ? "immersive" : "compact";
}

export function voiceOrbModeFor(voice: VoiceState, ready: boolean): VoiceOrbMode {
  return ready ? voice.phase : "setup";
}

export function voiceOrbStatusFor(mode: VoiceOrbMode): string {
  switch (mode) {
    case "setup":
      return "Voice cần thiết lập";
    case "off":
      return "Microphone đang tắt";
    case "starting":
      return "Đang chuẩn bị Voice…";
    case "idle":
      return "Sẵn sàng lắng nghe";
    case "listening":
      return "Đang lắng nghe";
    case "transcribing":
      return "Đang nhận giọng nói";
    case "thinking":
      return "Đang soạn câu trả lời";
    case "speaking":
      return "SoCa đang nói";
  }
}

/** Peak over the frames rendered in one 100 ms UI update, not a fabricated envelope. */
function recentLevel(levels: number[]): number {
  return peakLevel(levels.slice(-6));
}

interface VoiceOrbProps {
  voice: VoiceState;
  ready: boolean;
  presentation: VoicePresentation;
}

export function VoiceOrb({ voice, ready, presentation }: VoiceOrbProps) {
  const mode = voiceOrbModeFor(voice, ready);
  const level =
    mode === "listening"
      ? recentLevel(voice.levels)
      : mode === "speaking"
        ? recentLevel(voice.assistantLevels)
        : 0;
  const status = voiceOrbStatusFor(mode);

  return (
    <div
      className={cn(
        "voice-orb",
        `voice-orb--${mode}`,
        `voice-orb--${presentation}`,
      )}
      style={{ "--voice-orb-level": level } as CSSProperties}
      role="img"
      aria-label={status}
      data-testid="voice-orb"
      data-voice-orb-mode={mode}
      data-voice-orb-presentation={presentation}
    >
      <img className="voice-orb__aura" src={voiceOrbAsset} alt="" aria-hidden="true" />
      <div className="voice-orb__motion">
        <img className="voice-orb__core" src={voiceOrbAsset} alt="" aria-hidden="true" />
      </div>
    </div>
  );
}
