import type { CSSProperties } from "react";

import type { OrbState } from "@/engine/orb";
import { cn } from "@/lib/utils";

interface ActivityOrbProps {
  state: OrbState;
  /** Diameter of the main sphere in CSS pixels. */
  size: number;
  className?: string;
}

/**
 * A calm, fixed cluster of separate spheres for engine activity.
 *
 * The spheres never orbit, overlap into a ring, pulse, or claim progress.
 * Colour is only a secondary cue; each placement has adjacent state text.
 */
export function ActivityOrb({ state, size, className }: ActivityOrbProps) {
  return (
    <span
      className={cn("activity-orb", className)}
      data-state={state}
      data-visual="static-sphere-cluster"
      style={{ "--activity-orb-size": `${size}px` } as CSSProperties}
      aria-hidden="true"
    >
      <span className="activity-orb__sphere activity-orb__sphere--primary" />
      <span className="activity-orb__sphere activity-orb__sphere--companion" />
      <span className="activity-orb__sphere activity-orb__sphere--accent" />
    </span>
  );
}
