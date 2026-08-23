/**
 * LLM provider, key, model and runtime-profile state.
 *
 * The key boundary is the important part here. `llm_set_key` carries a secret
 * one way, into the engine's keyring; nothing in this module ever stores, logs
 * or renders a key. The engine answers with `llm_key_status`, which carries at
 * most a `masked` form, and that is the only key material a UI may show.
 */

import type { EngineFrame } from "./protocol";

export interface Provider {
  key: string;
  label: string;
  hasKey: boolean;
  hasPricingApi: boolean;
}

export interface CatalogModel {
  id: string;
  label: string;
  context_length: number | null;
  price_prompt_per_1m: number | null;
  price_completion_per_1m: number | null;
  pricing_source: string | null;
  max_output_tokens: number | null;
  reasoning_supported: boolean;
  reasoning_mandatory: boolean;
}

export interface KeyStatus {
  provider: string;
  ok: boolean;
  pending: boolean;
  masked: string | null;
  message: string | null;
}

export interface LlmConfig {
  backend: string;
  provider: string;
  model: string;
  maxTokens: number | null;
  effectiveMaxTokens: number | null;
  reasoningEnabled: boolean;
  effectiveReasoningEnabled: boolean;
  reasoningSupported: boolean;
  reasoningMandatory: boolean;
  contextLength: number | null;
  runtimeReady: boolean;
  runtimeReason: string | null;
  localModelPath: string | null;
  settingsError: string | null;
}

export interface RuntimeProfile {
  key: string;
  status: string;
  asr: string | null;
  llm: string | null;
  tts: string | null;
  voice: string | null;
}

export interface SettingsState {
  providers: Provider[];
  /** Keyed by provider; an empty array can be a valid empty catalog. */
  catalog: Record<string, CatalogModel[]>;
  /** True only while the engine is fetching a catalog. */
  catalogLoading: Record<string, boolean>;
  pricingAsOf: string | null;
  keyStatus: Record<string, KeyStatus>;
  config: LlmConfig | null;
  profiles: RuntimeProfile[];
  activeProfile: string | null;
}

export const initialSettings: SettingsState = {
  providers: [],
  catalog: {},
  catalogLoading: {},
  pricingAsOf: null,
  keyStatus: {},
  config: null,
  profiles: [],
  activeProfile: null,
};

function str(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback;
}

function numOrNull(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

export function reduceSettings(state: SettingsState, frame: EngineFrame): SettingsState {
  switch (frame.event) {
    case "llm_providers": {
      const raw = Array.isArray(frame.providers) ? frame.providers : [];
      return {
        ...state,
        providers: raw.map((item) => {
          const provider = item as Record<string, unknown>;
          return {
            key: str(provider.key),
            label: str(provider.label),
            hasKey: provider.has_key === true,
            hasPricingApi: provider.has_pricing_api === true,
          };
        }),
      };
    }

    case "llm_catalog": {
      const provider = str(frame.provider);
      const models = Array.isArray(frame.models) ? (frame.models as CatalogModel[]) : [];
      // The engine marks loading explicitly. An empty completed catalog is a
      // truthful provider response and must not look like a hanging picker.
      const loading = frame.loading === true;
      return {
        ...state,
        catalog: { ...state.catalog, [provider]: models },
        catalogLoading: { ...state.catalogLoading, [provider]: loading },
        pricingAsOf: str(frame.pricing_as_of) || state.pricingAsOf,
      };
    }

    case "llm_key_status": {
      const provider = str(frame.provider);
      return {
        ...state,
        keyStatus: {
          ...state.keyStatus,
          [provider]: {
            provider,
            ok: frame.ok === true,
            pending: frame.pending === true,
            masked: typeof frame.masked === "string" ? frame.masked : null,
            message: typeof frame.message === "string" ? frame.message : null,
          },
        },
        providers: state.providers.map((item) =>
          item.key === provider && frame.ok === true ? { ...item, hasKey: true } : item,
        ),
      };
    }

    case "llm_config":
      return {
        ...state,
        config: {
          backend: str(frame.backend),
          provider: str(frame.provider),
          model: str(frame.model),
          maxTokens: numOrNull(frame.max_tokens),
          effectiveMaxTokens: numOrNull(frame.effective_max_tokens),
          reasoningEnabled: frame.reasoning_enabled === true,
          effectiveReasoningEnabled: frame.effective_reasoning_enabled === true,
          reasoningSupported: frame.reasoning_supported === true,
          reasoningMandatory: frame.reasoning_mandatory === true,
          contextLength: numOrNull(frame.context_length),
          runtimeReady: frame.runtime_ready === true,
          runtimeReason: typeof frame.runtime_reason === "string" ? frame.runtime_reason : null,
          localModelPath: typeof frame.local_model_path === "string" ? frame.local_model_path : null,
          settingsError: typeof frame.settings_error === "string" ? frame.settings_error : null,
        },
      };

    case "status": {
      const raw = Array.isArray(frame.profiles) ? frame.profiles : [];
      return {
        ...state,
        activeProfile: typeof frame.active_profile === "string" ? frame.active_profile : state.activeProfile,
        profiles: raw.map((item) => {
          const profile = item as Record<string, unknown>;
          return {
            key: str(profile.key),
            status: str(profile.status, "unknown"),
            asr: typeof profile.asr === "string" ? profile.asr : null,
            llm: typeof profile.llm === "string" ? profile.llm : null,
            tts: typeof profile.tts === "string" ? profile.tts : null,
            voice: typeof profile.voice === "string" ? profile.voice : null,
          };
        }),
      };
    }

    default:
      return state;
  }
}

/**
 * What will actually happen, which is not always what was requested.
 *
 * A model can force reasoning on. `docs/18` §7 obligation 8 requires the UI to
 * show the effective value, so this returns both plus the reason they differ.
 */
export function reasoningSummary(config: LlmConfig | null): string {
  if (config === null) {
    return "unknown";
  }
  if (!config.reasoningSupported) {
    return "not supported by this model";
  }
  if (config.reasoningMandatory) {
    return "always on — this model requires it";
  }
  if (config.reasoningEnabled !== config.effectiveReasoningEnabled) {
    return `requested ${config.reasoningEnabled ? "on" : "off"}, effective ${
      config.effectiveReasoningEnabled ? "on" : "off"
    }`;
  }
  return config.effectiveReasoningEnabled ? "on" : "off";
}

/** Price per million tokens, or null when the provider publishes none. */
export function modelPrice(model: CatalogModel): string | null {
  const { price_prompt_per_1m: prompt, price_completion_per_1m: completion } = model;
  if (prompt === null || completion === null) {
    return null;
  }
  return `$${prompt.toFixed(2)} / $${completion.toFixed(2)} per 1M`;
}
