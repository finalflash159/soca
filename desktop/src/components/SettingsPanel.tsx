/**
 * Phase 5 surface: one settings screen.
 *
 * The plan's most directly applicable lesson (§5.6.6): SoCa currently answers
 * "how is this configured?" through six entry points — `/settings`,
 * `soca status`, `soca profiles`, `soca llm-models`, `soca asr-models`,
 * `soca knowledge model`. One surface removes the question of which one to open.
 *
 * The key field is write-only. It is sent with `llm_set_key` and cleared
 * immediately; the engine holds it in its keyring and answers with at most a
 * masked form. Nothing here stores or renders a raw key.
 */

import { useEffect, useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { PanelEmpty, PanelSection } from "@/components/PanelSection";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { SettingsState } from "@/engine/settings";
import { modelPrice, reasoningSummary } from "@/engine/settings";

interface SettingsPanelProps {
  settings: SettingsState;
  connected: boolean;
  onLoadProviders: () => void;
  onSetKey: (provider: string, key: string) => void;
  onLoadModels: (provider: string, query: string) => void;
  onSelectModel: (provider: string, modelId: string) => void;
  onSelectProfile: (profileKey: string) => void;
  /** Backend, output cap and reasoning all travel on the same `llm_select`. */
  onApplyGeneration: (change: {
    backend?: string;
    maxTokens?: number;
    reasoningEnabled?: boolean;
  }) => void;
}

function ActiveConfig({ settings }: { settings: SettingsState }) {
  const config = settings.config;
  if (config === null) {
    return (
      <PanelEmpty>No configuration loaded yet.</PanelEmpty>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      {!config.runtimeReady && (
        <p className="text-destructive text-sm">
          Runtime not ready{config.settingsError !== null ? ` · ${config.settingsError}` : ""}
        </p>
      )}
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
        <dt className="text-muted-foreground">backend</dt>
        <dd>{config.backend}</dd>
        <dt className="text-muted-foreground">provider</dt>
        <dd>{config.provider}</dd>
        <dt className="text-muted-foreground">model</dt>
        <dd className="font-mono text-xs">{config.model}</dd>
        <dt className="text-muted-foreground">max output</dt>
        {/* §7 obligation 8: show the effective value, not the request. */}
        <dd>
          {config.effectiveMaxTokens ?? "—"}
          {config.maxTokens !== null && config.maxTokens !== config.effectiveMaxTokens && (
            <span className="text-muted-foreground"> (requested {config.maxTokens})</span>
          )}
        </dd>
        <dt className="text-muted-foreground">reasoning</dt>
        <dd>{reasoningSummary(config)}</dd>
        <dt className="text-muted-foreground">context</dt>
        <dd>{config.contextLength ?? "—"}</dd>
      </dl>
    </div>
  );
}

export function SettingsPanel({
  settings,
  connected,
  onLoadProviders,
  onSetKey,
  onLoadModels,
  onSelectModel,
  onSelectProfile,
  onApplyGeneration,
}: SettingsPanelProps) {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [keyDraft, setKeyDraft] = useState("");
  const [query, setQuery] = useState("");

  const provider = selectedProvider ?? settings.config?.provider ?? null;
  const activeModel = settings.config?.model ?? null;

  // Three clicks to see one model — Refresh, pick a provider, Search — is not a
  // settings screen, it is a scavenger hunt. Load on open instead.
  useEffect(() => {
    if (connected) {
      onLoadProviders();
    }
  }, [connected, onLoadProviders]);

  useEffect(() => {
    if (!connected || provider === null) {
      return;
    }
    const timer = window.setTimeout(() => onLoadModels(provider, query), 300);
    return () => window.clearTimeout(timer);
  }, [connected, provider, query, onLoadModels]);
  const models = provider !== null ? (settings.catalog[provider] ?? []) : [];
  const loading = provider !== null && settings.catalogLoading[provider] === true;
  const keyStatus = provider !== null ? settings.keyStatus[provider] : undefined;

  return (
    <div className="mx-auto flex w-full max-w-[46rem] flex-col gap-3">
      <PanelSection
        title="Active configuration"
        description={settings.config?.backend ?? "not loaded"}
        action={
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onLoadProviders}>
            Refresh
          </Button>
        }
      >
        <ActiveConfig settings={settings} />
      </PanelSection>

      <PanelSection
        title="Generation"
        description="Áp dụng qua llm_select, cùng đường với việc chọn model"
      >
        {settings.config === null ? (
          <PanelEmpty>Chưa nạp cấu hình.</PanelEmpty>
        ) : (
          <div className="flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="text-muted-foreground w-28 shrink-0 text-xs">backend</span>
              {(["local", "remote"] as const).map((backend) => (
                <Button
                  key={backend}
                  size="sm"
                  variant={settings.config?.backend === backend ? "default" : "outline"}
                  disabled={!connected}
                  onClick={() => onApplyGeneration({ backend })}
                >
                  {backend}
                </Button>
              ))}
            </div>

            <div className="flex items-center gap-2">
              <Label htmlFor="max-tokens" className="text-muted-foreground w-28 shrink-0 text-xs">
                max output
              </Label>
              <input
                id="max-tokens"
                type="number"
                min={1}
                className="border-input bg-background h-8 w-32 rounded-md border px-2 text-sm"
                defaultValue={settings.config.maxTokens ?? undefined}
                disabled={!connected}
                onBlur={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (Number.isFinite(value) && value > 0 && value !== settings.config?.maxTokens) {
                    onApplyGeneration({ maxTokens: value });
                  }
                }}
              />
              <span className="text-muted-foreground text-[10px]">
                hiệu lực {settings.config.effectiveMaxTokens ?? "—"}
              </span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-muted-foreground w-28 shrink-0 text-xs">reasoning</span>
              <Button
                size="sm"
                variant={settings.config.effectiveReasoningEnabled ? "default" : "outline"}
                // A model can force reasoning on or not support it at all; in
                // both cases the toggle is not the user's to flip (§7 obl. 8).
                disabled={
                  !connected ||
                  !settings.config.reasoningSupported ||
                  settings.config.reasoningMandatory
                }
                onClick={() =>
                  onApplyGeneration({
                    reasoningEnabled: !(settings.config?.reasoningEnabled ?? false),
                  })
                }
              >
                {settings.config.effectiveReasoningEnabled ? "bật" : "tắt"}
              </Button>
              <span className="text-muted-foreground text-[10px]">
                {reasoningSummary(settings.config)}
              </span>
            </div>
          </div>
        )}
      </PanelSection>

      <PanelSection title="Providers" description="Keys live in the engine keyring">
        <div className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            {settings.providers.length === 0 ? (
              <PanelEmpty>None loaded yet.</PanelEmpty>
            ) : (
              settings.providers.map((item) => (
                <Button
                  key={item.key}
                  size="sm"
                  variant={provider === item.key ? "default" : "outline"}
                  onClick={() => setSelectedProvider(item.key)}
                >
                  {item.label}
                  {item.hasKey && <Badge variant="secondary" className="ml-2">key</Badge>}
                </Button>
              ))
            )}
          </div>

          {provider !== null && (
            <>
              <div className="flex flex-col gap-1">
                <Label htmlFor="api-key">API key for {provider}</Label>
                <div className="flex gap-2">
                  <input
                    id="api-key"
                    type="password"
                    autoComplete="off"
                    className="border-input bg-background h-9 flex-1 rounded-md border px-3 text-sm"
                    value={keyDraft}
                    onChange={(event) => setKeyDraft(event.target.value)}
                    placeholder={keyStatus?.masked ?? "paste key"}
                  />
                  <Button
                    disabled={!connected || keyDraft.trim() === ""}
                    onClick={() => {
                      onSetKey(provider, keyDraft.trim());
                      // Cleared immediately — the engine's keyring owns it now.
                      setKeyDraft("");
                    }}
                  >
                    Save
                  </Button>
                </div>
                {keyStatus !== undefined && (
                  <p
                    className={
                      keyStatus.ok
                        ? "text-muted-foreground text-xs"
                        : "text-destructive text-xs"
                    }
                  >
                    {keyStatus.pending
                      ? "validating…"
                      : (keyStatus.message ?? (keyStatus.ok ? "key accepted" : "not validated"))}
                  </p>
                )}
              </div>

              <div className="flex gap-2">
                <input
                  className="border-input bg-background h-9 flex-1 rounded-md border px-3 text-sm"
                  value={query}
                  placeholder="filter models"
                  aria-label="Model filter"
                  onChange={(event) => setQuery(event.target.value)}
                />

              </div>

              <ScrollArea className="h-56">
                {loading ? (
                  /* docs/18 §4: the first frame is empty while the fetch runs. */
                  <PanelEmpty>Fetching catalog…</PanelEmpty>
                ) : models.length === 0 ? (
                  <PanelEmpty>No models.</PanelEmpty>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {models.map((model) => (
                      <li
                        key={model.id}
                        className={
                          model.id === activeModel
                            ? "bg-accent/60 flex items-center gap-2 rounded-md px-2 py-1 text-sm"
                            : "hover:bg-muted/50 flex items-center gap-2 rounded-md px-2 py-1 text-sm"
                        }
                      >
                        <div className="flex flex-1 flex-col">
                          <span className="font-mono text-xs">{model.id}</span>
                          <span className="text-muted-foreground text-[10px]">
                            {model.context_length ?? "—"} ctx
                            {modelPrice(model) !== null && ` · ${modelPrice(model)}`}
                            {model.reasoning_mandatory && " · reasoning required"}
                          </span>
                        </div>
                        {model.id === activeModel ? (
                          <Badge variant="secondary">đang dùng</Badge>
                        ) : (
                          <Button
                            size="sm"
                            variant="ghost"
                            disabled={!connected}
                            onClick={() => onSelectModel(provider, model.id)}
                          >
                            Chọn
                          </Button>
                        )}
                      </li>
                    ))}
                  </ul>
                )}
              </ScrollArea>
              {settings.pricingAsOf !== null && (
                <p className="text-muted-foreground text-[10px]">
                  Pricing table as of {settings.pricingAsOf}.
                </p>
              )}
            </>
          )}
        </div>
      </PanelSection>

      <PanelSection title="Runtime profiles" description="ASR, TTS and voice per profile">
        <div className="flex flex-col gap-2">
          {settings.profiles.length === 0 ? (
            <PanelEmpty>Refresh to load profiles.</PanelEmpty>
          ) : (
            settings.profiles.map((profile) => (
              <div key={profile.key} className="flex items-center gap-2 text-sm">
                <Badge variant={profile.status === "ok" ? "secondary" : "outline"}>
                  {profile.status}
                </Badge>
                <span className="font-mono text-xs">{profile.key}</span>
                <span className="text-muted-foreground flex-1 font-mono text-[10px]">
                  {[profile.asr, profile.tts, profile.voice].filter(Boolean).join(" · ")}
                </span>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={!connected || profile.status !== "ok"}
                  onClick={() => onSelectProfile(profile.key)}
                >
                  Use
                </Button>
              </div>
            ))
          )}
        </div>
      </PanelSection>
    </div>
  );
}
