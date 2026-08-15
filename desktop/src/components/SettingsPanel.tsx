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

import { useState } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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
}

function ActiveConfig({ settings }: { settings: SettingsState }) {
  const config = settings.config;
  if (config === null) {
    return (
      <p className="text-muted-foreground text-sm">
        No configuration loaded yet.
      </p>
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
}: SettingsPanelProps) {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [keyDraft, setKeyDraft] = useState("");
  const [query, setQuery] = useState("");

  const provider = selectedProvider ?? settings.config?.provider ?? null;
  const models = provider !== null ? (settings.catalog[provider] ?? []) : [];
  const loading = provider !== null && settings.catalogLoading[provider] === true;
  const keyStatus = provider !== null ? settings.keyStatus[provider] : undefined;

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader className="flex-row items-center justify-between">
          <CardTitle className="text-base">Active configuration</CardTitle>
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onLoadProviders}>
            Load providers
          </Button>
        </CardHeader>
        <CardContent>
          <ActiveConfig settings={settings} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Providers</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-3">
          <div className="flex flex-wrap gap-2">
            {settings.providers.length === 0 ? (
              <p className="text-muted-foreground text-sm">None loaded.</p>
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
                <Button
                  variant="outline"
                  disabled={!connected}
                  onClick={() => onLoadModels(provider, query)}
                >
                  Search
                </Button>
              </div>

              <ScrollArea className="h-56">
                {loading ? (
                  // docs/18 §4: the first frame is empty while the fetch runs.
                  <p className="text-muted-foreground text-sm">Fetching catalog…</p>
                ) : models.length === 0 ? (
                  <p className="text-muted-foreground text-sm">No models.</p>
                ) : (
                  <ul className="flex flex-col gap-1">
                    {models.map((model) => (
                      <li
                        key={model.id}
                        className="hover:bg-muted/50 flex items-center gap-2 rounded-md px-2 py-1 text-sm"
                      >
                        <div className="flex flex-1 flex-col">
                          <span className="font-mono text-xs">{model.id}</span>
                          <span className="text-muted-foreground text-[10px]">
                            {model.context_length ?? "—"} ctx
                            {modelPrice(model) !== null && ` · ${modelPrice(model)}`}
                            {model.reasoning_mandatory && " · reasoning required"}
                          </span>
                        </div>
                        <Button
                          size="sm"
                          variant="ghost"
                          disabled={!connected}
                          onClick={() => onSelectModel(provider, model.id)}
                        >
                          Select
                        </Button>
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
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Runtime profiles</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-col gap-2">
          {settings.profiles.length === 0 ? (
            <p className="text-muted-foreground text-sm">Ask for status to load profiles.</p>
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
        </CardContent>
      </Card>
    </div>
  );
}
