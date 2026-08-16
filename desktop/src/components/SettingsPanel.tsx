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

import { useEffect, useRef, useState } from "react";

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

/**
 * One row: the current value *is* the control.
 *
 * There used to be two sections here — a read-only list of backend / model /
 * max output / reasoning, and then a second section with buttons for the same
 * four things. Two places showing one fact is how they end up disagreeing, and
 * it doubled the height of the screen for nothing.
 */
function Row({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex min-h-8 items-center gap-3">
      <span className="text-muted-foreground w-24 shrink-0 text-xs">{label}</span>
      <div className="flex flex-1 flex-wrap items-center gap-2">{children}</div>
      {hint !== undefined && (
        <span className="text-muted-foreground shrink-0 text-[10px]">{hint}</span>
      )}
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
  // A provider list and a hosted-model catalog are meaningless on the local
  // backend, and offering them there made "Máy này" look like a dead end.
  const isRemote = settings.config?.backend !== "local";

  // The callbacks arrive as fresh closures on every render. Depending on their
  // identity would re-run these effects each time, each run sending a command,
  // each command changing state, each change re-rendering — a loop that floods
  // the engine and leaves the panel looking dead. Hold them in refs and depend
  // only on the values that should actually trigger a fetch.
  const loadProvidersRef = useRef(onLoadProviders);
  const loadModelsRef = useRef(onLoadModels);
  loadProvidersRef.current = onLoadProviders;
  loadModelsRef.current = onLoadModels;

  // Three clicks to see one model — Refresh, pick a provider, Search — is not a
  // settings screen, it is a scavenger hunt. Load on open instead.
  useEffect(() => {
    if (connected) {
      loadProvidersRef.current();
    }
  }, [connected]);

  useEffect(() => {
    if (!connected || provider === null) {
      return;
    }
    const timer = window.setTimeout(() => loadModelsRef.current(provider, query), 300);
    return () => window.clearTimeout(timer);
  }, [connected, provider, query]);
  const models = provider !== null ? (settings.catalog[provider] ?? []) : [];
  const loading = provider !== null && settings.catalogLoading[provider] === true;
  const keyStatus = provider !== null ? settings.keyStatus[provider] : undefined;

  return (
    <div className="mx-auto flex w-full max-w-[46rem] flex-col gap-3">
      <PanelSection
        title="Đang chạy"
        description="Mọi thay đổi ở đây đi qua cùng một lệnh llm_select"
        action={
          <Button size="sm" variant="ghost" disabled={!connected} onClick={onLoadProviders}>
            Tải lại
          </Button>
        }
      >
        {settings.config === null ? (
          <PanelEmpty>Chưa nạp cấu hình.</PanelEmpty>
        ) : (
          <div className="flex flex-col gap-2">
            {!settings.config.runtimeReady && (
              <p className="text-destructive text-sm">
                Runtime chưa sẵn sàng
                {settings.config.settingsError !== null && ` · ${settings.config.settingsError}`}
              </p>
            )}

            <Row label="nguồn">
              {(["remote", "local"] as const).map((backend) => (
                <Button
                  key={backend}
                  size="sm"
                  variant={settings.config?.backend === backend ? "default" : "outline"}
                  disabled={!connected}
                  onClick={() => onApplyGeneration({ backend })}
                >
                  {backend === "remote" ? "Từ xa" : "Máy này"}
                </Button>
              ))}
            </Row>

            <Row label="model" hint={`${settings.config.contextLength ?? "—"} ctx`}>
              <span className="truncate font-mono text-xs">{settings.config.model}</span>
              {isRemote && (
                <span className="text-muted-foreground text-[10px]">
                  qua {settings.config.provider}
                </span>
              )}
            </Row>

            <Row
              label="giới hạn ra"
              // §7 obligation 8: the effective value, not the request.
              hint={
                settings.config.maxTokens !== null &&
                settings.config.maxTokens !== settings.config.effectiveMaxTokens
                  ? `hiệu lực ${settings.config.effectiveMaxTokens ?? "—"} — model chặn`
                  : "token"
              }
            >
              <Label htmlFor="max-tokens" className="sr-only">
                Giới hạn token đầu ra
              </Label>
              <input
                id="max-tokens"
                type="number"
                min={1}
                className="border-input bg-background h-8 w-28 rounded-md border px-2 text-sm"
                defaultValue={settings.config.maxTokens ?? undefined}
                disabled={!connected}
                onBlur={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (Number.isFinite(value) && value > 0 && value !== settings.config?.maxTokens) {
                    onApplyGeneration({ maxTokens: value });
                  }
                }}
              />
            </Row>

            <Row label="suy luận" hint={reasoningSummary(settings.config)}>
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
            </Row>
          </div>
        )}
      </PanelSection>

      {!isRemote ? (
        <PanelSection
          title="Model trên máy"
          description="Nạp khi dùng lần đầu, không nạp lúc khởi động"
        >
          <PanelEmpty>
            Model chạy máy này lấy theo profile đang chọn, không chọn riêng ở đây — protocol chưa có
            lệnh liệt kê model cục bộ. Đổi ở mục “Profile” bên dưới. Chuyển sang “Từ xa” sẽ giải
            phóng model đang nạp.
          </PanelEmpty>
        </PanelSection>
      ) : (
        <PanelSection title="Nhà cung cấp" description="Key nằm trong keyring của engine">
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-2">
              {settings.providers.length === 0 ? (
                <PanelEmpty>Chưa nạp nhà cung cấp nào.</PanelEmpty>
              ) : (
                settings.providers.map((item) => {
                  const active = settings.config?.provider === item.key;
                  return (
                    <Button
                      key={item.key}
                      size="sm"
                      variant={provider === item.key ? "default" : "outline"}
                      // Browsing a provider and running on it are different
                      // states; the ring marks the one actually in use.
                      className={active ? "ring-primary/60 ring-1" : undefined}
                      onClick={() => setSelectedProvider(item.key)}
                    >
                      {item.label}
                      {active && <span className="ml-1.5 text-[10px]">●</span>}
                      {!item.hasKey && (
                        <span className="text-muted-foreground ml-1.5 text-[10px]">
                          chưa có key
                        </span>
                      )}
                    </Button>
                  );
                })
              )}
            </div>

            {provider !== null && (
              <>
                <div className="flex flex-col gap-1">
                  <Label htmlFor="api-key">API key cho {provider}</Label>
                  <div className="flex gap-2">
                    <input
                      id="api-key"
                      type="password"
                      autoComplete="off"
                      className="border-input bg-background h-9 flex-1 rounded-md border px-3 text-sm"
                      value={keyDraft}
                      onChange={(event) => setKeyDraft(event.target.value)}
                      placeholder={keyStatus?.masked ?? "dán key vào đây"}
                    />
                    <Button
                      disabled={!connected || keyDraft.trim() === ""}
                      onClick={() => {
                        onSetKey(provider, keyDraft.trim());
                        // Cleared immediately — the engine's keyring owns it now.
                        setKeyDraft("");
                      }}
                    >
                      Lưu
                    </Button>
                  </div>
                  {keyStatus !== undefined && (
                    <p
                      className={
                        keyStatus.ok ? "text-muted-foreground text-xs" : "text-destructive text-xs"
                      }
                    >
                      {keyStatus.pending
                        ? "đang kiểm tra…"
                        : (keyStatus.message ?? (keyStatus.ok ? "key hợp lệ" : "chưa kiểm tra"))}
                    </p>
                  )}
                </div>

                <div className="flex gap-2">
                  <input
                    className="border-input bg-background h-9 rounded-md border px-3 text-sm"
                    value={query}
                    placeholder="lọc model"
                    aria-label="Lọc model"
                    onChange={(event) => setQuery(event.target.value)}
                  />
                </div>

                <ScrollArea className="h-56">
                  {loading ? (
                    /* docs/18 §4: the first frame is empty while the fetch runs. */
                    <PanelEmpty>Đang tải danh mục…</PanelEmpty>
                  ) : models.length === 0 ? (
                    <PanelEmpty>Không có model nào khớp.</PanelEmpty>
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
                    Bảng giá tính đến {settings.pricingAsOf}.
                  </p>
                )}
              </>
            )}
          </div>
        </PanelSection>
      )}

      <PanelSection title="Profile" description="Mỗi profile là một bộ ASR, TTS và voice">
        <div className="flex flex-col gap-2">
          {settings.profiles.length === 0 ? (
            <PanelEmpty>Bấm “Tải lại” để nạp profile.</PanelEmpty>
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
                  Dùng
                </Button>
              </div>
            ))
          )}
        </div>
      </PanelSection>
    </div>
  );
}
