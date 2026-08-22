/** User-controlled appearance, session privacy, model, key, and voice settings. */

import { AudioLines, Cpu, HardDrive, KeyRound, Palette } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Field, Section } from "@/components/Page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { UpdaterPanel } from "@/components/UpdaterPanel";
import type { LlmConfig, SettingsState } from "@/engine/settings";
import type { SessionHistoryState } from "@/engine/session-history";
import { modelPrice } from "@/engine/settings";
import type { ThemeChoice } from "@/theme";
import { cn } from "@/lib/utils";

interface SettingsPanelProps {
  settings: SettingsState;
  connected: boolean;
  themeChoice: ThemeChoice;
  onSetTheme: (choice: ThemeChoice) => void;
  onLoadProviders: () => void;
  onSetKey: (provider: string, key: string) => void;
  onLoadModels: (provider: string, query: string) => void;
  onSelectModel: (provider: string, modelId: string) => Promise<boolean>;
  onSelectProfile: (profileKey: string) => Promise<boolean>;
  /** Backend, output cap and reasoning all travel on the same `llm_select`. */
  onApplyGeneration: (change: {
    backend?: string;
    maxTokens?: number;
    reasoningEnabled?: boolean;
  }) => Promise<boolean>;
  engineError: string | null;
  sessionHistory: SessionHistoryState;
  persistenceChangePending: boolean;
  onRequestSessionPersistence: (enabled: boolean) => void;
  onSetAutoOpenLast: (enabled: boolean) => void;
}

const INPUT =
  "border-input bg-background focus-visible:border-ring focus-visible:ring-ring/30 h-10 w-full rounded-lg border px-3 text-sm outline-none transition-colors focus-visible:ring-[3px] disabled:opacity-50";

function reasoningHint(config: LlmConfig): string {
  if (!config.reasoningSupported) return "Model này không hỗ trợ suy luận.";
  if (config.reasoningMandatory) return "Model này luôn dùng suy luận; bạn không thể tắt.";
  return config.effectiveReasoningEnabled
    ? "Suy luận đang có hiệu lực cho các lượt mới."
    : "Suy luận đang tắt cho các lượt mới.";
}

/** A row of mutually exclusive choices, styled as one control. */
function Segmented<T extends string>({
  value,
  options,
  disabled,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  disabled?: boolean;
  onChange: (value: T) => void;
}) {
  return (
    <div className="bg-secondary inline-flex rounded-lg p-1">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          disabled={disabled}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
          className={cn(
            "rounded-md px-3 py-1.5 text-sm transition-colors disabled:opacity-50",
            value === option.value
              ? "bg-background text-foreground shadow-sm"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

export function SettingsPanel({
  settings,
  connected,
  themeChoice,
  onSetTheme,
  onLoadProviders,
  onSetKey,
  onLoadModels,
  onSelectModel,
  onSelectProfile,
  onApplyGeneration,
  sessionHistory,
  persistenceChangePending,
  onRequestSessionPersistence,
  onSetAutoOpenLast,
  engineError,
}: SettingsPanelProps) {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [keyDraft, setKeyDraft] = useState("");
  const [query, setQuery] = useState("");
  const [maxTokensDraft, setMaxTokensDraft] = useState("");
  const [generationPending, setGenerationPending] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [profilePending, setProfilePending] = useState<string | null>(null);
  const lastEngineError = useRef<string | null>(engineError);

  const config = settings.config;
  const provider = selectedProvider ?? config?.provider ?? null;
  const activeModel = config?.model ?? null;
  // A provider list and a hosted-model catalog are meaningless on the local
  // backend, and offering them there made "Máy này" look like a dead end.
  const isRemote = config?.backend !== "local";

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
    if (!connected || provider === null || !isRemote) {
      return;
    }
    const timer = window.setTimeout(() => loadModelsRef.current(provider, query), 300);
    return () => window.clearTimeout(timer);
  }, [connected, provider, query, isRemote]);

  const models = provider !== null ? (settings.catalog[provider] ?? []) : [];
  const loading = provider !== null && settings.catalogLoading[provider] === true;
  const keyStatus = provider !== null ? settings.keyStatus[provider] : undefined;

  useEffect(() => {
    setMaxTokensDraft(config?.maxTokens?.toString() ?? "");
    setGenerationPending(false);
    setGenerationError(config?.settingsError ?? null);
  }, [config?.backend, config?.provider, config?.model, config?.maxTokens, config?.settingsError]);

  useEffect(() => {
    if (engineError === lastEngineError.current) return;
    lastEngineError.current = engineError;
    if (engineError !== null && generationPending) {
      setGenerationPending(false);
      setGenerationError(engineError);
    }
    if (engineError !== null && profilePending !== null) {
      setProfilePending(null);
    }
  }, [engineError, generationPending, profilePending]);

  useEffect(() => {
    if (profilePending !== null && settings.activeProfile === profilePending) {
      setProfilePending(null);
    }
  }, [profilePending, settings.activeProfile]);

  const runGeneration = async (operation: () => Promise<boolean>) => {
    setGenerationPending(true);
    setGenerationError(null);
    const accepted = await operation();
    if (!accepted) {
      setGenerationPending(false);
      setGenerationError("Không thể gửi thay đổi tới engine. Giá trị có hiệu lực không đổi.");
    }
  };

  return (
    <div className="divide-border flex flex-col divide-y">
      <Section
        icon={Palette}
        title="Giao diện"
        description="Áp dụng ngay, và được nhớ cho lần mở sau."
      >
        <Field label="Chế độ màu" hint="“Theo hệ thống” đổi theo cài đặt sáng/tối của máy.">
          <Segmented
            value={themeChoice}
            onChange={onSetTheme}
            options={[
              { value: "system", label: "Theo hệ thống" },
              { value: "light", label: "Sáng" },
              { value: "dark", label: "Tối" },
            ]}
          />
        </Field>
      </Section>

      <Section
        icon={HardDrive}
        title="Phiên trên máy"
        description="Sơn Ca chỉ ghi nội dung phiên sau khi bạn đồng ý rõ ràng. Audio và ASR partial không được lưu."
      >
        <Field
          label="Lưu phiên trên máy"
          hint={
            sessionHistory.persistence === "local_resumable"
              ? "Đang lưu chat/voice dạng văn bản, context làm việc và trạng thái mục tiêu trong thư mục dữ liệu riêng của Sơn Ca."
              : "Phiên hiện tại chỉ ở trong bộ nhớ và sẽ không được ghi lại khi đóng ứng dụng."
          }
        >
          <div className="border-border flex items-center justify-between gap-4 rounded-lg border p-3">
            <span className="text-sm font-medium">
              {sessionHistory.persistence === "local_resumable" ? "Đang bật" : "Đang tắt"}
            </span>
            <Button
              size="sm"
              variant={sessionHistory.persistence === "local_resumable" ? "outline" : "default"}
              disabled={!connected || sessionHistory.persistence === null || persistenceChangePending}
              onClick={() => onRequestSessionPersistence(sessionHistory.persistence !== "local_resumable")}
            >
              {persistenceChangePending
                ? "Đang khởi động lại…"
                : sessionHistory.persistence === "local_resumable"
                  ? "Tắt lưu phiên"
                  : "Bật lưu phiên"}
            </Button>
          </div>
        </Field>

        <Field
          label="Mở lại phiên gần nhất khi khởi động"
          hint={
            sessionHistory.persistence === "local_resumable"
              ? "Chỉ mở lại nội dung đã hoàn tất; Sơn Ca không tự chạy lại lượt, tool hay mic còn dang dở."
              : "Bật lưu phiên trên máy trước khi dùng tùy chọn này."
          }
          htmlFor="auto-open-last-session"
        >
          <label className="border-border flex cursor-pointer items-center justify-between gap-4 rounded-lg border p-3" htmlFor="auto-open-last-session">
            <span className="text-sm">Tự mở phiên gần nhất</span>
            <input
              id="auto-open-last-session"
              type="checkbox"
              checked={sessionHistory.autoOpenLast}
              disabled={
                !connected ||
                sessionHistory.persistence !== "local_resumable" ||
                sessionHistory.operation?.action === "preferences_set" &&
                  sessionHistory.operation.status === "started"
              }
              onChange={(event) => onSetAutoOpenLast(event.target.checked)}
              className="accent-primary size-4"
            />
          </label>
        </Field>
      </Section>

      <UpdaterPanel />

      <Section
        icon={Cpu}
        title="Mô hình"
        description="Mọi thay đổi ở đây đi qua cùng một lệnh llm_select của engine."
        actions={
          <Button size="sm" variant="outline" disabled={!connected} onClick={onLoadProviders}>
            Tải lại
          </Button>
        }
      >
        {config === null ? (
          <p className="text-muted-foreground text-sm">Chưa nạp cấu hình.</p>
        ) : (
          <>
            {generationError !== null && (
              <p id="generation-error" className="text-destructive text-sm" role="alert">
                {generationError}
              </p>
            )}
            {!config.runtimeReady && (
              <p className="text-destructive text-sm">
                Runtime chưa sẵn sàng
                {config.settingsError !== null && ` · ${config.settingsError}`}
              </p>
            )}

            <Field
              label="Nguồn"
              hint={
                isRemote
                  ? "Gọi API bên ngoài. Câu hỏi rời khỏi máy này."
                  : "Chạy trên máy này. Model chỉ nạp khi dùng lần đầu, không nạp lúc khởi động."
              }
            >
              <Segmented
                value={isRemote ? "remote" : "local"}
                disabled={!connected || generationPending}
                onChange={(backend) => void runGeneration(() => onApplyGeneration({ backend }))}
                options={[
                  { value: "remote", label: "Từ xa" },
                  { value: "local", label: "Máy này" },
                ]}
              />
            </Field>

            <Field
              label="Model đang dùng"
              hint={
                isRemote
                  ? `Qua ${config.provider} · cửa sổ ngữ cảnh ${config.contextLength ?? "—"}`
                  : "Lấy theo profile bên dưới — protocol chưa có lệnh liệt kê model cục bộ."
              }
            >
              <div className="border-border bg-muted/40 flex h-10 items-center rounded-lg border px-3">
                <span className="truncate font-mono text-sm">{config.model}</span>
              </div>
            </Field>

            <Field
              label="Giới hạn token đầu ra"
              // §7 obligation 8: the effective value, not the request.
              hint={
                config.maxTokens !== null && config.maxTokens !== config.effectiveMaxTokens
                  ? `Model chặn ở ${config.effectiveMaxTokens ?? "—"}, nên đó mới là con số có hiệu lực.`
                  : "Số token tối đa cho một câu trả lời."
              }
              htmlFor="max-tokens"
            >
              <input
                id="max-tokens"
                type="number"
                min={1}
                className={INPUT}
                value={maxTokensDraft}
                disabled={!connected || generationPending}
                aria-invalid={generationError !== null}
                aria-describedby={generationError !== null ? "generation-error" : undefined}
                onChange={(event) => setMaxTokensDraft(event.target.value)}
                onBlur={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (event.target.value.trim() === "" || !Number.isInteger(value) || value < 1) {
                    setGenerationError("Nhập số nguyên dương cho giới hạn token.");
                    return;
                  }
                  if (value !== config.maxTokens) {
                    void runGeneration(() => onApplyGeneration({ maxTokens: value }));
                  }
                }}
                onKeyDown={(event) => {
                  if (event.key === "Enter") event.currentTarget.blur();
                }}
              />
            </Field>

            <Field label="Suy luận" hint={reasoningHint(config)}>
              <Segmented
                value={config.effectiveReasoningEnabled ? "on" : "off"}
                // A model can force reasoning on or not support it at all; in
                // both cases the toggle is not the user's to flip (§7 obl. 8).
                disabled={!connected || generationPending || !config.reasoningSupported || config.reasoningMandatory}
                onChange={(next) =>
                  void runGeneration(() => onApplyGeneration({ reasoningEnabled: next === "on" }))
                }
                options={[
                  { value: "off", label: "Tắt" },
                  { value: "on", label: "Bật" },
                ]}
              />
            </Field>
          </>
        )}
      </Section>

      {isRemote && (
        <Section
          icon={KeyRound}
          title="Nhà cung cấp"
          description="Key nằm trong keyring của engine, không bao giờ lưu ở giao diện."
        >
          <Field label="Chọn nhà cung cấp" hint="Chấm tròn đánh dấu nơi đang chạy thật.">
            <div className="flex flex-wrap gap-2">
              {settings.providers.length === 0 ? (
                <p className="text-muted-foreground text-sm">Chưa nạp nhà cung cấp nào.</p>
              ) : (
                settings.providers.map((item) => {
                  const active = config?.provider === item.key;
                  return (
                    <Button
                      key={item.key}
                      size="sm"
                      variant={provider === item.key ? "default" : "outline"}
                      className={active ? "ring-primary/60 ring-1" : undefined}
                      onClick={() => setSelectedProvider(item.key)}
                    >
                      {item.label}
                      {active && <span className="ml-1.5 text-[10px]">●</span>}
                      {!item.hasKey && (
                        <span className="ml-1.5 text-[10px] opacity-70">chưa có key</span>
                      )}
                    </Button>
                  );
                })
              )}
            </div>
          </Field>

          {provider !== null && (
            <>
              <Field
                label={`API key cho ${provider}`}
                hint={
                  keyStatus === undefined
                    ? "Dán key vào rồi bấm Lưu; giao diện xoá nó ngay sau khi gửi."
                    : keyStatus.pending
                      ? "Đang kiểm tra…"
                      : (keyStatus.message ?? (keyStatus.ok ? "Key hợp lệ." : "Chưa kiểm tra."))
                }
                htmlFor="api-key"
              >
                <div className="flex gap-2">
                  <input
                    id="api-key"
                    type="password"
                    autoComplete="off"
                    className={INPUT}
                    value={keyDraft}
                    onChange={(event) => setKeyDraft(event.target.value)}
                    placeholder={keyStatus?.masked ?? "dán key vào đây"}
                  />
                  <Button
                    className="h-10 shrink-0"
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
              </Field>

              <Field
                label="Danh mục model"
                hint={
                  settings.pricingAsOf !== null
                    ? `Bảng giá tính đến ${settings.pricingAsOf}.`
                    : "Gõ để lọc theo tên."
                }
                htmlFor="model-filter"
              >
                <input
                  id="model-filter"
                  className={INPUT}
                  value={query}
                  placeholder="lọc model"
                  onChange={(event) => setQuery(event.target.value)}
                />
                <ScrollArea className="border-border mt-2 h-64 rounded-lg border">
                  {loading ? (
                    /* docs/18 §4: the first frame is empty while the fetch runs. */
                    <p className="text-muted-foreground p-3 text-sm">Đang tải danh mục…</p>
                  ) : models.length === 0 ? (
                    <p className="text-muted-foreground p-3 text-sm">Không có model nào khớp.</p>
                  ) : (
                    <ul className="flex flex-col p-1">
                      {models.map((model) => (
                        <li
                          key={model.id}
                          className={cn(
                            "flex items-center gap-2 rounded-md px-2 py-1.5 text-sm",
                            model.id === activeModel ? "bg-accent" : "hover:bg-muted",
                          )}
                        >
                          <div className="flex min-w-0 flex-1 flex-col">
                            <span className="truncate font-mono text-xs">{model.id}</span>
                            <span className="text-muted-foreground text-[10px]">
                              {model.context_length ?? "—"} ctx
                              {modelPrice(model) !== null && ` · ${modelPrice(model)}`}
                              {model.reasoning_mandatory && " · bắt buộc suy luận"}
                            </span>
                          </div>
                          {model.id === activeModel ? (
                            <Badge variant="secondary">đang dùng</Badge>
                          ) : (
                            <Button
                              size="sm"
                              variant="ghost"
                              disabled={!connected || generationPending}
                              onClick={() => void runGeneration(() => onSelectModel(provider, model.id))}
                            >
                              Chọn
                            </Button>
                          )}
                        </li>
                      ))}
                    </ul>
                  )}
                </ScrollArea>
              </Field>
            </>
          )}
        </Section>
      )}

      <Section
        icon={AudioLines}
        title="Thoại"
        description="Mỗi profile là một bộ ASR, TTS và giọng đọc đi liền nhau."
      >
        <Field
          label="Profile"
          hint="Đổi profile áp dụng cho lượt thoại tiếp theo, không cắt lượt đang chạy."
        >
          {settings.profiles.length === 0 ? (
            <p className="text-muted-foreground text-sm">Bấm “Tải lại” ở trên để nạp profile.</p>
          ) : (
            <ul className="border-border divide-border flex flex-col divide-y rounded-lg border">
              {settings.profiles.map((profile) => (
                <li key={profile.key} className="flex items-center gap-3 px-3 py-2.5 text-sm">
                  <Badge variant={settings.activeProfile === profile.key ? "secondary" : "outline"}>
                    {settings.activeProfile === profile.key ? "đang dùng" : profile.status}
                  </Badge>
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate font-mono text-xs">{profile.key}</span>
                    <span className="text-muted-foreground truncate font-mono text-[10px]">
                      {[profile.asr, profile.tts, profile.voice].filter(Boolean).join(" · ")}
                    </span>
                  </div>
                  {settings.activeProfile !== profile.key && (
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={!connected || profile.status !== "ok" || profilePending !== null}
                      onClick={async () => {
                        setProfilePending(profile.key);
                        if (!(await onSelectProfile(profile.key))) setProfilePending(null);
                      }}
                    >
                      {profilePending === profile.key ? "Đang áp dụng…" : "Dùng"}
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Field>
      </Section>
    </div>
  );
}
