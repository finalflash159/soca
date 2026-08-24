/** User-controlled appearance, session privacy, model, key, and voice settings. */

import { AudioLines, Cpu, HardDrive, KeyRound, Palette } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Field, Section } from "@/components/Page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { UpdaterPanel } from "@/components/UpdaterPanel";
import type { LlmConfig, RuntimeComponent, SettingsState } from "@/engine/settings";
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
    provider?: string;
    model?: string;
    maxTokens?: number;
    reasoningEnabled?: boolean;
  }) => Promise<boolean>;
  modelRoot: { path: string; source: "managed" | "external" } | null;
  /** Returns an error message instead of hiding a failed native configuration change. */
  onSetModelRoot: (path: string | null) => Promise<string | null>;
  engineError: string | null;
  sessionHistory: SessionHistoryState;
  persistenceChangePending: boolean;
  onRequestSessionPersistence: (enabled: boolean) => void;
  onSetAutoOpenLast: (enabled: boolean) => void;
  /** Set only when navigation redirected an unavailable microphone here. */
  focusVoiceSetup?: boolean;
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

function profileName(profileKey: string): string {
  switch (profileKey) {
    case "baseline":
      return "Standard voice";
    case "qwen-release":
      return "Qwen ASR — Release";
    case "qwen-reference":
      return "Qwen ASR — Reference";
    default:
      return profileKey;
  }
}

function profileStatus(profile: SettingsState["profiles"][number]): string {
  if (profile.key.startsWith("qwen-") && profile.status === "invalid") return "Not included";
  const { status } = profile;
  if (status === "ok") return "Ready";
  if (status === "missing" || status === "blocked") return "Needs setup";
  if (status === "invalid") return "Unavailable";
  return "Checking";
}

function voiceSetupMessage(component: RuntimeComponent | undefined): string {
  if (component === undefined) return "Đang kiểm tra các thành phần thoại…";
  switch (component.id) {
    case "voice_asr":
      return "Speech recognition is not installed for the selected voice profile.";
    case "voice_llm":
      return "The response model for voice needs setup.";
    case "tts":
      return "Speech output needs setup.";
    default:
      return "A required voice component needs setup.";
  }
}

function chatSetupMessage(config: LlmConfig): string {
  return config.backend === "remote"
    ? "The selected remote chat route needs setup."
    : "An on-device chat model needs setup.";
}

function profileSummary(profile: SettingsState["profiles"][number]): string {
  const summary =
    profile.key === "baseline"
      ? "Default on-device Vietnamese voice profile."
      : profile.key === "qwen-release"
        ? "Qwen ASR release profile for local speech recognition."
        : profile.key === "qwen-reference"
          ? "Qwen ASR reference profile for local speech recognition."
          : "Voice profile.";
  if (profile.status === "ok") return summary;
  if (profile.key.startsWith("qwen-") && profile.status === "invalid") {
    return `${summary} Qwen is not included in the desktop release. It requires a separately verified worker runtime and model artifact.`;
  }
  if (profile.status === "invalid") {
    return `${summary} This profile is unavailable because its configuration could not be validated.`;
  }
  return `${summary} Required model files need setup.`;
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
  modelRoot,
  onSetModelRoot,
  sessionHistory,
  persistenceChangePending,
  onRequestSessionPersistence,
  onSetAutoOpenLast,
  engineError,
  focusVoiceSetup = false,
}: SettingsPanelProps) {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [remoteSetup, setRemoteSetup] = useState(false);
  const [keyDraft, setKeyDraft] = useState("");
  const [query, setQuery] = useState("");
  const [maxTokensDraft, setMaxTokensDraft] = useState("");
  const [generationPending, setGenerationPending] = useState(false);
  const [generationError, setGenerationError] = useState<string | null>(null);
  const [profilePending, setProfilePending] = useState<string | null>(null);
  const [modelRootDraft, setModelRootDraft] = useState("");
  const [modelRootPending, setModelRootPending] = useState(false);
  const [modelRootError, setModelRootError] = useState<string | null>(null);
  const lastEngineError = useRef<string | null>(engineError);
  const voiceSetupRef = useRef<HTMLDivElement>(null);

  const config = settings.config;
  const provider =
    selectedProvider ?? config?.provider ?? settings.providers[0]?.key ?? null;
  const activeModel = config?.model ?? null;
  // A provider list and a hosted-model catalog are meaningless on the local
  // backend. During an explicit Local → Remote setup, show the controls before
  // committing the backend so the selection cannot inherit a local GGUF id.
  const isRemote = config?.backend === "remote";
  const showRemoteSettings = isRemote || remoteSetup;
  const chatComponent = settings.runtimeComponents.find((component) => component.id === "chat_llm");
  const voiceComponents = settings.runtimeComponents.filter((component) =>
    ["voice_asr", "voice_llm", "tts"].includes(component.id),
  );
  const voiceBlocker = voiceComponents.find(
    (component) => !["ready", "loaded", "configured"].includes(component.status),
  );
  const voiceReady = voiceComponents.length === 3 && voiceBlocker === undefined;

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
    if (!connected || provider === null || !showRemoteSettings) {
      return;
    }
    const timer = window.setTimeout(() => loadModelsRef.current(provider, query), 300);
    return () => window.clearTimeout(timer);
  }, [connected, provider, query, showRemoteSettings]);

  const models = provider !== null ? (settings.catalog[provider] ?? []) : [];
  const loading = provider !== null && settings.catalogLoading[provider] === true;
  const keyStatus = provider !== null ? settings.keyStatus[provider] : undefined;

  useEffect(() => {
    if (isRemote) setRemoteSetup(false);
  }, [isRemote]);

  useEffect(() => {
    setMaxTokensDraft(config?.maxTokens?.toString() ?? "");
    setGenerationPending(false);
    setGenerationError(config?.settingsError ?? null);
  }, [config?.backend, config?.provider, config?.model, config?.maxTokens, config?.settingsError]);

  useEffect(() => {
    setModelRootDraft(modelRoot?.path ?? "");
  }, [modelRoot?.path]);

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

  useEffect(() => {
    if (focusVoiceSetup) {
      voiceSetupRef.current?.scrollIntoView?.({ behavior: "smooth", block: "start" });
    }
  }, [focusVoiceSetup]);

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
        description="SoCa chỉ ghi nội dung phiên sau khi bạn đồng ý rõ ràng. Audio và ASR partial không được lưu."
      >
        <Field
          label="Lưu phiên trên máy"
          hint={
            sessionHistory.persistence === "local_resumable"
              ? "Đang lưu chat/voice dạng văn bản, context làm việc và trạng thái mục tiêu trong thư mục dữ liệu riêng của SoCa."
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
              ? "Chỉ mở lại nội dung đã hoàn tất; SoCa không tự chạy lại lượt, tool hay mic còn dang dở."
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
            <Field
              label="Trạng thái chat"
              hint="Kiểm tra riêng route chat đang chọn. Thiết lập thoại ở phần bên dưới không chặn chat từ xa."
            >
              <div className="border-border bg-muted/30 flex items-center gap-3 rounded-lg border px-3 py-2.5 text-sm">
                <Badge variant={config.runtimeReady ? "secondary" : "outline"}>
                  {config.runtimeReady ? "Ready" : "Needs setup"}
                </Badge>
                <div className="min-w-0 flex-1 text-xs leading-5">
                  {config.runtimeReady
                    ? isRemote
                      ? "Remote chat is configured."
                      : "On-device chat model is available."
                    : chatSetupMessage(config)}
                  {!config.runtimeReady &&
                    (config.runtimeReason ?? config.settingsError ?? chatComponent?.detail) !== undefined && (
                      <details className="text-muted-foreground mt-2 text-xs">
                        <summary className="cursor-pointer select-none">Thông tin kỹ thuật</summary>
                        <p className="mt-1 break-words font-mono leading-5">
                          {config.runtimeReason ?? config.settingsError ?? chatComponent?.detail}
                        </p>
                      </details>
                    )}
                </div>
              </div>
            </Field>

            <Field
              label="Nơi xử lý"
              hint={
                isRemote
                  ? "Remote API. Requests leave this device."
                  : "On-device. The model loads only when you use it."
              }
            >
              <Segmented
                value={showRemoteSettings ? "remote" : "local"}
                disabled={!connected || generationPending}
                onChange={(backend) => {
                  if (backend === "remote") {
                    setRemoteSetup(true);
                    setGenerationError(null);
                    loadProvidersRef.current();
                    return;
                  }
                  setRemoteSetup(false);
                  void runGeneration(() => onApplyGeneration({ backend: "local" }));
                }}
                options={[
                  { value: "remote", label: "Remote" },
                  { value: "local", label: "On-device" },
                ]}
              />
            </Field>

            <Field
              label="Model đang dùng"
              hint={
                isRemote
                  ? `Context window: ${config.contextLength ?? "—"}`
                  : config.localModelPath === null
                    ? "Model On-device được chọn bởi profile đang hoạt động."
                    : `Tệp đang được kiểm tra: ${config.localModelPath}`
              }
            >
              <div className="border-border bg-muted/40 flex h-10 items-center rounded-lg border px-3">
                <span className="truncate font-mono text-sm">
                  {remoteSetup && !isRemote ? "Chưa áp dụng — hãy chọn model remote bên dưới" : config.model}
                </span>
              </div>
            </Field>

            <Field
              label="Thư mục model Voice và On-device"
              hint={
                modelRoot?.source === "external"
                  ? "Đang dùng thư mục bạn đã chọn cho Voice và model On-device. Lưu thay đổi sẽ khởi động lại engine; dữ liệu không bị sao chép."
                  : "Kho riêng của SoCa hiện đang được dùng. Voice vẫn cần ASR và TTS ở đây, kể cả khi câu trả lời dùng Remote."
              }
              htmlFor="local-model-root"
            >
              <div className="flex gap-2">
                <input
                  id="local-model-root"
                  className={`${INPUT} font-mono text-xs`}
                  value={modelRootDraft}
                  disabled={!connected || modelRootPending}
                  placeholder="/đường/dẫn/tới/models"
                  onChange={(event) => setModelRootDraft(event.target.value)}
                />
                <Button
                  className="h-10 shrink-0"
                  disabled={!connected || modelRootPending || modelRootDraft.trim() === ""}
                  onClick={async () => {
                    setModelRootPending(true);
                    setModelRootError(null);
                    const error = await onSetModelRoot(modelRootDraft.trim());
                    setModelRootPending(false);
                    setModelRootError(error);
                  }}
                >
                  {modelRootPending ? "Đang áp dụng…" : "Dùng thư mục này"}
                </Button>
              </div>
              {modelRoot?.source === "external" && (
                <Button
                  className="mt-2"
                  size="sm"
                  variant="outline"
                  disabled={!connected || modelRootPending}
                  onClick={async () => {
                    setModelRootPending(true);
                    setModelRootError(null);
                    const error = await onSetModelRoot(null);
                    setModelRootPending(false);
                    setModelRootError(error);
                  }}
                >
                  Trở về kho SoCa
                </Button>
              )}
              {modelRootError !== null && (
                <p className="text-destructive mt-2 text-sm" role="alert">{modelRootError}</p>
              )}
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

      {showRemoteSettings && (
        <Section
          icon={KeyRound}
          title="Nhà cung cấp"
          description="Key nằm trong keyring của engine, không bao giờ lưu ở giao diện."
        >
          {remoteSetup && !isRemote && (
            <p className="bg-muted/40 text-muted-foreground rounded-lg border p-3 text-sm">
              Chọn provider, xác thực API key và chọn model. SoCa chỉ chuyển sang remote sau khi
              bạn chọn một model hợp lệ từ danh mục.
            </p>
          )}
          <Field label="Chọn nhà cung cấp" hint="Chấm tròn đánh dấu nơi đang chạy thật.">
            <div className="flex flex-wrap gap-2">
              {settings.providers.length === 0 ? (
                <p className="text-muted-foreground text-sm">Chưa nạp nhà cung cấp nào.</p>
              ) : (
                settings.providers.map((item) => {
                  const active = isRemote && config?.provider === item.key;
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

      <div ref={voiceSetupRef}>
        <Section
          icon={AudioLines}
          title="Thoại"
          description="Chat và Voice được thiết lập riêng. Mở Voice để xem trạng thái; dùng phần này để hoàn tất các thành phần còn thiếu."
        >
        <Field
          label="Trạng thái thoại"
          hint="Âm thanh luôn được thu và phát trên máy này."
        >
          {voiceComponents.length === 0 ? (
            <p className="text-muted-foreground text-sm">Đang kiểm tra các thành phần thoại…</p>
          ) : (
            <div className="border-border bg-muted/30 flex items-start gap-3 rounded-lg border px-3 py-2.5 text-sm">
              <Badge variant={voiceReady ? "secondary" : "outline"}>
                {voiceReady ? "Ready" : "Needs setup"}
              </Badge>
              <div className="min-w-0 flex-1 leading-5">
                {voiceReady
                  ? "Speech recognition, response model, and speech output are ready."
                  : voiceBlocker === undefined
                    ? "Đang chờ trạng thái thoại hoàn chỉnh…"
                    : voiceSetupMessage(voiceBlocker)}
                {!voiceReady && voiceBlocker?.detail !== null && voiceBlocker?.detail !== undefined && (
                  <details className="text-muted-foreground mt-2 text-xs">
                    <summary className="cursor-pointer select-none">Thông tin kỹ thuật</summary>
                    <p className="mt-1 break-words font-mono leading-5">{voiceBlocker.detail}</p>
                  </details>
                )}
                {!voiceReady && (
                  <Button
                    className="mt-3"
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      const input = document.getElementById("local-model-root");
                      input?.scrollIntoView({ behavior: "smooth", block: "center" });
                      window.setTimeout(() => (input as HTMLInputElement | null)?.focus(), 250);
                    }}
                  >
                    Choose a SoCa model folder
                  </Button>
                )}
              </div>
            </div>
          )}
        </Field>
        <Field
          label="Cấu hình thoại"
          hint="Profile chỉ áp dụng cho phiên thoại kế tiếp; chuyển profile không ngắt cuộc gọi đang diễn ra."
        >
          {settings.profiles.length === 0 ? (
            <p className="text-muted-foreground text-sm">Tải lại cài đặt để xem voice profile.</p>
          ) : (
            <ul className="border-border divide-border flex flex-col divide-y rounded-lg border">
              {settings.profiles.map((profile) => (
                <li key={profile.key} className="flex items-center gap-3 px-3 py-2.5 text-sm">
                  <Badge variant={settings.activeProfile === profile.key ? "secondary" : "outline"}>
                    {settings.activeProfile === profile.key
                      ? profile.status === "ok"
                        ? "Active"
                        : "Active · Needs setup"
                      : profileStatus(profile)}
                  </Badge>
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate text-sm font-medium">{profileName(profile.key)}</span>
                    <span className="text-muted-foreground text-xs leading-5">{profileSummary(profile)}</span>
                    <details className="text-muted-foreground mt-1 text-[10px]">
                      <summary className="cursor-pointer select-none">Thông tin kỹ thuật</summary>
                      {profile.note !== null && profile.note !== undefined && (
                        <p className="mt-1 break-words leading-5">{profile.note}</p>
                      )}
                      <p className="mt-1 break-words font-mono leading-5">
                        {[profile.asr, profile.tts, profile.voice].filter(Boolean).join(" · ")}
                      </p>
                    </details>
                  </div>
                  {settings.activeProfile !== profile.key && profile.status === "ok" && (
                    <Button
                      size="sm"
                      variant="ghost"
                      disabled={!connected || profile.status !== "ok" || profilePending !== null}
                      onClick={async () => {
                        setProfilePending(profile.key);
                        if (!(await onSelectProfile(profile.key))) setProfilePending(null);
                      }}
                    >
                      {profilePending === profile.key ? "Đang áp dụng…" : "Chọn"}
                    </Button>
                  )}
                </li>
              ))}
            </ul>
          )}
        </Field>
        </Section>
      </div>
    </div>
  );
}
