/**
 * Settings.
 *
 * Form shape from the reference app's settings screen: grouped sections, each
 * with an icon and a sentence saying what the group is for, then fields with
 * the label above and a helper line under it. The previous version was
 * label-left / control-right rows, which physically had nowhere to put the
 * helper — so every non-obvious setting went unexplained.
 *
 * Ordered by how often it is touched, not by how the engine is built:
 * appearance, then the model, then keys, then voice.
 *
 * The key field is write-only. It is sent with `llm_set_key` and cleared
 * immediately; the engine holds it in its keyring and answers with at most a
 * masked form. Nothing here stores or renders a raw key.
 */

import { AudioLines, Cpu, KeyRound, Palette } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { Field, Section } from "@/components/Page";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import type { SettingsState } from "@/engine/settings";
import { modelPrice, reasoningSummary } from "@/engine/settings";
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
  onSelectModel: (provider: string, modelId: string) => void;
  onSelectProfile: (profileKey: string) => void;
  /** Backend, output cap and reasoning all travel on the same `llm_select`. */
  onApplyGeneration: (change: {
    backend?: string;
    maxTokens?: number;
    reasoningEnabled?: boolean;
  }) => void;
}

const INPUT =
  "border-input bg-background focus-visible:border-ring focus-visible:ring-ring/30 h-10 w-full rounded-lg border px-3 text-sm outline-none transition-colors focus-visible:ring-[3px] disabled:opacity-50";

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
}: SettingsPanelProps) {
  const [selectedProvider, setSelectedProvider] = useState<string | null>(null);
  const [keyDraft, setKeyDraft] = useState("");
  const [query, setQuery] = useState("");

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
                disabled={!connected}
                onChange={(backend) => onApplyGeneration({ backend })}
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
                defaultValue={config.maxTokens ?? undefined}
                disabled={!connected}
                onBlur={(event) => {
                  const value = Number.parseInt(event.target.value, 10);
                  if (Number.isFinite(value) && value > 0 && value !== config.maxTokens) {
                    onApplyGeneration({ maxTokens: value });
                  }
                }}
              />
            </Field>

            <Field label="Suy luận" hint={reasoningSummary(config)}>
              <Segmented
                value={config.effectiveReasoningEnabled ? "on" : "off"}
                // A model can force reasoning on or not support it at all; in
                // both cases the toggle is not the user's to flip (§7 obl. 8).
                disabled={!connected || !config.reasoningSupported || config.reasoningMandatory}
                onChange={(next) => onApplyGeneration({ reasoningEnabled: next === "on" })}
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
                  <Badge variant={profile.status === "ok" ? "secondary" : "outline"}>
                    {profile.status}
                  </Badge>
                  <div className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate font-mono text-xs">{profile.key}</span>
                    <span className="text-muted-foreground truncate font-mono text-[10px]">
                      {[profile.asr, profile.tts, profile.voice].filter(Boolean).join(" · ")}
                    </span>
                  </div>
                  <Button
                    size="sm"
                    variant="ghost"
                    disabled={!connected || profile.status !== "ok"}
                    onClick={() => onSelectProfile(profile.key)}
                  >
                    Dùng
                  </Button>
                </li>
              ))}
            </ul>
          )}
        </Field>
      </Section>
    </div>
  );
}
