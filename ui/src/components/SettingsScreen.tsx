import { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput, useStdin, useStdout } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import type { LlmConfigEvent, RemoteModelEvent } from "../protocol.js";
import type { LlmProviderStatus } from "../store.js";
import { COLOR, ICON } from "../theme.js";
import { Panel } from "./Primitives.js";

type FocusTarget =
  | "resume"
  | "providers"
  | "key"
  | "search"
  | "models"
  | "maxTokens"
  | "reasoning";

interface ProviderChoice {
  key: string;
  label: string;
  hasKey: boolean;
  hasPricingApi: boolean;
}

interface PendingSelection {
  backend: "local" | "remote";
  provider: string;
  model: string;
  info: RemoteModelEvent | null;
}

export interface SettingsScreenProps {
  config: LlmConfigEvent | null;
  providers: LlmProviderStatus[];
  catalog: RemoteModelEvent[];
  catalogProvider: string;
  keyPendingProvider: string | null;
  notice: string;
  onRequestModels: (provider: string, query: string) => void;
  onSetKey: (provider: string, key: string) => void;
  onSelect: (selection: {
    backend: "local" | "remote";
    provider: string;
    model: string;
    max_tokens: number;
    reasoning_enabled: boolean;
  }) => void;
  onExit: () => void;
}

const LOCAL_FALLBACK_MODEL = "arcee_vylinh_3b_q4_k_m";
export const MIN_MAX_TOKENS = 2_048;
export const MAX_MAX_TOKENS = 500_000;

function choicesFrom(providers: LlmProviderStatus[]): ProviderChoice[] {
  return [
    { key: "local", label: "Local", hasKey: true, hasPricingApi: false },
    ...providers.map((provider) => ({
      key: provider.key,
      label: provider.label,
      hasKey: provider.has_key,
      hasPricingApi: provider.has_pricing_api,
    })),
  ];
}

export function filterCatalog(
  catalog: RemoteModelEvent[],
  query: string,
): RemoteModelEvent[] {
  const tokens = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (tokens.length === 0) return catalog;
  return catalog.filter((model) => {
    const haystack = `${model.id} ${model.label}`.toLowerCase();
    return tokens.every((token) => haystack.includes(token));
  });
}

function formatContext(contextLength: number | null): string {
  if (contextLength === null) return "context: ?";
  if (contextLength >= 1_000)
    return `context: ${Math.round(contextLength / 1_000)}k`;
  return `context: ${contextLength}`;
}

function formatPrice(model: RemoteModelEvent): string {
  const { price_prompt_per_1m: prompt, price_completion_per_1m: completion } =
    model;
  if (prompt === null || completion === null) return "giá: không rõ";
  return `$${prompt.toFixed(2)} / $${completion.toFixed(2)} / 1M`;
}

function modelItems(catalog: RemoteModelEvent[]): Array<{
  key: string;
  label: string;
  value: RemoteModelEvent;
}> {
  return catalog.map((model) => ({
    key: model.id,
    label: `${model.id} · ${formatContext(model.context_length)} · ${formatPrice(model)} [${model.pricing_source}]`,
    value: model,
  }));
}

function providerLabel(
  providerKey: string,
  providers: LlmProviderStatus[],
): string {
  if (providerKey === "local") return "Local";
  return (
    providers.find((provider) => provider.key === providerKey)?.label ??
    providerKey
  );
}

export function validateMaxTokens(raw: string): string {
  if (!raw) return "Nhập số token đầu ra.";
  const value = Number(raw);
  if (!Number.isSafeInteger(value))
    return "Giá trị token không hợp lệ.";
  if (value < MIN_MAX_TOKENS)
    return `Tối thiểu ${MIN_MAX_TOKENS.toLocaleString("vi-VN")} token.`;
  if (value > MAX_MAX_TOKENS)
    return `Tối đa ${MAX_MAX_TOKENS.toLocaleString("vi-VN")} token.`;
  return "";
}

function savedSummary(
  config: LlmConfigEvent,
  providers: LlmProviderStatus[],
): string {
  const provider =
    config.backend === "local"
      ? "Local"
      : providerLabel(config.provider, providers);
  const effectiveReasoning = config.effective_reasoning_enabled ?? null;
  const desiredReasoning = config.reasoning_enabled ?? false;
  const effectiveMax = config.effective_max_tokens ?? config.max_tokens;
  const reasoning =
    effectiveReasoning === null
      ? `reasoning theo model (đã chọn ${desiredReasoning ? "bật" : "tắt"})`
      : `reasoning ${effectiveReasoning ? "bật" : "tắt"}`;
  const output =
    effectiveMax < config.max_tokens
      ? `${config.max_tokens.toLocaleString("vi-VN")} → ${effectiveMax.toLocaleString("vi-VN")} output tok`
      : `${config.max_tokens.toLocaleString("vi-VN")} output tok`;
  return `${provider} · ${config.model} · ${output} · ${reasoning}`;
}

export function SettingsScreen({
  config,
  providers,
  catalog,
  catalogProvider,
  keyPendingProvider,
  notice,
  onRequestModels,
  onSetKey,
  onSelect,
  onExit,
}: SettingsScreenProps) {
  const rawInput = Boolean(useStdin().isRawModeSupported);
  const { stdout } = useStdout();
  const panelWidth = Math.max(24, Math.min(88, (stdout?.columns ?? 80) - 2));
  const configProvider =
    config?.backend === "remote" ? config.provider : "local";
  const [selectedProviderKey, setSelectedProviderKey] =
    useState(configProvider);
  const [focus, setFocus] = useState<FocusTarget>("providers");
  const [apiKey, setApiKey] = useState("");
  const [query, setQuery] = useState("");
  const [pending, setPending] = useState<PendingSelection | null>(null);
  const [maxTokens, setMaxTokens] = useState("4096");
  const [reasoningEnabled, setReasoningEnabled] = useState(false);
  const touched = useRef(false);

  useEffect(() => {
    if (touched.current || !config) return;
    setSelectedProviderKey(
      config.backend === "remote" ? config.provider : "local",
    );
    setMaxTokens(String(config.max_tokens));
    setReasoningEnabled(config.reasoning_enabled ?? false);
    setFocus("resume");
  }, [config]);

  const choices = choicesFrom(providers);
  const selectedIndex = Math.max(
    0,
    choices.findIndex((choice) => choice.key === selectedProviderKey),
  );
  const selectedProvider = choices[selectedIndex] ?? choices[0];
  const isLocal = selectedProvider?.key === "local";
  const keyPending = selectedProvider?.key === keyPendingProvider;
  const visibleCatalog =
    selectedProvider?.key === catalogProvider ? catalog : [];
  const filtered = useMemo(
    () => filterCatalog(visibleCatalog, query),
    [visibleCatalog, query],
  );
  const items = modelItems(filtered);
  const showKeyInput = !selectedProvider?.hasKey || focus === "key";
  const browsing = focus === "search" || focus === "models";
  const showModels = browsing || visibleCatalog.length > 0;
  const editingGeneration = focus === "maxTokens" || focus === "reasoning";
  const validationError = validateMaxTokens(maxTokens);

  const pendingMax = pending?.info?.max_output_tokens ?? null;
  const parsedMax = validationError ? null : Number(maxTokens);
  const effectiveMax =
    parsedMax === null
      ? null
      : pendingMax === null
        ? parsedMax
        : Math.min(parsedMax, pendingMax);
  const pendingReasoningSupported =
    pending?.info?.reasoning_supported ??
    (config &&
    pending?.backend === config.backend &&
    pending?.model === config.model
      ? (config.reasoning_supported ?? null)
      : null);
  const pendingReasoningMandatory =
    pending?.info?.reasoning_mandatory ??
    (config &&
    pending?.backend === config.backend &&
    pending?.model === config.model
      ? (config.reasoning_mandatory ?? false)
      : false);
  const effectiveReasoning = pendingReasoningMandatory
    ? true
    : pendingReasoningSupported === true
      ? reasoningEnabled
      : pendingReasoningSupported === false
        ? false
        : null;

  const prevHasKey = useRef(Boolean(selectedProvider?.hasKey));
  useEffect(() => {
    const hasKey = Boolean(selectedProvider?.hasKey);
    if (focus === "key" && hasKey && !prevHasKey.current) {
      setApiKey("");
      setFocus("providers");
    }
    prevHasKey.current = hasKey;
  }, [selectedProvider?.hasKey, focus]);

  const wasKeyPending = useRef(false);
  useEffect(() => {
    if (
      wasKeyPending.current &&
      !keyPending &&
      notice.startsWith("API key đã")
    ) {
      setApiKey("");
      setFocus("providers");
    }
    wasKeyPending.current = keyPending;
  }, [keyPending, notice]);

  const clearKeyOnDelete = useRef(false);
  const clearMaxOnDelete = useRef(false);

  function moveSelection(delta: number): void {
    touched.current = true;
    const next =
      choices[(selectedIndex + delta + choices.length) % choices.length];
    if (!next) return;
    setSelectedProviderKey(next.key);
    setApiKey("");
    setQuery("");
  }

  function beginGeneration(selection: PendingSelection): void {
    setPending(selection);
    setMaxTokens(String(config?.max_tokens ?? 4096));
    setReasoningEnabled(config?.reasoning_enabled ?? false);
    setFocus("maxTokens");
  }

  function descend(): void {
    if (!selectedProvider) return;
    if (selectedProvider.key === "local") {
      beginGeneration({
        backend: "local",
        provider: config?.provider ?? "openrouter",
        model:
          config?.backend === "local" ? config.model : LOCAL_FALLBACK_MODEL,
        info: null,
      });
      return;
    }
    if (!selectedProvider.hasKey) {
      setFocus("key");
      return;
    }
    if (catalogProvider !== selectedProvider.key) {
      onRequestModels(selectedProvider.key, "");
    }
    setFocus("search");
  }

  function applyGeneration(): void {
    if (!pending || validationError || parsedMax === null) return;
    onSelect({
      backend: pending.backend,
      provider: pending.provider,
      model: pending.model,
      max_tokens: parsedMax,
      reasoning_enabled: reasoningEnabled,
    });
    onExit();
  }

  useInput(
    (input, key) => {
      if (key.escape) {
        if (focus === "models") setFocus("search");
        else if (focus === "search" || focus === "key")
          setFocus("providers");
        else if (editingGeneration)
          setFocus(pending?.backend === "remote" ? "search" : "providers");
        else if (focus === "resume") setFocus("providers");
        else onExit();
        return;
      }
      if (focus === "resume") {
        if (key.return) onExit();
        else if (
          input === "e" ||
          key.downArrow ||
          key.rightArrow ||
          key.tab
        ) {
          touched.current = true;
          setFocus("providers");
        }
        return;
      }
      if (focus === "key" && key.delete) {
        clearKeyOnDelete.current = true;
        setApiKey("");
        return;
      }
      if (focus === "maxTokens" && key.delete) {
        clearMaxOnDelete.current = true;
        setMaxTokens("");
        return;
      }
      if (focus === "providers") {
        if (key.leftArrow || key.upArrow || input === "h" || input === "k") {
          moveSelection(-1);
        } else if (
          key.rightArrow ||
          key.downArrow ||
          input === "l" ||
          input === "j"
        ) {
          moveSelection(1);
        } else if (input === "r" && !isLocal && selectedProvider?.hasKey) {
          setApiKey("");
          setFocus("key");
        } else if (key.return) {
          descend();
        }
        return;
      }
      if (
        focus === "search" &&
        (key.downArrow || key.tab) &&
        items.length > 0
      ) {
        setFocus("models");
        return;
      }
      if (focus === "maxTokens" && (key.return || key.tab)) {
        if (!validationError) setFocus("reasoning");
        return;
      }
      if (focus === "reasoning") {
        if (
          input === " " ||
          key.leftArrow ||
          key.rightArrow ||
          key.upArrow ||
          key.downArrow
        ) {
          setReasoningEnabled((value) => !value);
        } else if (key.return) {
          applyGeneration();
        }
      }
    },
    { isActive: rawInput },
  );

  return (
    <Box flexDirection="column" paddingX={1} marginBottom={1}>
      <Text bold color={COLOR.accent}>
        Cài đặt LLM
      </Text>
      <Text color={COLOR.warn}>
        {ICON.err} Remote gửi transcript đến provider bên thứ ba.
      </Text>

      {config ? (
        <Box marginTop={1}>
          <Panel
            title="Cấu hình gần nhất"
            subtitle={focus === "resume" ? "đang chọn" : undefined}
            variant={focus === "resume" ? "focus" : "idle"}
            width={panelWidth}
          >
            <Text color={COLOR.text}>{savedSummary(config, providers)}</Text>
            <Text color={focus === "resume" ? COLOR.alt : COLOR.muted}>
              Enter dùng lại · e hoặc ↓ để cấu hình
            </Text>
          </Panel>
        </Box>
      ) : null}

      <Box marginTop={1} flexWrap="wrap">
        {choices.map((choice) => {
          const selected = choice.key === selectedProviderKey;
          const active = selected && focus === "providers";
          const status =
            choice.key === "local" || choice.hasKey ? ICON.on : ICON.off;
          return (
            <Text
              key={choice.key}
              color={
                active
                  ? COLOR.bg
                  : selected
                    ? COLOR.accent
                    : COLOR.muted
              }
              backgroundColor={active ? COLOR.accent : undefined}
              bold={selected}
            >
              {` ${status} ${choice.label} `}
            </Text>
          );
        })}
      </Box>
      <Text color={COLOR.muted}>
        {focus === "providers"
          ? "←/→ chọn provider · Enter mở · r đổi key · Esc thoát"
          : focus === "key"
            ? "dán API key · Enter xác thực · Delete xóa hết · Esc quay lại"
            : focus === "search"
              ? "gõ để lọc realtime · ↓ vào danh sách · Esc quay lại"
              : focus === "models"
                ? "↑/↓ chọn · Enter cấu hình model · Esc quay lại"
                : editingGeneration
                  ? "Enter tiếp tục/xác nhận · Esc quay lại"
                  : ""}
      </Text>

      {editingGeneration && pending ? (
        <Box marginTop={1} flexDirection="column">
          <Panel
            title="Generation"
            subtitle={`${providerLabel(pending.backend === "local" ? "local" : pending.provider, providers)} · ${pending.model}`}
            variant="info"
            width={panelWidth}
          >
            <Box flexDirection="column">
              <Text
                bold={focus === "maxTokens"}
                color={focus === "maxTokens" ? COLOR.accent : COLOR.text}
              >
                Max output tokens
              </Text>
              <Box>
                <Text color={COLOR.alt}>
                  {focus === "maxTokens" ? `${ICON.pointer} ` : "  "}
                </Text>
                <TextInput
                  focus={focus === "maxTokens"}
                  value={maxTokens}
                  onChange={(value) => {
                    if (clearMaxOnDelete.current) {
                      clearMaxOnDelete.current = false;
                      setMaxTokens("");
                      return;
                    }
                    setMaxTokens(value.replace(/\D/g, ""));
                  }}
                  onSubmit={() => {
                    if (!validationError) setFocus("reasoning");
                  }}
                  placeholder="2048–500000"
                />
              </Box>
              <Text color={validationError ? COLOR.bad : COLOR.muted}>
                {validationError ||
                  (pendingMax !== null && effectiveMax !== parsedMax
                    ? `Model giới hạn ${pendingMax.toLocaleString("vi-VN")}; chạy thực tế ${effectiveMax?.toLocaleString("vi-VN")} token.`
                    : `Chạy thực tế ${effectiveMax?.toLocaleString("vi-VN")} token.`)}
              </Text>

              <Box marginTop={1}>
                <Text
                  bold={focus === "reasoning"}
                  color={focus === "reasoning" ? COLOR.accent : COLOR.text}
                >
                  {focus === "reasoning" ? `${ICON.pointer} ` : "  "}
                  Reasoning{" "}
                </Text>
                <Text color={reasoningEnabled ? COLOR.good : COLOR.muted}>
                  {reasoningEnabled ? "● bật" : "○ tắt"}
                </Text>
                <Text color={COLOR.muted}> · Space/←/→ đổi</Text>
              </Box>
              <Text color={COLOR.muted}>
                {pendingReasoningMandatory
                  ? "Model bắt buộc reasoning; chạy thực tế luôn bật."
                  : pendingReasoningSupported === false
                    ? "Model không hỗ trợ reasoning; chạy thực tế tắt."
                    : effectiveReasoning === null
                      ? "Model không công bố capability; SoCa giữ mặc định của model/provider."
                      : `Chạy thực tế: reasoning ${effectiveReasoning ? "bật" : "tắt"}.`}
              </Text>
            </Box>
          </Panel>
        </Box>
      ) : isLocal ? (
        <Box marginTop={1} flexDirection="column">
          <Text color={COLOR.good}>
            {ICON.on} Local GGUF — không gửi dữ liệu lên cloud.
          </Text>
          <Text color={COLOR.muted}>Enter để cấu hình Local.</Text>
        </Box>
      ) : selectedProvider ? (
        <Box marginTop={1} flexDirection="column">
          {showKeyInput ? (
            <>
              <Panel
                title={`${selectedProvider.label} API key`}
                width={panelWidth}
                focused={focus === "key"}
              >
                <Box>
                  <Text color={COLOR.alt}>{ICON.pointer} </Text>
                  <TextInput
                    focus={focus === "key" && !keyPending}
                    value={apiKey}
                    mask="•"
                    onChange={(value) => {
                      if (clearKeyOnDelete.current) {
                        clearKeyOnDelete.current = false;
                        setApiKey("");
                        return;
                      }
                      setApiKey(value);
                    }}
                    onSubmit={(value) => {
                      if (!keyPending && value.trim())
                        onSetKey(selectedProvider.key, value.trim());
                    }}
                    placeholder="dán API key rồi Enter để xác thực…"
                  />
                </Box>
              </Panel>
              <Text
                color={selectedProvider.hasKey ? COLOR.good : COLOR.muted}
              >
                {keyPending
                  ? `${ICON.on} đang xác thực API key…`
                  : selectedProvider.hasKey
                    ? `${ICON.on} nhập key mới rồi Enter`
                    : `${ICON.off} cần API key hợp lệ trước khi chọn model`}
              </Text>
            </>
          ) : (
            <Box flexDirection="column">
              <Box>
                <Text color={COLOR.good}>{ICON.on} API key đã lưu</Text>
                <Text color={COLOR.muted}> · r để đổi key</Text>
                {selectedProvider.hasPricingApi ? (
                  <Text color={COLOR.alt}> · giá live</Text>
                ) : null}
              </Box>
              {showModels ? (
                <Box marginTop={1} flexDirection="column">
                  <Panel
                    title="Lọc model"
                    width={panelWidth}
                    focused={focus === "search"}
                  >
                    <Box>
                      <Text color={COLOR.alt}>{ICON.pointer} </Text>
                      <TextInput
                        focus={focus === "search"}
                        value={query}
                        onChange={setQuery}
                        onSubmit={() => {
                          if (items.length > 0) setFocus("models");
                        }}
                        placeholder="gõ từ khóa để lọc realtime…"
                      />
                    </Box>
                  </Panel>
                  <Text color={COLOR.muted}>
                    {visibleCatalog.length === 0
                      ? "đang tải danh sách model…"
                      : `${filtered.length}/${visibleCatalog.length} model`}
                  </Text>
                  {items.length === 0 ? (
                    <Text color={COLOR.muted}>
                      {visibleCatalog.length === 0
                        ? ""
                        : "Không có model khớp từ khóa."}
                    </Text>
                  ) : (
                    <SelectInput<RemoteModelEvent>
                      items={items}
                      isFocused={focus === "models"}
                      limit={6}
                      onSelect={(item) =>
                        beginGeneration({
                          backend: "remote",
                          provider: selectedProvider.key,
                          model: item.value.id,
                          info: item.value,
                        })
                      }
                    />
                  )}
                </Box>
              ) : (
                <Text color={COLOR.muted}>
                  Enter để chọn model của {selectedProvider.label}…
                </Text>
              )}
            </Box>
          )}
        </Box>
      ) : null}

      {notice ? (
        <Text color={notice.startsWith("API key đã") ? COLOR.good : COLOR.bad}>
          {notice}
        </Text>
      ) : null}
    </Box>
  );
}
