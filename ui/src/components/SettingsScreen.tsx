import { useEffect, useMemo, useRef, useState } from "react";
import { Box, Text, useInput, useStdin, useStdout } from "ink";
import SelectInput from "ink-select-input";
import TextInput from "ink-text-input";
import type { LlmConfigEvent, RemoteModelEvent } from "../protocol.js";
import type { LlmProviderStatus } from "../store.js";
import { COLOR, ICON } from "../theme.js";
import { Panel } from "./Primitives.js";

type FocusTarget = "providers" | "key" | "search" | "models";

interface ProviderChoice {
  key: string;
  label: string;
  hasKey: boolean;
  hasPricingApi: boolean;
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
  }) => void;
  /** Leave the settings screen (Esc while on the provider row). */
  onExit: () => void;
}

const LOCAL_FALLBACK_MODEL = "arcee_vylinh_3b_q4_k_m";

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

// Client-side mirror of soca.llm.providers.search_models: case-insensitive,
// whitespace-split tokens AND-ed across id + label. Runs on every keystroke so
// the list filters in real time without a server round-trip.
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
    // ink-select-input keys each row by `key ?? value`; without a string key the
    // object value stringifies to `[object Object]` and every row collides.
    key: model.id,
    label: `${model.id} · ${formatContext(model.context_length)} · ${formatPrice(model)} [${model.pricing_source}]`,
    value: model,
  }));
}

function hintFor(focus: FocusTarget): string {
  switch (focus) {
    case "providers":
      return "←/→ chọn provider · Enter mở · Esc thoát cài đặt";
    case "key":
      return "dán API key · Enter xác thực · Esc quay lại provider";
    case "search":
      return "gõ để lọc realtime · ↓ vào danh sách · Esc quay lại provider";
    case "models":
      return "↑/↓ chọn · Enter dùng model · Esc quay lại tìm kiếm";
  }
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
  const panelWidth = Math.max(24, Math.min(72, (stdout?.columns ?? 80) - 2));

  const configProvider =
    config?.backend === "remote" ? config.provider : "local";
  const [selectedProviderKey, setSelectedProviderKey] =
    useState(configProvider);
  const [focus, setFocus] = useState<FocusTarget>("providers");
  const [apiKey, setApiKey] = useState("");
  const [query, setQuery] = useState("");

  // Sync the highlight from config until the user first navigates. config
  // arrives asynchronously, so the initial "local" must catch up once known.
  const touched = useRef(false);
  useEffect(() => {
    if (touched.current || !config) return;
    setSelectedProviderKey(
      config.backend === "remote" ? config.provider : "local",
    );
  }, [config?.backend, config?.provider]);

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

  // A saved provider hides its key field entirely; it only reappears when the
  // user explicitly asks to replace the key (press "r" -> focus "key").
  const showKeyInput = !selectedProvider?.hasKey || focus === "key";
  const browsing = focus === "search" || focus === "models";
  const showModels = browsing || visibleCatalog.length > 0;

  // Key just validated (hasKey false -> true while entering it): clear the field
  // and back out to the provider row. Choosing a model is a separate, user-
  // initiated step, so do NOT auto-jump into search. Watches the flag only, so
  // no per-render refetch churn.
  const prevHasKey = useRef(Boolean(selectedProvider?.hasKey));
  useEffect(() => {
    const hasKey = Boolean(selectedProvider?.hasKey);
    if (focus === "key" && hasKey && !prevHasKey.current) {
      setApiKey("");
      setFocus("providers");
    }
    prevHasKey.current = hasKey;
  }, [selectedProvider?.hasKey, focus]);

  // A successful replacement must leave the key editor even when the
  // provider already had a key (hasKey therefore stays true). Failed
  // validation keeps the editor open so the user can retry.
  const wasKeyPending = useRef(false);
  useEffect(() => {
    if (wasKeyPending.current && !keyPending) {
      if (notice.startsWith("API key đã")) {
        setApiKey("");
        setFocus("providers");
      }
    }
    wasKeyPending.current = keyPending;
  }, [keyPending, notice]);

  // ink-text-input treats Delete like Backspace. For a masked API key field,
  // Delete should clear the whole temporary buffer in one operation.
  const clearKeyOnDelete = useRef(false);

  function moveSelection(delta: number): void {
    touched.current = true;
    const next =
      choices[(selectedIndex + delta + choices.length) % choices.length];
    if (!next) return;
    setSelectedProviderKey(next.key);
    setApiKey("");
    setQuery("");
  }

  function descend(): void {
    if (!selectedProvider) return;
    if (selectedProvider.key === "local") {
      onSelect({
        backend: "local",
        provider: config?.provider ?? "openrouter",
        model:
          config?.backend === "local" ? config.model : LOCAL_FALLBACK_MODEL,
      });
      onExit(); // choosing a backend is the final step -> back to main UI
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

  useInput(
    (input, key) => {
      if (key.escape) {
        if (focus === "models") setFocus("search");
        else if (focus === "search" || focus === "key") setFocus("providers");
        else onExit();
        return;
      }
      if (focus === "key" && key.delete) {
        clearKeyOnDelete.current = true;
        setApiKey("");
        return;
      }
      // Only the provider row reacts to navigation keys; while a text field or
      // the model list owns focus, let its own input handler consume them.
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
          setFocus("key"); // reveal an empty field to replace a saved/expired key
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
        {ICON.err} Remote gửi transcript đến provider bên thứ ba. Local vẫn là
        mặc định.
      </Text>

      <Box marginTop={1} flexWrap="wrap">
        {choices.map((choice) => {
          const selected = choice.key === selectedProviderKey;
          const active = selected && focus === "providers";
          const status =
            choice.key === "local" || choice.hasKey ? ICON.on : ICON.off;
          const color = active
            ? COLOR.bg
            : selected
              ? COLOR.accent
              : COLOR.muted;
          return (
            <Text
              key={choice.key}
              color={color}
              backgroundColor={active ? COLOR.accent : undefined}
              bold={selected}
            >
              {` ${status} ${choice.label} `}
            </Text>
          );
        })}
      </Box>
      <Text color={COLOR.muted}>{hintFor(focus)}</Text>
      {config ? (
        <Text color={COLOR.muted}>
          giới hạn đầu ra mặc định: {config.max_tokens.toLocaleString("vi-VN")} token
        </Text>
      ) : null}

      {isLocal ? (
        <Box marginTop={1} flexDirection="column">
          <Text color={COLOR.good}>
            {ICON.on} Local GGUF — không gửi dữ liệu lên cloud.
          </Text>
          <Text color={COLOR.muted}>
            Model:{" "}
            {config?.backend === "local" ? config.model : LOCAL_FALLBACK_MODEL}
          </Text>
          <Text color={COLOR.muted}>Enter để dùng backend Local.</Text>
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
                    placeholder={
                      selectedProvider.hasKey
                        ? "nhập key mới để thay…"
                        : "dán API key rồi Enter để xác thực…"
                    }
                  />
                </Box>
              </Panel>
              <Box marginTop={1}>
                <Text
                  color={selectedProvider.hasKey ? COLOR.good : COLOR.muted}
                >
                  {keyPending
                    ? `${ICON.on} đang xác thực API key…`
                    : selectedProvider.hasKey
                      ? `${ICON.on} nhập key mới rồi Enter · Esc để hủy`
                      : `${ICON.off} cần API key hợp lệ trước khi chọn model`}
                </Text>
                {selectedProvider.hasPricingApi ? (
                  <Text color={COLOR.alt}> · giá live</Text>
                ) : (
                  <Text color={COLOR.muted}>
                    {" "}
                    · giá không rõ nếu provider không trả API
                  </Text>
                )}
              </Box>
            </>
          ) : (
            // Key already saved: hide the input entirely, offer the model step.
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
                      : `${filtered.length}/${visibleCatalog.length} model · ${hintFor(focus)}`}
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
                      onSelect={(item) => {
                        onSelect({
                          backend: "remote",
                          provider: selectedProvider.key,
                          model: item.value.id,
                        });
                        onExit(); // model picked -> leave settings, back to chat
                      }}
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
