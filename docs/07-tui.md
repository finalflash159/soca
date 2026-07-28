# 07 — Terminal UI (Ink + `soca engine`)

Giao diện terminal của SoCa là app **Ink (TypeScript + React)** ở thư mục `ui/`,
nói chuyện với engine Python qua giao thức **NDJSON trên stdio**:

```
ui/ (Ink)  ── commands (stdin) ──▶  soca engine (Python)
           ◀── event stream (stdout) ──  ASR/LLM/TTS/mic/AEC/barge-in
```

- Audio không bao giờ vượt biên process — mic, DuplexAecSink (barge-in), phát TTS
  đều nằm trong engine; UI chỉ render state.
- Protocol + engine: `soca/app/engine.py` (commands
  `status/context/chat/voice_start/voice_stop/memory/memory_compact/usage/quit`).
- Voice loop controller dùng chung: `soca/app/voice_controller.py` (`VoiceMonitorController`).

## Chạy

```bash
cd ui && npm install && npm run build   # một lần
uv run soca ui                          # splash → ↵ chat · v voice · s settings
uv run soca ui voice baseline           # vào thẳng voice mode
```

Dev UI: `cd ui && npm run dev`. Override lệnh engine: env `SOCA_ENGINE_CMD`.

## Cấu trúc `ui/src/`

```
theme.ts          design tokens "bình minh" (đồng bộ soca/app/style/palette.py)
protocol.ts       wire types của NDJSON protocol
engine.ts         spawn + đọc/ghi engine child process
store.ts          reducer: engine events → app state
App.tsx           layout chính (Static history + composer + footer)
components/       Logo (bird gradient), Timeline, VoiceStatus, HelpOverlay, Primitives
```

> Bản Textual TUI cũ (`soca/app/tui/`) đã gỡ sau khi Ink UI đạt parity voice
> (live-test 2026-07-03).

## Slash command và informational view

`ui/src/keymap.ts` là source of truth duy nhất cho command palette và help.
Gõ `/` mở toàn bộ lệnh; nhập tiếp lọc theo prefix; `↑/↓` di chuyển, `Tab` điền
lệnh và `Enter` thực thi. `/inspect` cũ đã bị loại vì engine chưa từng có
implementation tương ứng.

Các lệnh chỉ xem (`/status`, `/context`, `/memory`, `/usage`) không đổi mode và
không ghi output vào timeline. Chúng mở panel tạm thời trên chat/voice hiện tại;
bắt đầu nhập nội dung mới sẽ đóng panel ngay. `/settings` và `/memory proposals`
là interaction surface nên vẫn giữ focus cho tới khi người dùng hoàn tất hoặc
đóng.

- `/memory` chỉ mô tả working session memory, summary/recent turn và compaction.
- `/context` mô tả prompt resident, output reserve, model window và các phần
  query-dependent như knowledge/archive memory.
- `/usage` là tổng token/latency đã thực sự ghi nhận qua các lượt LLM; nó không
  phải dung lượng context đang giữ.

Token bar dùng cùng counter bảo thủ `utf8_bytes_div_4` với
`WorkingMemoryPolicy`, hiển thị `current / hard limit` (mặc định 16.384 token).
Vì đây không phải tokenizer riêng của model, UI luôn đánh dấu số bằng `~`.
