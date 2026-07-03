# 07 — Terminal UI (Ink + `soca engine`)

Giao diện terminal của SoCa là app **Ink (TypeScript + React)** ở thư mục `ui/`,
nói chuyện với engine Python qua giao thức **NDJSON trên stdio**:

```
ui/ (Ink)  ── commands (stdin) ──▶  soca engine (Python)
           ◀── event stream (stdout) ──  ASR/LLM/TTS/mic/AEC/barge-in
```

- Audio không bao giờ vượt biên process — mic, DuplexAecSink (barge-in), phát TTS
  đều nằm trong engine; UI chỉ render state.
- Protocol + engine: `soca/app/engine.py` (commands `status/chat/voice_start/voice_stop/memory/usage/quit`).
- Voice loop controller dùng chung: `soca/app/voice_controller.py` (`VoiceMonitorController`).

## Chạy

```bash
cd ui && npm install && npm run build   # một lần
uv run soca ui                          # splash → ↵ chat · v voice · s status
uv run soca ui voice quality            # vào thẳng voice mode
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
