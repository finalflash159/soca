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

Các lệnh chỉ xem (`/status`, `/context`, `/memory`) không đổi mode và không ghi
output vào timeline. Chúng mở panel trên chat/voice hiện tại;
bắt đầu nhập nội dung mới sẽ đóng panel ngay. `/settings` và `/memory-proposals`
là interaction surface nên vẫn giữ focus cho tới khi người dùng hoàn tất hoặc
đóng.

`/status` hiển thị runtime đang cấu hình/thực thi, không dùng profile tĩnh làm
đại diện cho chat: Chat LLM, voice ASR/LLM/TTS, SmartTurn, VAD, ASR guards,
working-summary worker, archive memory, embedding và các tool router. Model
được khởi tạo lazy sẽ ghi `configured`/`ready`; chỉ worker đã tạo trong process
mới ghi `loaded`. Knowledge index hiển thị riêng sparse/dense generation hiện
tại.

- `/memory` chỉ mô tả working session memory, summary/recent turn và compaction.
- `/compact` mở progress panel và tự poll worker tới trạng thái cuối.
  Khi hoàn tất panel hiển thị token trước/sau; `/compact-show` mở riêng
  working summary đã tạo mà không bung toàn bộ recent conversation. Manual
  compact bỏ qua ngưỡng 15K nhưng yêu cầu ít nhất 5 lượt hoàn chỉnh và luôn giữ
  2 lượt gần nhất; nếu chưa đủ, panel trả `noop` kèm bộ đếm `hiện có X/5`.
  Mỗi artifact bắt buộc có prose continuity summary, kể cả khi không có
  decision/constraint/open item bền vững. Nếu model trả summary rỗng,
  coordinator từ chối publish và giữ nguyên toàn bộ lịch sử cùng summary cũ.
- `/context` gộp hai lát cắt liên quan nhưng không đồng nhất: context hiện đang
  resident (prompt, output reserve, model window và phần query-dependent) và
  usage LLM tích lũy của phiên (prompt/completion token, TTFT, throughput).
- `/usage` được giữ làm alias tương thích cho `/context`, không chiếm một mục
  riêng trong command palette.

Token bar dùng cùng counter bảo thủ `utf8_bytes_div_4` với
`WorkingMemoryPolicy`, hiển thị `current / hard limit` (mặc định 16.384 token).
Vì đây không phải tokenizer riêng của model, UI luôn đánh dấu số bằng `~`.

## Turn progress và timeline

Engine phát `turn_progress` từ stage runtime đang chạy thật; UI không dùng timer
để giả lập tiến độ. Các phase ổn định gồm chuẩn bị, phân tích, định tuyến,
memory, knowledge retrieval, tool, tổng hợp LLM, validation và TTS. Chat và
voice dùng chung event này, còn chi tiết nội bộ nằm trong trường `operation`.
Panel progress đổi accent theo loại công việc và giữ tối đa bốn phase vừa hoàn
tất để người dùng thấy luồng xử lý mà không làm timeline quá ồn.

Tin nhắn người dùng giữ layout phẳng hiện tại. Mỗi câu trả lời hoàn chỉnh của
SoCa được đặt trong một khung hairline dùng màu dawn palette đã pha với border
nền; progress biến mất khi engine phát trạng thái `done`.
