---
type: learning_note
domain: llm-serving
topic: local-and-remote-inference
status: active
created: 2026-07-24
updated: 2026-07-28
tags: [llm, serving, local, remote, quantization, latency, kv-cache]
source_kind: personal-study-simulation
---

# LLM serving: local và remote — tôi cân bằng chất lượng với tài nguyên thế nào

## Tôi từng nhìn sai trade-off

Tôi từng thấy model lớn trả lời hay rồi muốn dùng cho mọi lượt. Sau đó gặp cold
start, RAM, context, API latency, reasoning token và privacy thì thấy “hay” chỉ
là một trục. Assistant desktop cần usable, không phải chỉ thắng một prompt.

## Local serving

Local không gửi transcript ra provider, có thể offline và kiểm soát lifecycle model.
Đổi lại máy phải chứa weight, backend, tokenizer và context KV cache. Load model
lâu, compile provider hoặc swap memory có thể làm UI tưởng bị treo.

Tôi tách:

- download/provision một lần;
- load model khi engine cần;
- warm-up một request nhỏ;
- generate;
- unload/shutdown khi summarizer không còn cần.

## Remote serving

Remote giảm gánh weight/RAM local và thường có model mạnh hơn. Đổi lại transcript
ra ngoài, network/provider failure, pricing, rate limit và behavior thay đổi theo
model/provider. API key có thể lưu local nhưng data request vẫn có policy riêng.

UI phải nói rõ provider/model áp dụng cho chat và voice nếu dùng chung setting;
không hiện “local mặc định” khi user đã chọn remote mà runtime voice vẫn local.

## Latency tôi phân biệt

- cold start: từ process đến model usable;
- TTFT: từ request đến token đầu;
- prompt processing: đọc context;
- generation speed: token/second;
- total latency: đến token cuối;
- audio first byte: với voice streaming.

TTFT cao không nhất thiết generation chậm. Prompt dài, reasoning, queue hoặc
network đều có thể làm TTFT tăng.

## KV cache

Decoder phải dùng lại representation của prefix khi sinh token. KV cache giữ
key/value để không tính lại toàn bộ prefix mỗi bước. Cache lớn theo sequence
length, layer, head và precision; context dài làm memory tăng dù output ngắn.

Prefix caching có thể giảm latency nhưng phải invalidate đúng khi system prompt,
model revision hoặc user data thay đổi. Không dùng cache của user A cho user B.

## Quantization

Quantization giảm số bit của weight/activation, giảm disk/RAM và đôi khi tăng
throughput. Đổi lại có thể giảm chất lượng, thay sampling behavior hoặc gây lỗi
ở layer nhạy cảm. Q4 không phải một chất lượng cố định; backend, group size,
calibration và model architecture đều ảnh hưởng.

Tôi benchmark cùng prompt set ở FP16/Q8/Q4 nếu resource cho phép, ghi model file
size, peak RSS, TTFT, tok/s và answer quality. Không chọn chỉ vì file nhỏ.

## Reasoning và max output

Một provider có thể bắt buộc reasoning hoặc tính reasoning vào output budget.
`max_tokens` do user chọn phải clamp theo model capability và reserve; nhập số
500K không nghĩa model có thể tạo 500K. Nếu reasoning on nhưng model không support,
engine phải nương theo capability và hiển thị effective config.

## Batching và concurrency

Batching tăng utilization nhưng có thể làm một request ngắn chờ request dài.
Assistant desktop ưu tiên tail latency và cancellation hơn throughput server.

Voice cần barge-in: khi user nói lại, generation/TTS cũ phải cancel thật và không
append partial answer thành final history.

## Failure handling

Provider empty content, timeout, 429, context overflow và invalid reasoning config
là các lỗi khác nhau. Retry chỉ áp dụng lỗi transient và không lặp side effect.
Empty content cần xem finish reason, effective output, reasoning mode và raw usage;
không chỉ tăng token mù.

## Benchmark record mẫu

| Field | Tại sao cần |
| --- | --- |
| model/provider/revision | reproduce behavior |
| hardware/backend | giải thích latency/RSS |
| prompt token/output reserve | giải thích overflow/empty |
| TTFT/total/tok/s | phân biệt bottleneck |
| answer/citation/grounding | quality không tách latency |
| cache state | cold/warm không trộn |

## Cách tôi chọn model

1. lọc theo context/capability/license/resource;
2. chạy quality trên held-out data, không lấy demo làm release gate;
3. đo local cold/warm và remote TTFT;
4. kiểm Vietnamese, code-mix, citation và abstention;
5. ghi rõ điểm yếu, không chỉ ghi model thắng;
6. chọn default theo product risk chứ không theo một câu trả lời hay nhất.

## Tóm tắt

Serving là một phần của chất lượng. Model tốt nhưng load quá lâu, tràn context,
không cancel được hoặc gửi data sai policy thì chưa phải model phù hợp cho SoCa.

## Tôi tách các con số latency

TTFT là thời gian đến token đầu, decode throughput là tốc độ token sau đó, còn
total latency bao gồm tool, queue, network và TTS nếu voice. Remote có network
jitter và provider queue; local có model load, memory pressure và contention với
ASR/TTS. Chỉ báo total average sẽ che failure p95.

Tôi đo cold start, warm start, prompt ngắn/dài, output ngắn/dài và cancellation.
Một model có throughput cao nhưng TTFT lâu có thể khó dùng cho voice. Một model
nhỏ trả lời nhanh nhưng hallucination nhiều cũng không thắng product objective.

## Resource budget

Model weight chưa phải toàn bộ RAM. Còn runtime overhead, tokenizer, KV cache,
batch, temporary buffer và OS cache. Quantization giảm memory/compute theo trade-
off chất lượng. Tôi ghi peak sau load và peak trong context dài; không lấy file
size làm RAM estimate.

Summary model không phải core LLM. Nó chỉ cần chạy khi compaction và có thể unload
sau job. Vì vậy startup cost, context support, Vietnamese quality, output schema,
empty response và giới hạn tài nguyên phải được benchmark riêng. Không chọn 14B
chỉ vì điểm một task đẹp nếu nó chiếm resource của voice/chat.

## Local/remote parity

UI có thể cho user chọn provider áp dụng cả chat và voice, nhưng status phải nói
đúng boundary. Local/remote khác model capability, reasoning requirement, max
output và error shape. Engine normalize config trước khi gọi, clamp output theo
model limit, và không gửi reasoning flag nếu endpoint không hỗ trợ.

Remote request cần redact hoặc consent cho context nhạy cảm. Local request cần
health check model, provider và resource; “local” không có nghĩa lúc nào cũng
responsive.

## Release checklist

- model revision/license được ghi;
- cold process tải được từ cache trống;
- max context/output không vượt model;
- empty completion có fallback/diagnostic;
- cancel/timeout không để worker zombie;
- trace không lộ secret/raw private text;
- benchmark có Vietnamese, code-mix, no-answer, citation và tool loop;
- chọn default theo risk, không theo một screenshot.

## Cách tôi ra quyết định

Tôi cho điểm từng slice và giữ failure examples cạnh điểm. Nếu model A hay hơn
nhưng cold start 40 giây và chiếm hết RAM, nó có thể là optional profile chứ không
phải default. Nếu model B nhỏ hơn nhưng grounded và ổn định, nó phù hợp core hơn.
Summary model được đánh giá theo việc giữ goal/constraint/uncertainty, không theo
văn phong đẹp.
