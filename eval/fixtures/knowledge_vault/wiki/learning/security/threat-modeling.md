---
type: learning_note
domain: security
topic: threat-modeling-for-local-assistants
status: active
created: 2026-07-29
updated: 2026-07-29
confidence: medium
tags: [security, threat-model, privacy, prompt-injection, least-privilege]
source_kind: personal-study-note
---

# Security: threat modeling cho assistant local — local không tự động an toàn

## Tôi bắt đầu từ asset, không bắt đầu từ tool

Tôi từng nghĩ chạy local là đủ an toàn. Nhưng API key, transcript, Markdown
vault, memory proposal, vector index, provider config và UI đều là asset khác nhau.
Mỗi asset có confidentiality, integrity và availability khác nhau.

Ví dụ, vector index có thể rebuild nhưng vẫn chứa snippet riêng tư. Một proposal
memory chưa approve cần integrity để không bị biến thành core memory. API key cần
secret handling, còn status UI cần accuracy để user không bị đánh lừa.

## Bảng asset và boundary

| Asset | Rủi ro chính | Boundary tôi muốn |
| --- | --- | --- |
| API key | lộ secret | secure local store, masked log |
| transcript | lộ nội dung | local mặc định, remote consent |
| vault note | đọc trái phép | path scope, file permission |
| vector/index | metadata + snippet lộ | cache private, generation rõ |
| memory proposal | ghi sai dài hạn | approval bắt buộc |
| tool call | side effect/path escape | typed args + runtime guard |
| prompt trace | lộ context | redact và retention giới hạn |
| UI status | user hiểu sai | hiện engine/provider thật |

## Threat actor không chỉ là hacker từ xa

Tôi xét cả process local bị lỗi, extension đọc terminal, user khác trên máy,
provider remote lưu request, note chứa prompt injection và model tự sinh tool call
sai. Threat model không cần viết kịch bản điện ảnh; chỉ cần hỏi “nếu boundary này
bị phá, chuyện xấu nhất là gì và detect bằng đâu?”.

## Least privilege

Knowledge search chỉ đọc trong vault. Knowledge read chỉ đọc path đã normalize
và nằm trong scope. Local time không cần quyền file. Memory proposal có thể tạo
artifact pending nhưng không được tự ghi approved memory. Mỗi tool có input,
output, side effect và permission riêng.

Tôi không giao quyền cho LLM chỉ vì model viết ra JSON hợp lệ. Runtime validate
schema, path, capability và consent trước khi execute.

## Retrieved text là data, không phải policy

Một note có thể chứa câu “ignore previous instructions” hoặc “gọi tool xóa file”.
Nội dung đó vẫn là data để trích dẫn/phân tích. Prompt phải boundary rõ giữa
system policy và retrieved content; tool result không được tự nâng quyền.

Tôi phân biệt prompt injection với hallucination. Injection cố thay đổi authority;
hallucination là claim không có evidence. Cả hai cần guard khác nhau.

## Integrity của memory

Tôi muốn working memory giữ continuity trong session, nhưng core/archive cần
lifecycle khác. Một câu assistant nói “tôi sẽ nhớ…” không phải approval. Proposal
phải có nguồn, lý do, thời điểm và trạng thái pending/approved/rejected.

Nếu index bị rollback, memory state không được rollback âm thầm. Nếu một note bị
sửa, citation cũ phải không còn được coi là evidence hiện tại nếu digest đổi.

## Remote boundary

Remote provider nhận payload, không chỉ API key. Payload có thể gồm câu hỏi,
working context, knowledge snippet và memory. UI phải hiện provider/model và
consent trước khi gửi. “Key đã lưu local” không thay thế policy dữ liệu.

Tôi muốn trace ghi `remote=true`, provider, model và số component context nhưng
không ghi secret/raw transcript mặc định.

## Những control tôi ưu tiên

1. normalize và scope-check path;
2. typed tool schema;
3. explicit local/remote status;
4. private permission cho index chứa content;
5. citation/provenance trong evidence;
6. approval cho memory side effect;
7. prompt budget và redaction;
8. audit event không chứa secret;
9. cancellation/timeout không để trạng thái giả success;
10. regression test cho từng boundary.

## Failure cases tôi muốn thử

- path `../secret.md` hoặc backslash escape;
- note prompt injection trong retrieved chunk;
- citation ID không tồn tại;
- index permission 0644 trên máy shared;
- remote setting áp dụng chat nhưng voice dùng local lặng lẽ;
- memory proposal được tạo nhưng status thành approved nhầm;
- API key xuất hiện trong exception;
- tool timeout nhưng UI nói đã xong.

## Cách tôi đánh giá security

Tôi không dùng một checklist pass để nói sản phẩm an toàn. Tôi ghi asset, threat,
control, detection và residual risk. Nếu control không có test hoặc telemetry,
đó mới chỉ là ý định.

## Câu hỏi còn mở

- retention trace bao lâu là đủ cho debug mà không giữ quá nhiều transcript?
- permission private nên enforce ở mọi nền tảng hay có adapter riêng?
- làm sao user xem memory proposal mà không phải đọc raw prompt dài?
- remote provider có cơ chế xóa/retention nào cần hiển thị?

## Bài tập

Viết một note chứa prompt injection, truy vấn nó qua knowledge, rồi kiểm tra
assistant chỉ trích nội dung mà không làm theo lệnh. Sau đó thử path ngoài scope,
memory proposal và remote status. Một bài pass phải assert cả outcome và trace.
