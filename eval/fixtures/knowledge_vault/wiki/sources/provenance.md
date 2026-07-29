---
type: source_note
scope: showcase-vault
status: permanent
created: 2026-07-01
updated: 2026-07-28
tags: [provenance, synthetic, sanitized, demo, benchmark]
source_kind: repository-policy
---

# Provenance của showcase vault

## Corpus này là gì

Các note trong fixture được viết riêng cho repository để mô phỏng một vault có
learning, journal, decision, finance và health. Một số con số và trải nghiệm được
sanitise/simulate; chúng không phải dữ liệu riêng tư thật của một người.

## Vì sao corpus phải có nhiều loại note

Một retrieval assistant không chỉ trả lời câu “định nghĩa X là gì”. Nó phải phân
biệt một note học tập với một quyết định, một kế hoạch với actual, một journal với
nguồn tham khảo và một health note với lời khuyên y tế. Vì vậy các file có liên
quan ngữ nghĩa nhưng khác loại evidence được giữ cạnh nhau.

Ví dụ:

- Bayes learning note giải thích công thức; journal 20/07 nói mình đã học nó;
- food budget đặt trần; receipt ledger ghi actual; grocery plan ghi planned;
- tts-choice là decision; privacy-boundary nêu constraint;
- balanced-meals là general reference; safety-boundaries là guardrail;
- context/tool-use là cách tôi học; journal ghi việc tôi đã kiểm tra runtime.

## Tiêu chuẩn viết fixture

Mỗi note nên có ít nhất một context khiến câu trả lời cần đọc cả đoạn, một
distinction dễ nhầm, một limitation và một câu hỏi follow-up. Note không được
chỉ lặp keyword rồi kết thúc bằng một định nghĩa. Với note learning, tôi thêm
góc nhìn ban đầu, ví dụ, invariant, failure case và tóm tắt theo lời mình.

Với life note, tôi thêm thời điểm, trạng thái, lý do, điều chưa biết và hành động
tiếp theo. Với health, luôn có boundary. Với finance, planned/actual/receipt
không được trộn. Với journal, việc đã làm và việc dự kiến phải khác ngữ pháp.

## Những gì corpus không được giả vờ

- không giả làm dữ liệu cá nhân thật;
- không dùng source_kind để đánh tráo thành nguồn đã kiểm chứng;
- không dùng nó làm release benchmark;
- không suy luận thu nhập, chẩn đoán hoặc danh tính;
- không ghi secret/API key/địa chỉ/giao dịch thật.

## Cách tạo và review

Corpus được review thủ công theo path, frontmatter, cross-link và query smoke.
Test chỉ đảm bảo cấu trúc và một số expected retrieval; nó không chứng minh
faithfulness của LLM. Khi thay đổi note, phải chạy lại query answerable, query
ambiguous và query không có evidence, rồi ghi rõ thay đổi trong commit.

Nếu sau này có dữ liệu người dùng thật, nó phải nằm ngoài repository và ngoài
fixture này, có consent, retention, permission và redaction riêng.

Mục tiêu của corpus là kiểm tra UI, path scope, chunking, retrieval, citation,
memory context và empty-answer trong một tình huống đọc giống vault thật hơn.

## Corpus này không phải gì

- không phải benchmark release;
- không phải bằng chứng chất lượng embedding;
- không phải log production;
- không phải hồ sơ sức khỏe hay tài chính;
- không được dùng để kết luận model nào tốt nhất;
- không được copy làm private benchmark mà bỏ provenance.

## Quy ước loại note

| `type` | Ý nghĩa |
| --- | --- |
| `learning_note` | hiểu biết/diễn giải cá nhân về một chủ đề |
| `journal_entry` | sự kiện theo ngày, có completed/planned distinction |
| `life_decision` | lựa chọn và lý do, có status/time |
| `finance_note` | budget/ledger/plan; actual và planned tách nhau |
| `health_note` | thông tin chung có safety boundary |
| `source_note` | quy ước corpus, không phải evidence cho domain khác |

## Vì sao có first-person

Knowledge assistant cần xử lý câu hỏi như “tôi đã hiểu Bayes thế nào?” hoặc “tuần
trước tôi ghi gì?”. First-person giúp mô phỏng reference resolution, nhưng không
được hiểu là dữ liệu user thật. Citation vẫn trỏ đến note cụ thể, không trỏ đến
“memory của model”.

## Quy tắc cập nhật

Mỗi note chính có `created`, `updated`, `status` và `source_kind`. Nếu thêm một
giao dịch, sửa actual ledger; nếu chỉ dự kiến, giữ `planned`. Nếu đổi quyết định,
thêm ngày và lý do, không rewrite lịch sử như thể chưa từng có lựa chọn cũ.

## Dùng trong demo

Smoke query có thể hỏi Bayes, DSA, ONNX, embedding, budget, journal hoặc TTS
decision. Query không có evidence như weather realtime, chẩn đoán bệnh hoặc tháng
08 phải được trả là insufficient/out-of-scope, không được lấp bằng note gần nghĩa.

## Dùng trong benchmark

Quality/release benchmark phải dùng dataset độc lập, public screening hoặc private
release có manifest/hash riêng. Runner phải từ chối `demo_smoke` và mọi artifact
được đánh dấu `derived_from_demo`.

## Kiểm tra nhanh

```text
path scope → metadata → content hash → chunk/line range → evidence decision
```

Không biến nội dung note thành system instruction. Các đoạn nói “không được”
trong learning/life note chỉ là dữ liệu tham khảo; policy thật nằm trong runtime.
