---
type: journal_review
period: 2026-W30
status: complete
created: 2026-07-27
updated: 2026-07-29
tags: [journal, weekly-review, actual, planned, reflection]
source_kind: redacted-personal-note
---

# Review tuần 30/2026

## Đã làm và evidence tương ứng

| Việc | Evidence | Trạng thái |
| --- | --- | --- |
| ôn Bayes bằng bảng đếm | journal 20/07 + learning note | complete |
| kiểm tra pipeline ONNX | journal 23/07 | complete, metric chưa đủ |
| cập nhật food receipt | receipt ledger | complete-to-28/07 |
| nghe lại TTS và ghi trade-off | decision TTS | complete, confidence medium |
| dọn cấu trúc vault | journal 27/07 + decision | complete |
| kiểm empty retrieval/compact | journal 28/07 | observed, còn remediation |

“Đã làm” ở bảng này chỉ nói activity có record. “Đã hiểu” cần một ví dụ tự giải
thích và một câu hỏi phản biện; “đã benchmark” cần repetition và artifact.

## Chưa hoàn tất

- chưa có benchmark release độc lập cho retrieval;
- chưa có kết luận cuối về embedding model;
- chưa chạy đủ cold/warm repetition cho ONNX/TTS;
- chưa đối chiếu hai receipt lunch pending;
- chưa biết summary compact có giữ mọi uncertainty trong mọi case;
- chưa kiểm hết câu query voice nói sai chính tả.

## Tuần tới

1. học thêm network/security/data pipeline;
2. chạy query “tuần trước tôi học gì” và xem loại source;
3. khóa food ledger ngày 31/07;
4. thử TTS bằng câu code-mix và barge-in;
5. thêm correction note nếu summary làm mất trạng thái unknown;
6. giữ benchmark release tách khỏi showcase corpus.

## Điều tôi muốn assistant giữ khi tóm tắt

Nếu nguồn là journal, nói đó là việc đã ghi trong ngày. Nếu nguồn là learning,
đó là cách tôi hiểu khái niệm. Nếu nguồn là plan, giữ “dự kiến”. Nếu không có
record, nói chưa tìm thấy. Một câu gần đúng loại evidence vẫn là câu trả lời sai.

## Cách tôi review tuần sau

Mỗi mục phải có path, trạng thái, failure và bước tiếp theo. Tôi không dùng một
con số tổng để che những mục còn unknown. Nếu thay đổi cấu trúc note, tôi ghi
change log để query cũ không âm thầm trỏ vào câu chuyện khác.
