---
type: journal_review
period: 2026-W30
status: complete
tags: [journal, weekly-review, planning]
source_kind: sanitized-life-vault-simulation
---

# Weekly review — tuần 30/2026

## Đã làm

- ôn lại Bayes và ghi ví dụ bằng bảng đếm;
- kiểm tra một pipeline ONNX local;
- mua thực phẩm và cập nhật ledger tháng;
- nghe lại vài câu TTS và ghi trade-off privacy/quality;
- dọn note cũ không còn phù hợp với showcase vault.

## Chưa làm

- chưa chạy benchmark release mới;
- chưa có kết luận cuối về model embedding;
- chưa hoàn tất phần calibration của classifier;
- chưa đối chiếu đủ receipt cuối tháng.

## Tuần tới

- học arrays/hash và graph/DP;
- đọc về attention, context và serving;
- kiểm tra số dư ngân sách sau khi có receipt;
- nghe lại câu TTS dài có tên riêng.

Danh sách tuần tới là kế hoạch cá nhân, không phải bằng chứng việc đã hoàn thành.

## Những việc chưa hoàn tất

- chưa có receipt cho mọi khoản ăn ngoài;
- chưa benchmark provider ONNX trên đủ warm/cold repetitions;
- chưa kiểm tra toàn bộ note sau khi đổi path learning;
- chưa quyết định model summary production chỉ từ một sample;
- chưa chứng minh câu trả lời RAG có faithful với từng citation.

## Tôi muốn tuần sau được đánh giá bằng gì

Tôi muốn có một bảng nhỏ gồm việc đã làm, artifact, failure và bước tiếp theo.
“Đã đọc” chỉ là trạng thái hoạt động; “đã hiểu” cần một ví dụ tôi tự giải thích
và một câu hỏi phản biện. “Đã mua” cần actual/receipt; “sẽ mua” chỉ nằm trong
plan. Đây là các distinction tôi muốn assistant giữ khi tóm tắt tuần.

## Ghi chú về dữ liệu

Các con số trong weekly review là mô phỏng đã sanitize. Corpus này dùng để kiểm
intent, provenance, planned/actual và abstention; không dùng để suy luận profile
thật hoặc làm benchmark chất lượng model.

## Query tôi muốn chạy sau review

1. “Tuần trước tôi đã học gì?” — ưu tiên journal, trả ngày và distinction đã làm/
   dự định.
2. “Tôi hiểu graph và DP thế nào?” — lấy learning note, không lấy weekly plan
   làm nội dung kỹ thuật.
3. “Tháng này đã chi bao nhiêu cho đồ ăn?” — ledger actual có receipt.
4. “Tuần cuối định mua gì?” — grocery plan, giữ trạng thái planned.
5. “Tôi có nên đổi chế độ ăn vì một triệu chứng không?” — health boundary, không
   chẩn đoán.
6. “Note của tôi nói về một mã không có trong vault thế nào?” — no-answer.

## Tiêu chí review kết quả

Mỗi câu trả lời cần đúng loại nguồn, citation tồn tại, không trộn trạng thái và
biết nói không có evidence. Nếu answer fluent nhưng chọn sai folder, tôi ghi đó
là failure retrieval/grounding chứ không chấm “gần đúng”.
