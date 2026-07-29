---
type: life_decision
area: personal-tooling
status: current
created: 2026-07-18
updated: 2026-07-29
confidence: medium
review_after: 2026-08-31
tags: [tts, privacy, local, voice, latency, decision]
source_kind: redacted-personal-note
---

# Vì sao tôi giữ TTS local làm mặc định

## Bối cảnh thật của quyết định

Tôi không chỉ đọc một câu chào. Tôi muốn dùng voice để hỏi note, nghe câu trả
lời dài, cắt lời giữa chừng và khởi động lại engine nhiều lần. Vì transcript có
thể chứa thông tin riêng, chất lượng âm thanh phải được cân với privacy, latency
và khả năng barge-in.

## Tiêu chí trước khi nghe mẫu

| Tiêu chí | Trọng số | Cách tôi kiểm |
| --- | ---: | --- |
| transcript không rời máy | rất cao | xem provider và request boundary |
| phát âm tiếng Việt | cao | câu có dấu, tên riêng, từ code-mix |
| first audio | cao | đo từ lúc text sẵn sàng đến audio đầu |
| barge-in | cao | cắt ở giữa câu và kiểm audio cũ dừng |
| ổn định | cao | 20 lượt liên tiếp, ghi lỗi riêng |
| tài nguyên | vừa | cold start, RSS, queue |
| giọng tự nhiên | vừa | nghe mù, không biết model trước |

## Tập câu tôi dùng

Tôi giữ cùng một tập câu để tránh chọn model từ một sample nghe hay:

1. “Chào bạn, hôm nay mình kiểm tra một ghi chú ngắn.”
2. “Định lý Bayes cập nhật xác suất khi có bằng chứng mới.”
3. “ONNX Runtime dùng CoreML trước rồi fallback CPU nếu cần.”
4. “Tháng này còn 557 nghìn đồng trong ngân sách thực phẩm.”
5. “Model `google/gemini-3.5-flash-lite` đang chạy ở provider remote.”
6. “Tôi đang nghe, nếu bạn nói tiếp thì câu hiện tại phải dừng.”
7. Một đoạn khoảng 90 giây có dấu phẩy, ngoặc, số và từ viết tắt.

Tôi nghe riêng lỗi nuốt dấu, ngắt câu, phát âm tên file và thời điểm audio cũ
thật sự dừng. “Nghe khá tự nhiên” không thay cho log latency và cancel.

## Kết quả quan sát gần nhất

| Hạng mục | Local | Remote thử nghiệm | Kết luận |
| --- | --- | --- | --- |
| privacy | tốt | phụ thuộc consent/provider | local thắng default |
| cold start | chậm hơn ở lượt đầu | thường nhanh hơn nếu warm | cần hiển thị progress |
| câu ngắn | đủ tự nhiên | tốt hơn một chút | chưa đáng đổi default |
| tên code/file | có lỗi phát âm | tùy model | cần lexicon/test riêng |
| barge-in | kiểm soát được local queue | phụ thuộc stream | local dễ predict hơn |
| mạng chập chờn | không ảnh hưởng | có thể fail giữa câu | remote cần fallback rõ |

Đây là kết luận có độ tin cậy trung bình vì tập câu chưa đủ lớn để gọi là
benchmark. Tôi ghi “đủ cho default hiện tại”, không ghi “local tốt nhất”.

## Trade-off tôi chấp nhận

Local có thể kém mềm mại hơn ở câu dài và lượt đầu có cold start. Tôi chấp nhận
điều đó khi câu chứa dữ liệu riêng hoặc khi mạng không ổn định. Nếu cần đánh giá
chất lượng remote, tôi sẽ bật có chủ ý, nhìn provider/model và không gọi đó là
default privacy.

## Khi nào tôi đổi quyết định

- local phát âm sai lặp lại trên tập câu đã chốt;
- first audio hoặc RSS làm ASR/LLM bị starvation;
- barge-in không dừng sạch sau nhiều lần;
- model local mới có license hoặc lifecycle không phù hợp;
- remote có lợi ích rõ nhưng consent/retention được thiết kế lại.

## Lịch sử

| Ngày | Quan sát | Hành động |
| --- | --- | --- |
| 18/07 | remote nghe hay ở câu mẫu | chưa đổi default |
| 22/07 | local giữ privacy và chạy offline | giữ local |
| 26/07 | cần test code-mix và barge-in | thêm vào tập câu |
| 29/07 | tách cold/warm khỏi quality | ghi benchmark sau, không đoán |

## Liên kết

- `life/decisions/privacy-boundary.md` — boundary dữ liệu.
- `learning/llm/serving-local-remote.md` — mô hình hóa latency và tài nguyên.
- `life/journal/2026-07-23.md` — log lần test provider và ONNX.
