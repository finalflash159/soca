# Ghi chú học tập — Định lý Bayes

#learning #notes #bayes #probability

Ngày ghi: 2026-07-18
Slice: learning_notes
Provenance: authored_demo_note — ghi chú học tập minh họa, không phải dữ liệu cá nhân.

## Ý chính

Định lý Bayes cập nhật xác suất của một giả thuyết khi có thêm bằng chứng. Cách
viết thường dùng là:

```text
P(A | B) = P(B | A) * P(A) / P(B)
```

- `P(A)` là prior: xác suất của A trước khi quan sát B.
- `P(B | A)` là likelihood: khả năng thấy B nếu A đúng.
- `P(B)` là evidence: xác suất quan sát B trong toàn bộ các trường hợp.
- `P(A | B)` là posterior: xác suất cập nhật của A sau khi biết B.

## Ví dụ tự kiểm tra

Nếu một bài test có tỷ lệ dương tính giả đáng kể, không thể suy ra ngay rằng
người có kết quả dương tính chắc chắn mắc bệnh. Cần dùng cả tỷ lệ mắc ban đầu
`P(A)` và độ chính xác có điều kiện `P(B | A)`. Đây là điểm dễ nhầm giữa
`P(A | B)` và `P(B | A)`.

## Câu nhắc khi trả lời

Khi giải thích cho người khác, hãy nêu rõ prior, likelihood và evidence trước
khi thay số. Nếu đề bài thiếu một trong các đại lượng, không tự bịa giá trị.
