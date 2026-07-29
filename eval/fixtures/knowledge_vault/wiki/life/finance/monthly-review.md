---
type: finance_note
area: review-process
period: monthly
status: active
created: 2026-07-03
updated: 2026-07-27
tags: [budget, review, receipt, process]
source_kind: sanitized-life-vault-simulation
---

# Cách tôi review chi tiêu cuối tháng

## Tôi không muốn review kiểu nào

Tôi không muốn nhìn một con số tổng rồi tự kể một câu chuyện. Tổng thấp có thể
do thiếu receipt, tổng cao có thể do một khoản bất thường. Vì vậy review cần giữ
ledger, category, thời gian và uncertainty.

## Quy trình sáu bước

1. gom receipt và giao dịch ngân hàng;
2. normalize tên merchant nhưng giữ raw description;
3. gán category và đánh dấu `needs_review` khi không chắc;
4. đối chiếu với budget tháng;
5. ghi variance và lý do, không chỉ ghi “vượt”;
6. tạo kế hoạch tháng sau riêng, không sửa lịch sử tháng cũ.

## Các câu hỏi tôi dùng

- khoản nào lặp lại?
- khoản nào một lần?
- khoản nào chưa có bằng chứng?
- danh sách planned nào đã thành purchased?
- variance có do thay đổi giá, số lượng hay category?
- budget tháng sau cần đổi vì dữ liệu hay vì cảm giác?

## Quy tắc provenance

Số “đã chi” phải đến từ ledger/receipt. Số “dự kiến” phải đến từ plan. Nếu user
nói “chắc khoảng”, assistant nên giữ uncertainty hoặc hỏi lại, không biến thành
con số chính xác trong memory.

## Output tôi muốn

Một review tốt có bảng actual/budget/variance, danh sách unknown và vài hành động
nhỏ cho tháng tới. Nó không phán xét thói quen, không tự thay đổi note và không
đưa lời khuyên đầu tư từ một tháng tiền ăn.

## Mẫu review tôi sẽ điền

### 1. Phạm vi và độ tin cậy

- kỳ: 07/2026;
- số receipt đã nhập: 6;
- ngày cuối cùng đã đối chiếu: 23/07;
- dòng planned chưa tính actual: grocery plan cuối tháng;
- unknown cần kiểm tra: giao dịch ngoài nhà và receipt thiếu.

### 2. So sánh

Tôi đặt budget cạnh actual theo cùng kỳ và cùng category. Nếu category không
khớp, ghi rõ mapping thay vì ép số. Variance có thể do timing, mua cho nhiều
tuần, hoặc dữ liệu thiếu. Một câu “vượt 10%” không đủ nếu không nói denominator
và phạm vi.

### 3. Hành động nhỏ

- đóng các receipt còn thiếu;
- đánh dấu các khoản mua cho tuần sau;
- xem pantry tồn trước khi tạo grocery plan mới;
- giữ một buffer cho giá biến động;
- không thay đổi nhiều thói quen chỉ từ một tháng.

## Bài học về uncertainty

Tôi dễ bị hấp dẫn bởi con số tổng vì nó có vẻ khách quan. Nhưng ledger có thể
thiếu một giao dịch hoặc gộp sai category. Vì vậy review phải có mục “chưa biết”.
Assistant được phép nói “chưa đủ dữ liệu để tính chính xác”; đó là output tốt hơn
một phép trừ có vẻ chính xác nhưng dựa trên hai nguồn khác kỳ.
