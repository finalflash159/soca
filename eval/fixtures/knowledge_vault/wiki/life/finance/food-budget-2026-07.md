---
type: finance_note
area: food
period: 2026-07
status: current
created: 2026-07-01
updated: 2026-07-27
tags: [budget, food, monthly, synthetic]
source_kind: sanitized-life-vault-simulation
---

# Ngân sách ăn uống tháng 07/2026

## Mục tiêu

Tôi đặt ngân sách mẫu 2.400.000 đồng cho thực phẩm và bữa ăn tại nhà trong
tháng. Khoản này không gồm cà phê gặp bạn, giao đồ ăn ngoài kế hoạch, thực phẩm
chức năng hoặc mua thiết bị bếp.

## Phân bổ đầu tháng

| Nhóm | Ngân sách | Đã ghi nhận đến 27/07 | Còn lại dự kiến |
| --- | ---: | ---: | ---: |
| chợ/siêu thị | 1.600.000 | 1.182.000 | 418.000 |
| bữa trưa khi di chuyển | 500.000 | 326.000 | 174.000 |
| dự phòng cuối tháng | 300.000 | 0 | 300.000 |
| **tổng** | **2.400.000** | **1.508.000** | **892.000** |

Các số trong fixture là số mô phỏng đã được sanitise để demo. Tôi không dùng note
này như dữ liệu tài chính thật của người dùng.

## Ledger đã ghi

| Ngày | Nhóm | Nội dung | Số tiền |
| --- | --- | --- | ---: |
| 02/07 | chợ/siêu thị | rau, trứng, đậu phụ | 186.000 |
| 05/07 | chợ/siêu thị | cá, rau, trái cây | 244.000 |
| 08/07 | bữa trưa | cơm khi ở ngoài | 78.000 |
| 11/07 | chợ/siêu thị | gạo, yến mạch, sữa chua | 312.000 |
| 14/07 | bữa trưa | hai bữa đi làm | 96.000 |
| 18/07 | chợ/siêu thị | thịt nạc, rau đông lạnh | 220.000 |
| 23/07 | chợ/siêu thị | trứng, rau, trái cây | 220.000 |
| 26/07 | bữa trưa | một bữa khi di chuyển | 152.000 |

## Quy tắc ghi nhận

- receipt chưa có thì đánh dấu pending, không tự bịa số;
- danh sách mua dự kiến không tính là đã chi;
- hoàn tiền ghi một dòng đảo chiều, không xóa lịch sử;
- cuối tháng đối chiếu với bank/receipt thật;
- nếu thay đổi ngân sách, ghi reason và ngày thay đổi.

## Câu hỏi thường gặp

“Ngân sách tháng này là bao nhiêu?” → nhìn mục tiêu đầu tháng, 2.400.000 đồng.

“Đã chi bao nhiêu?” → nhìn ledger đã ghi, 1.508.000 đồng tính đến 27/07.

“Còn bao nhiêu?” → 892.000 đồng theo dữ liệu đã ghi, chưa phải dự báo chắc chắn.

“Chiều nay mua gì?” → xem grocery plan, không lấy nó làm giao dịch đã hoàn tất.

## Review

Tuần cuối tháng tôi sẽ kiểm tra khoản nào thiếu receipt và điều chỉnh phần dự
phòng. Tôi không muốn assistant tự ghi đè note khi chỉ nghe tôi nói “hình như đã
mua rồi”; câu đó cần clarification hoặc proposal.

## Các mốc trong tháng

| Mốc | Budget còn theo kế hoạch | Actual có receipt | Việc cần làm |
| --- | ---: | ---: | --- |
| 01/07 | 2.400.000 | 0 | tạo envelope |
| 10/07 | 1.900.000 | 500.000 | đối chiếu receipt |
| 20/07 | 1.430.000 | 970.000 | kiểm tra nhóm pantry |
| 27/07 | 1.117.000 | 1.283.000 | xem grocery plan và khoản lệch |

Hai cột trên không hoàn toàn cùng nghĩa: “budget còn theo kế hoạch” là target
chưa phân bổ, còn actual là tổng receipt đã nhập. Tôi ghi cả hai để tránh câu
trả lời kiểu lấy 2.400.000 trừ một con số không cùng thời điểm.

## Cách tôi xử lý khoản lệch

Nếu actual vượt kế hoạch, tôi kiểm tra receipt thiếu category, khoản mua ngoài
list và khoản mua cho nhiều tuần. Không tự kết luận “chi tiêu thất bại”. Nếu một
receipt thuộc hai kỳ, ghi allocation và link về receipt gốc. Nếu chưa chắc, giữ
`unknown` và đưa vào monthly review.

## Contract cho assistant

- câu “ngân sách ban đầu” → 2.400.000 đồng;
- câu “đã chi có receipt” → ledger actual;
- câu “dự kiến mua tuần cuối” → grocery plan;
- câu “còn chính xác bao nhiêu” → nêu thiếu dữ liệu trước khi tính;
- câu hỏi dinh dưỡng → không dùng budget làm health advice.

Một câu trả lời tốt phải nói ngày cập nhật, phạm vi và distinction actual/planned.
