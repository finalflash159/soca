---
type: finance_note
area: food
period: 2026-07
status: current
created: 2026-07-02
updated: 2026-07-29
confidence: high
tags: [finance, receipts, ledger, actual, july-2026]
source_kind: redacted-personal-note
---

# Sổ receipt thực phẩm tháng 07/2026

## Quy ước

Mỗi dòng dưới đây là một khoản đã ghi nhận từ receipt hoặc lịch sử thanh toán.
Tôi giữ `receipt_id`, ngày sử dụng, nhóm và trạng thái đối chiếu. Sổ này chỉ ghi
actual; kế hoạch nằm ở `grocery-plan-2026-07.md`.

## Các khoản đã ghi

| Receipt | Ngày | Nhóm | Nội dung rút gọn | Số tiền | Đối chiếu |
| --- | --- | --- | --- | ---: | --- |
| R-0702-01 | 02/07 | grocery | rau, trứng, đậu phụ | 186.000 | matched |
| R-0705-01 | 05/07 | grocery | cá, rau, trái cây | 244.000 | matched |
| R-0708-01 | 08/07 | lunch | cơm khi ở ngoài | 78.000 | matched |
| R-0711-01 | 11/07 | grocery | gạo, thịt, rau | 312.000 | matched |
| R-0713-01 | 13/07 | lunch | bữa trưa di chuyển | 82.000 | matched |
| R-0716-01 | 16/07 | grocery | sữa chua, yến mạch, trái cây | 198.000 | matched |
| R-0720-01 | 20/07 | lunch | bữa trưa di chuyển | 86.000 | pending-check |
| R-0723-01 | 23/07 | grocery | cá, rau, đậu, đồ gia dụng tách riêng | 298.000 | matched |
| R-0726-01 | 26/07 | lunch | bữa trưa di chuyển | 95.000 | pending-check |
| R-0728-01 | 28/07 | grocery | rau, trứng, trái cây | 264.000 | matched |

Tổng grocery là 1.502.000 đồng. Tổng lunch là 341.000 đồng. Tổng actual trong
phạm vi budget là 1.843.000 đồng. Hai dòng `pending-check` đã được ghi theo ngày
dùng nhưng còn phải nhìn sao kê cuối kỳ; không được xóa chỉ vì app thanh toán
chưa hiển thị ngay.

## Tách hóa đơn 23/07

Hóa đơn 23/07 tổng 330.000 đồng. Trong đó 298.000 là thực phẩm; 32.000 là giấy
lau bếp. Tôi đưa phần giấy lau bếp ra khỏi food budget. Nếu sau này category
chung được dùng cho household, tôi sẽ tạo ledger khác thay vì nhập lại dòng cũ.

## Kiểm tra tổng

```text
grocery = 186000 + 244000 + 312000 + 198000 + 298000 + 264000
        = 1.502.000
lunch   = 78000 + 82000 + 86000 + 95000
        = 341.000
actual  = 1.502.000 + 341.000
        = 1.843.000
```

Con số này phải khớp `food-budget-2026-07.md`. Nếu không khớp, tôi mở correction
note, không sửa một bên cho xanh tổng.

## Những điều tôi không thể kết luận từ ledger

- không biết số dư tài khoản tổng;
- không biết khoản ăn ngoài phạm vi food budget;
- không biết chất lượng dinh dưỡng từ tên món ngắn;
- không biết món nào đã được ăn hết nếu không có journal;
- không biết plan cuối tháng đã mua nếu chưa có receipt mới.

## Việc tiếp theo

Ngày 31/07 tôi sẽ đối chiếu hai giao dịch pending, ghi receipt cuối tháng nếu có,
và khóa sổ. Giao dịch phát sinh sau khi khóa sẽ vào note tháng 08, không backdate
vào note này chỉ để làm tháng 07 đẹp hơn.
