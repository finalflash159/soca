---
type: finance_note
area: food
period: 2026-07
status: current
created: 2026-07-01
updated: 2026-07-29
confidence: high
tags: [budget, food, monthly, actuals, july-2026]
source_kind: redacted-personal-note
---

# Ngân sách ăn uống tháng 07/2026

## Tháng này tiền ăn đang đi tới đâu? — snapshot 29/07

Đến hết ngày 28/07, actual trong phạm vi food budget là **1.843.000 đồng** trên
trần **2.400.000 đồng**. Nếu mua hết grocery plan hiện tại, projected total là
**2.209.000 đồng**, còn **191.000 đồng** dưới trần. Đây là snapshot mới nhất của
tháng; các con số trong nhật ký trước ngày 29/07 chỉ là mốc lịch sử, không thay
thế cho actual trong note này.

## Tôi đặt ngân sách thế nào

Đầu tháng tôi đặt trần 2.400.000 đồng cho thực phẩm và các bữa trưa phải ăn
ngoài khi di chuyển. Tôi không gộp cà phê gặp bạn, giao đồ ăn tùy hứng, thực
phẩm chức năng hoặc đồ dùng bếp vào con số này. Tách phạm vi giúp tôi không dùng
budget để phán xét những khoản vốn không thuộc budget.

| Nhóm | Trần tháng | Actual đến 28/07 | Planned còn lại | Ghi chú |
| --- | ---: | ---: | ---: | --- |
| chợ/siêu thị | 1.600.000 | 1.502.000 | 366.000 | planned lấy từ grocery plan |
| bữa trưa khi di chuyển | 500.000 | 341.000 | 0 | chưa có plan thêm |
| dự phòng | 300.000 | 0 | 0 | chưa dùng |
| **tổng** | **2.400.000** | **1.843.000** | **366.000** | projected 2.209.000 |

`Actual` chỉ lấy từ receipt ledger. `Planned` không được cộng vào actual. Nếu
cuối tháng tôi không mua theo plan, actual vẫn giữ nguyên và plan chuyển sang
trạng thái cancelled hoặc carried-over.

## Tình hình hiện tại

Đã ghi nhận 1.843.000 đồng, còn 557.000 đồng dưới trần. Grocery plan cuối tháng
dự kiến 366.000 đồng, nên projected total là 2.209.000 đồng và còn buffer 191.000
đồng. Đây là phép chiếu, chưa phải số đã tiêu. Snapshot này đã được cập nhật
theo grocery plan chi tiết ngày 29/07; các nhật ký trước đó chỉ là mốc lịch sử.

Tôi không lấy “còn buffer” làm lý do mua thêm. Ngày 29/07 tôi kiểm tra đồ còn ở
nhà trước: gạo đủ, trứng còn 6 quả, rau xanh còn ít, yến mạch đủ cho vài bữa,
đậu phụ chưa mua. Kế hoạch vì thế chỉ cần rau, đậu phụ, cá và một ít trái cây.

## Điều làm budget dễ sai

- một receipt siêu thị có cả đồ ăn và đồ dùng, phải tách dòng;
- bữa trưa trả bằng ví điện tử có thể hiện ngày pending khác ngày dùng;
- kế hoạch mua hàng có giá ước tính, không được ghi như actual;
- khoản ăn ngoài với bạn thuộc phạm vi khác nhưng dễ bị nhập nhầm;
- hoàn tiền hoặc voucher phải ghi riêng để không làm lệch tổng;
- nếu thiếu receipt, trạng thái là `unknown`, không tự điền số gần đúng.

## Quy tắc cập nhật tôi đang dùng

1. Ghi receipt trong ngày hoặc đánh dấu `needs-receipt`.
2. Cuối mỗi tuần đối chiếu ledger với số dư nhóm.
3. Không sửa con số cũ để làm projected đẹp hơn; thêm correction.
4. Nếu chuyển tiền giữa nhóm, ghi lý do và ngày chuyển.
5. Cuối tháng khóa actual, sau đó mở note review tháng mới.

## Câu hỏi assistant phải trả lời đúng loại

- “Tháng này tôi đã chi bao nhiêu?” → 1.843.000 actual đến 28/07.
- “Tôi còn bao nhiêu trong budget?” → 557.000 dưới trần, chưa trừ plan.
- “Cuối tháng định mua gì?” → grocery plan, giữ nhãn planned.
- “Có khoản nào thiếu receipt?” → đọc ledger, không suy ra từ budget.
- “Tại sao projected không bằng actual?” → vì projected cộng thêm plan chưa mua.

## Lịch sử thay đổi

| Ngày | Thay đổi | Lý do |
| --- | --- | --- |
| 01/07 | đặt trần 2.400.000 | muốn có reserve riêng |
| 16/07 | tách bữa trưa khỏi grocery | nhìn ra hai nhóm có hành vi khác |
| 23/07 | cập nhật receipt mới | actual thay đổi, trần không đổi |
| 29/07 | cập nhật plan từ 280.000 lên 366.000 | dùng grocery plan chi tiết; phân biệt kế hoạch và số đã tiêu |

## Giới hạn

Note này chỉ nói về phạm vi food budget và không mô tả thu nhập, tài sản hay
nghĩa vụ tài chính khác. Khi hỏi ngoài phạm vi, assistant phải nói không có dữ
liệu thay vì suy luận profile từ một bảng chi tiêu.
