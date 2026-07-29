---
type: finance_note
area: food
period: 2026-07
status: partial-actual
created: 2026-07-23
updated: 2026-07-29
tags: [finance, food, ledger, actual, receipt, uncertainty]
source_kind: sanitized-life-vault-simulation
---

# Ledger thực phẩm tháng 07/2026 — actual có receipt mới được tính

## Phạm vi

Đây là ledger mô phỏng để phân biệt số đã chi với kế hoạch. Mục tiêu ngân sách
được ghi ở `food-budget-2026-07.md`; danh sách sẽ mua ở
`grocery-plan-2026-07.md`. Hai note đó không phải receipt.

Tôi dùng các cột: ngày, nhóm, mô tả ngắn, số tiền, trạng thái chứng từ và ghi chú.
Số tiền trong fixture không phải giao dịch thật của người dùng nào.

| Ngày | Nhóm | Mô tả | Số tiền (VND) | Receipt | Trạng thái |
| --- | --- | --- | ---: | --- | --- |
| 2026-07-03 | protein | trứng và đậu phụ | 186.000 | R-0703 | actual |
| 2026-07-06 | rau | rau xanh, nấm | 142.000 | R-0706 | actual |
| 2026-07-10 | pantry | yến mạch, gạo | 238.000 | R-0710 | actual |
| 2026-07-14 | protein | cá và thịt nạc | 326.000 | R-0714 | actual |
| 2026-07-18 | rau | rau củ đông lạnh | 177.000 | R-0718 | actual |
| 2026-07-23 | pantry | sữa chua, gia vị | 214.000 | R-0723 | actual |
| 2026-07-27 | planned | đồ trong grocery plan | 0 | — | planned |

## Cách tôi đọc ledger

Tổng actual của các dòng có receipt là 1.283.000 đồng. Dòng planned không được
cộng vào actual. Nếu câu hỏi là “đã chi bao nhiêu”, assistant phải lấy actual và
nói phạm vi receipt. Nếu câu hỏi là “còn ngân sách bao nhiêu”, nó cần đối chiếu
với budget, nêu rằng ledger có thể chưa đủ và không biến planned thành đã mua.

Tôi không muốn làm tròn im lặng. Nếu một receipt bị thiếu, ghi `unknown` thay vì
ước lượng. Nếu cửa hàng gộp nhiều nhóm, tôi giữ category `mixed` hoặc ghi cách
phân bổ; không tạo độ chính xác giả chỉ để tổng category đẹp.

## Những gì ledger chưa chứng minh

- chưa chứng minh mọi giao dịch ăn uống ngoài nhà đã được nhập;
- chưa chứng minh các dòng actual là hóa đơn cá nhân thật;
- chưa chứng minh phần tiền còn lại có thể chi tiêu tự do;
- không phải hướng dẫn dinh dưỡng hoặc kế hoạch y tế;
- không nên được dùng để suy ra thu nhập, thói quen hay đánh giá cá nhân.

## Quy trình cập nhật

1. Giữ receipt id hoặc ghi rõ thiếu receipt.
2. Nhập actual sau khi giao dịch hoàn tất.
3. Nếu mua khác kế hoạch, thêm dòng mới thay vì sửa planned thành actual.
4. Cuối tuần đối chiếu tổng ledger với app/ngăn ngân sách.
5. Cuối tháng đóng kỳ và ghi adjustment riêng.

Nếu phát hiện nhập nhầm, tôi không xóa dòng cũ. Tôi thêm correction với ngày sửa,
lý do và liên kết dòng gốc. Cách này giúp assistant không kể lại history đã được
viết lại như thể ngay từ đầu đã đúng.

## Câu hỏi để kiểm retrieval

- “actual food spending tháng 07 là bao nhiêu?” → ledger;
- “tuần cuối định mua gì?” → grocery plan;
- “ngân sách ban đầu là bao nhiêu?” → food budget;
- “khoản nào thiếu chứng từ?” → ledger và unknown fields;
- “tôi có nên đổi chế độ ăn không?” → không suy ra từ ledger, cần health boundary.

## Tóm tắt

Ledger là bằng chứng giao dịch theo phạm vi, không phải câu chuyện hoàn chỉnh về
đời sống. Tôi muốn assistant bảo toàn trạng thái actual/planned/unknown, vì sai
khác một chữ ở đây có thể khiến câu trả lời nghe rất cụ thể nhưng hoàn toàn sai.
