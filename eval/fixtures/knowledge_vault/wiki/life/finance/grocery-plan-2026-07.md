---
type: finance_note
area: food
period: 2026-07-week-4
status: planned
created: 2026-07-24
updated: 2026-07-27
tags: [grocery, plan, food, pending]
source_kind: sanitized-life-vault-simulation
---

# Danh sách mua thực phẩm tuần cuối tháng 07

## Vì sao lập danh sách

Tuần cuối tháng tôi muốn dùng phần còn lại của budget để mua đủ đồ cơ bản, nhưng
không biến kế hoạch thành giao dịch. Note này phải được trả lời khác với
`food-budget-2026-07.md` nếu tôi hỏi “đã chi bao nhiêu”.

## Danh sách và mức dự kiến

| Mặt hàng | Lượng dự kiến | Mức dự kiến | Trạng thái |
| --- | --- | ---: | --- |
| trứng | 1 khay | 80.000 | planned |
| đậu phụ | 4 bìa | 24.000 | planned |
| rau xanh | 4–5 bó | 120.000 | planned |
| trái cây | 2–3 loại | 180.000 | planned |
| yến mạch | 1 gói | 95.000 | planned |
| sữa chua không đường | 1 lốc | 70.000 | planned |
| cá hoặc thịt nạc | 1–1,5 kg | 220.000 | planned |

Tổng dự kiến khoảng 789.000 đồng, nhưng giá và lượng thật có thể thay đổi.

## Sau khi mua

Tôi sẽ ghi ngày, receipt, số tiền thật và mặt hàng thay thế. Nếu không mua một
món, không xóa dòng; chuyển thành `skipped` và ghi lý do. Nếu mua ngoài list,
thêm dòng `added` để monthly review biết vì sao lệch.

## Điều không được suy ra

Note này không chứng minh tôi đã mua hàng, không chứng minh còn budget chính xác
và không phải thực đơn y tế. Assistant phải giữ chữ “dự kiến” khi trả lời.

## Kế hoạch theo nhóm

| Nhóm | Mặt hàng | Lý do | Ước tính | Trạng thái |
| --- | --- | --- | ---: | --- |
| protein | trứng | dùng cho bữa sáng, dễ chia | 120.000 | planned |
| rau | rau lá và cà rốt | đủ cho vài bữa, ưu tiên ít lãng phí | 160.000 | planned |
| pantry | đậu, yến mạch | bổ sung phần đã gần hết | 210.000 | planned |
| fruit | chuối | ăn trong tuần, không mua quá nhiều | 80.000 | planned |

Tổng ước tính chỉ là planning number. Giá lúc mua, trọng lượng và món thay thế
có thể khác. Tôi không muốn assistant cộng nó vào ledger actual hay nói “đã mua
trứng” chỉ vì note có tên trứng.

## Điều kiện dừng hoặc đổi kế hoạch

Nếu budget actual đã gần trần, tôi giảm món pantry không cấp thiết trước. Nếu nhà
còn tồn, đánh dấu `defer` thay vì mua cho đủ list. Nếu cửa hàng hết món, thêm
`substitute` với món thực tế và ghi lý do. Nếu không mua, giữ dòng `skipped` để
monthly review thấy kế hoạch đã thay đổi.

## Câu hỏi assistant nên hỏi lại

- Bạn hỏi kế hoạch hay các món đã mua?
- Bạn muốn tổng ước tính hay tổng actual có receipt?
- Có cần tính món thay thế và đồ còn tồn không?

Ba câu hỏi này nghe chậm hơn một câu trả lời ngay, nhưng tránh trộn plan với
evidence. Với fixture, đó là distinction quan trọng để test grounding.
