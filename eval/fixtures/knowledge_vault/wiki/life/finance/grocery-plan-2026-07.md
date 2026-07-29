---
type: finance_note
area: food
period: 2026-07
status: planned
created: 2026-07-27
updated: 2026-07-29
confidence: medium
tags: [grocery, plan, food, planned, july-2026]
source_kind: redacted-personal-note
---

# Danh sách mua cuối tháng 07/2026

## Đây là kế hoạch, chưa phải receipt

Tôi viết note này sau khi nhìn tủ lạnh tối 28/07. Những dòng dưới đây chỉ là
`planned`; không được dùng để trả lời “tôi đã mua gì”. Khi mua xong, tôi sẽ thêm
receipt vào `receipt-ledger-2026-07.md` và đổi trạng thái từng dòng.

## Đồ cần mua

| Món | Lượng dự kiến | Giá dự kiến | Mục đích | Trạng thái |
| --- | ---: | ---: | --- | --- |
| rau xanh theo mùa | 3 bó | 90.000 | 3–4 bữa | planned |
| đậu phụ | 4 miếng | 32.000 | bữa nhanh | planned |
| cá | 0,8 kg | 130.000 | hai bữa | planned |
| chuối | 1 nải nhỏ | 35.000 | bữa phụ | planned |
| sữa chua không đường | 4 hộp | 44.000 | dùng trong tuần | planned |
| hành, gừng, gia vị thiếu | 1 set nhỏ | 35.000 | nấu ăn | planned |
| **tổng dự kiến** |  | **366.000** |  |  |

Tổng dự kiến cao hơn con số 280.000 trong budget vì tôi ghi cả một lần mua dự
phòng. Tôi sẽ ưu tiên rau, đậu phụ và cá; nếu giá thực tế vượt 280.000, bỏ sữa
chua hoặc mua cá ít hơn thay vì coi phần chênh là “không đáng kể”.

## Vì sao tôi chọn như vậy

Tôi không lập thực đơn lý tưởng cho cả tháng. Tôi lập kế hoạch đủ cho vài ngày,
dựa trên những gì còn lại và lịch di chuyển. Gạo và yến mạch đang có nên không
mua lại. Trứng còn sáu quả, vì vậy chưa cần thêm trừ khi giá tốt.

Tôi ưu tiên món có thể ghép nhiều bữa: rau + đậu phụ cho bữa nhanh, cá chia
phần cho hai bữa, chuối dùng ngay trước khi hỏng. Đây là quyết định tiết kiệm
thời gian và giảm bỏ phí, không phải khuyến nghị dinh dưỡng cá nhân.

## Quy tắc chuyển plan thành actual

- `planned`: ý định, chưa có bằng chứng mua;
- `purchased`: đã mua nhưng receipt chưa được đối chiếu;
- `actual`: có dòng receipt và số tiền thật;
- `cancelled`: không mua, không cộng vào chi tiêu;
- `carried-over`: chuyển sang tháng sau, không cộng tháng này.

Nếu mua một món thay thế, tôi không xóa món plan cũ. Tôi ghi `planned →
cancelled` và thêm món mới cùng receipt. Như vậy sau này câu “tôi định mua gì”
vẫn khác “tôi thực sự mua gì”.

## Cách tôi sẽ review ngày 31/07

1. đối chiếu từng dòng với receipt;
2. tách đồ ăn khỏi đồ gia dụng nếu cùng hóa đơn;
3. tính actual grocery và so với budget note;
4. ghi món không mua và lý do;
5. nếu có đồ hỏng, ghi ở monthly review chứ không sửa lịch sử.

## Query kiểm tra

- “Cuối tháng tôi định mua gì?” → note này.
- “Tôi đã mua cá chưa?” → phải kiểm ledger, không trả `planned` như actual.
- “Tôi còn bao nhiêu budget?” → budget note, không cộng tổng dự kiến vào actual.
- “Tại sao bỏ sữa chua?” → chỉ trả lời sau khi status được cập nhật.
