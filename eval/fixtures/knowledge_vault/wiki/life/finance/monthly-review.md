---
type: finance_note
area: monthly-review
period: 2026-07
status: in-progress
created: 2026-07-28
updated: 2026-07-29
confidence: medium
tags: [finance, review, actuals, planned, july-2026]
source_kind: redacted-personal-note
---

# Review chi tiêu giữa/cuối tháng 07/2026

## Tôi không muốn review bằng cảm giác

Cuối tháng rất dễ nhớ những lần mua lớn và quên các khoản nhỏ. Vì vậy tôi đọc
receipt ledger trước, rồi mới viết nhận xét. Con số trong note này là snapshot
đến 28/07; ngày 31/07 tôi sẽ chốt lại sau khi các giao dịch pending được xác nhận.

## Snapshot

| Chỉ số | Giá trị | Trạng thái |
| --- | ---: | --- |
| budget food tháng | 2.400.000 | current |
| actual đã có receipt | 1.843.000 | confirmed-to-28/07 |
| còn dưới trần | 557.000 | calculated |
| plan grocery chưa mua | 366.000 | planned |
| projected nếu mua hết plan | 2.209.000 | estimate |
| buffer sau projected | 191.000 | estimate |

Tôi đổi projected từ 280.000 thành 366.000 sau khi mở grocery plan chi tiết.
Điều này không làm actual tăng. Nó chỉ làm buffer dự kiến nhỏ hơn.

## Điều đã ổn

- tách grocery và lunch giúp tôi biết phần nào tăng;
- receipt có ngày và nhóm, không phải một tổng cuối tháng;
- reserve 300.000 chưa bị dùng;
- các món trong nhà được kiểm tra trước khi lập plan mới;
- những khoản chưa có receipt vẫn được để `unknown`.

## Điều chưa ổn

Hai khoản bữa trưa ngày 20 và 26/07 hiển thị pending ở app thanh toán một lúc;
tôi đã ghi theo ngày sử dụng nhưng cần đối chiếu lại sao kê. Một hóa đơn ngày
23/07 có cả giấy lau bếp; tôi tách 32.000 đồng ra ngoài food category thay vì
giữ cả hóa đơn trong grocery.

Plan cuối tháng hiện cao hơn budget còn lại của riêng grocery. Nếu tôi mua hết,
grocery sẽ vượt phần 1.600.000 khoảng 268.000, dù tổng food vẫn dưới trần vì
reserve còn. Tôi cần quyết định rõ đó là chuyển từ reserve hay cắt món, không để
con số tự “hợp thức hóa”.

## Quyết định tạm thời

Tôi ưu tiên mua rau, đậu phụ và cá; tạm bỏ sữa chua. Khi actual cập nhật, tôi sẽ
đánh giá lại. Đây là `decision-for-next-shop`, chưa phải quy tắc lâu dài.

## Việc còn phải làm

1. xác nhận hai giao dịch pending;
2. sửa category của phần đồ gia dụng trong receipt 23/07;
3. ghi actual cuối tháng;
4. ghi món plan bị hủy hoặc chuyển tháng 08;
5. xem tháng sau có cần tách budget “ăn ngoài” khỏi “bữa trưa công việc”.

## Cách assistant cần đọc note này

Nếu hỏi “tôi đã chi bao nhiêu?”, trả actual 1.843.000 và mốc 28/07. Nếu hỏi
“tôi sẽ chi bao nhiêu nếu mua hết?”, trả projected 2.209.000 và nói đó là ước
tính. Nếu hỏi “tôi còn bao nhiêu tiền?”, note này không đủ để suy ra số dư tài
khoản; chỉ có thể nói còn dưới trần food budget.

## Review cuối tháng

Tôi chỉ đổi `status: in-progress` thành `complete` sau khi receipt và pending
được đối chiếu. Nếu thiếu một record, giữ trạng thái mở và ghi rõ thiếu gì.
