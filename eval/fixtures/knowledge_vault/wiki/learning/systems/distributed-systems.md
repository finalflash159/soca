---
type: learning_note
domain: systems
topic: distributed-systems
status: active
created: 2026-07-17
updated: 2026-07-28
tags: [systems, distributed, consistency, retry, idempotency, observability]
source_kind: personal-study-simulation
---

# Distributed systems — nhiều máy không phải một máy chậm hơn

## À thì ra khó ở đâu

Tôi từng nghĩ gọi service khác chỉ là gọi function qua HTTP. Trong hệ phân tán,
message có thể đến trễ, đến hai lần, sai thứ tự hoặc không đến. Clock giữa máy
không hoàn toàn giống nhau. Một node có thể đã commit nhưng client chưa nhận
response.

Vì vậy “request fail” không đồng nghĩa “server chưa làm”. Đây là lý do retry một
side effect mà không có idempotency key có thể tạo giao dịch hai lần.

## Latency budget

Tôi chia một request thành:

```text
client queue
  → DNS/connect/TLS
  → gateway
  → service A
  → database/cache
  → service B
  → serialization
  → client render
```

Nếu tổng budget là 2 giây, không thể cho mọi service tự timeout 2 giây. Timeout
phải có deadline truyền xuống và mỗi hop tiêu thụ một phần. Retry cũng dùng cùng
deadline, không reset đồng hồ.

## Failure modes tôi ghi lại

| Failure | Tôi không được kết luận vội |
| --- | --- |
| timeout | server chưa làm hoặc đã làm nhưng response mất |
| connection reset | request có thể đã tới server |
| 503 | quá tải, deploy, dependency hoặc circuit open |
| duplicate response | retry hoặc consumer redelivery |
| stale read | replica lag hoặc cache chưa invalidated |
| out-of-order event | queue partition/key hoặc concurrent producer |

Mỗi failure cần error code, retryable, committed và receipt; không biến mọi
exception thành một string “failed” rồi tiếp tục như thành công.

## Idempotency

Idempotency nghĩa gửi cùng operation nhiều lần vẫn có cùng effect cuối. Key phải
đại diện cho ý định, không chỉ request UUID mới mỗi lần retry. Server lưu trạng
thái key đủ lâu và trả lại receipt cũ nếu nhận duplicate.

Tôi phân biệt idempotent với safe: GET thường không đổi state nhưng vẫn có thể
đắt; PUT có thể idempotent nếu cùng body, còn POST cần contract riêng.

## Consistency không phải một nút on/off

Strong consistency giúp read thấy write theo thứ tự mong muốn nhưng tốn latency
và coordination. Eventual consistency cho throughput/availability tốt hơn nhưng
user có thể thấy dữ liệu cũ một lúc.

Tôi hỏi “đọc sau ghi nào cần nhất quán?” thay vì hỏi hệ thống có consistent không.
Receipt sau write cần read-your-writes; dashboard thống kê có thể chấp nhận lag.

## Retry và circuit breaker

Retry chỉ hợp lý khi lỗi transient và operation an toàn. Backoff có jitter để
không làm cả fleet retry cùng lúc. Một retry budget chung tốt hơn việc mỗi layer
retry riêng, vì 3 layer x 3 retry biến một request thành 27 request.

Circuit breaker có closed, open và half-open. Open không phải mất dữ liệu; nó là
cách fail fast để dependency hồi phục. Fallback phải nói rõ degraded, không trả
dữ liệu cũ như thể mới.

## Queue và consumer

Queue tách producer khỏi consumer nhưng tạo vấn đề duplicate và lag. Tôi theo
dõi queue depth, age của message lâu nhất, success rate và dead-letter count.

Consumer nên commit offset sau khi xử lý thành công. Nếu crash giữa xử lý và
commit, message có thể chạy lại; đó là lý do handler cần idempotent hoặc lưu
deduplication key.

## Dữ liệu và schema

Schema event phải có version, event ID, occurred-at, producer và correlation ID.
Tôi không dùng timestamp để dedup vì hai event hợp lệ có thể cùng millisecond.

Thay đổi schema nên backward-compatible: thêm optional field trước, không đổi ý
nghĩa field cũ rồi mong consumer cũ tự hiểu.

## Observability

Một trace tốt nối được client request, service call, tool call và database query.
Log cần có correlation ID nhưng không được đẩy API key hoặc transcript nhạy cảm.

Tôi dùng ba lớp:

- metric: biết tần suất và phân vị;
- log: biết event cụ thể;
- trace: biết đường đi qua nhiều component.

Không có trace thì một câu “remote chậm” không cho biết chậm ở provider, mạng,
prompt serialization hay queue.

## Cách tôi thiết kế action trong assistant

Tool read-only có thể retry hạn chế. Tool local side-effect cần fingerprint,
approval và receipt. Public update “đang xử lý” không phải bằng chứng action đã
commit. Terminal outcome chỉ phát sau khi controller verify observation.

## Câu hỏi tự kiểm

1. request timeout thì effect có thể đã commit chưa?
2. retry này có idempotency key không?
3. deadline còn bao nhiêu cho dependency?
4. dữ liệu stale có chấp nhận được không?
5. consumer crash ở bước nào và message có chạy lại không?
6. metric nào cho biết queue đang phục hồi hay chỉ đang phình to?

## Tóm tắt

Hệ phân tán không đáng sợ vì có nhiều máy; nó khó vì không có một góc nhìn duy
nhất về sự thật. Tôi thiết kế quanh timeout, duplicate, thứ tự, receipt và
observability thay vì giả định mạng luôn hoạt động như một function call.

## Retry không phải một dòng `except`

Tôi chỉ retry khi lỗi có khả năng tạm thời và operation an toàn hoặc idempotent.
Timeout không nói server chưa xử lý; request có thể đã commit nhưng response
không về. Retry một payment/create không có idempotency key có thể tạo duplicate.

Mỗi retry cần budget: số lần, tổng thời gian, backoff và jitter. Nếu tất cả client
retry cùng lúc sau một outage, hệ thống bị thundering herd. Circuit breaker có thể
ngăn gọi thêm, nhưng phải có trạng thái half-open và metric để biết khi nào thử
lại.

## Độ nhất quán cần nói bằng user-visible behavior

Strong consistency, eventual consistency và read-your-writes không chỉ là tên
kiến trúc. Nếu user vừa approve memory proposal, họ kỳ vọng lượt kế tiếp thấy
thay đổi; nếu replica chưa catch up, UI phải tránh hứa “đã lưu ở mọi nơi”.

Tôi ghi rõ source of truth, lag chấp nhận được, conflict policy và cách hiển thị
stale. Một cache stale có thể tốt hơn lỗi, nhưng không tốt nếu dùng để quyết định
side effect.

## Time và ordering

Wall clock giữa máy có thể lệch, nên timestamp không luôn đủ để ordering. Logical
sequence, event id hoặc database commit order có thể phù hợp hơn. Tôi không sort
journal bằng timestamp client nếu có thể có clock skew mà không báo.

## Idempotency và exactly-once

Exactly-once end-to-end thường là claim quá mạnh. Tôi thiết kế at-least-once với
deduplication key, idempotent handler và durable receipt. Khi handler crash sau
side effect nhưng trước ack, message chạy lại; handler phải nhận ra duplicate.

Idempotency key cần scope và TTL rõ. Dùng nội dung request làm key có thể sai nếu
hai thao tác giống payload nhưng ý nghĩa khác; dùng random key nhưng không persist
thì retry không nhận ra chính request cũ.

## Failure matrix tôi hay viết

| Failure | Điều có thể đã xảy ra | Hành vi an toàn |
| --- | --- | --- |
| timeout trước response | server chưa/chắc đã commit | kiểm tra receipt trước retry |
| 500 sau commit | side effect có thể tồn tại | retry bằng idempotency key |
| duplicate message | consumer đã xử lý lần đầu | dedupe và ack |
| replica stale | write đã commit ở primary | đọc source phù hợp hoặc báo stale |
| worker chết giữa job | artifact partial | generation/transaction rollback |
| clock lệch | thứ tự hiển thị sai | sequence/event id |

## Observability không chỉ là log

Trace id nối các hop; metric cho rate/error/latency/queue; log có context nhưng
không lộ secret/chunk riêng tư. Tôi muốn biết request bị chậm ở network, queue,
provider hay synthesis. “Service A ok” không đủ nếu user vẫn nhận empty response.

Tôi ghi retry count, idempotency key fingerprint, outcome và reason code. Không ghi
raw prompt/response vào log mặc định nếu nó chứa dữ liệu cá nhân.

## Bài tập và test

- mô phỏng response mất sau commit;
- chạy duplicate event và kiểm tra output chỉ có một side effect;
- tạo replica lag rồi kiểm tra UI nói stale;
- kill worker sau khi ghi staging nhưng trước pointer swap;
- test backoff có jitter và tổng thời gian bị giới hạn;
- kiểm tra trace không chứa API key hoặc raw note.
