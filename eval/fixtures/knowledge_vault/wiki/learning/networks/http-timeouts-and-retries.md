---
type: learning_note
domain: networking
topic: http-timeouts-retries-idempotency
status: active
created: 2026-07-29
updated: 2026-07-29
confidence: medium
tags: [networking, http, timeout, retry, idempotency, resilience]
source_kind: personal-study-note
---

# Networking: HTTP timeout và retry — request fail chưa nói server chưa làm

## Tôi từng nghĩ HTTP như gọi function

Ở code local, gọi hàm trả về hoặc ném exception. Với HTTP, request đi qua nhiều
boundary: DNS, TCP, TLS, proxy, load balancer, service, database và đường về.
Client có thể timeout trong khi server đã commit. Response có thể mất sau khi
side effect đã xảy ra. Vì vậy `timeout` không đồng nghĩa `not executed`.

Tôi ghi câu này ở đầu note vì nó thay đổi toàn bộ cách thiết kế retry. Nếu retry
một `charge_card` sau timeout mà không có idempotency key, tôi có thể tạo hai
giao dịch dù người dùng chỉ bấm một lần.

## Timeout phải tách theo stage

Một timeout tổng khó debug. Tôi tách:

```text
queue → DNS → connect → TLS → upload → server processing → first byte → body
```

`connect_timeout` bảo vệ lúc chưa kết nối. `read_timeout` bảo vệ lúc đã gửi
nhưng chờ phản hồi. `total_deadline` là thời điểm cuối cùng toàn request được
phép hoàn tất. Các stage phải dùng chung deadline còn lại, không tự reset thành
30 giây ở mỗi hop.

## Ví dụ budget

Nếu user-facing request có deadline 2.000 ms, tôi có thể phân bổ:

| Stage | Budget gợi ý | Điều cần đo |
| --- | ---: | --- |
| queue | 100 ms | bị nghẽn trước call chưa |
| connect/TLS | 300 ms | mạng/DNS/provider |
| upstream | 1.300 ms | service + database |
| đọc body/render | 300 ms | client/UI |

Đây không phải số universal. Nó chỉ làm assumption hiện ra để tôi đo và chỉnh.

## Khi nào retry được

Tôi retry khi có cả bốn điều kiện: lỗi có khả năng tạm thời, deadline còn đủ,
request an toàn hoặc có idempotency key, và retry không tạo duplicate call vô hạn.

| Lỗi | Retry mặc định | Lý do |
| --- | --- | --- |
| DNS tạm lỗi | có giới hạn | chưa biết request đã tới server |
| connect reset trước upload | thường an toàn | server có thể chưa nhận body |
| 429 | theo `Retry-After` | tôn trọng rate limit |
| 503 | backoff có jitter | có thể quá tải |
| 400 schema | không | retry không đổi input |
| 401 key | không | cần đổi config |
| read timeout sau POST | chỉ có idempotency | side effect có thể đã commit |

Backoff exponential giảm synchronized retry, còn jitter tránh hàng nghìn client
retry cùng một thời điểm. Tôi không dùng retry để che latency budget sai.

## Idempotency key

Client tạo key ổn định cho một intent, gửi cùng request retry. Server lưu kết quả
theo key trong thời gian phù hợp. Nếu nhận lại cùng key, server trả kết quả cũ
thay vì chạy side effect lần nữa. Key không thay thế authorization; một key phải
gắn với user/operation để không bị reuse sai boundary.

Với một trợ lý, truy vấn knowledge là read-only nên retry dễ hơn. Ghi memory proposal
hoặc thay config là side effect local; runtime phải có operation ID và không tự
ghi lần hai khi response UI bị mất.

## Circuit breaker và bulkhead

Nếu provider remote lỗi liên tục, mỗi request tự retry sẽ làm tình hình xấu hơn.
Circuit breaker có trạng thái closed, open và half-open. Open chặn call trong một
thời gian, half-open thử ít request để xem service hồi phục chưa.

Bulkhead giới hạn số request/worker cho một dependency. Remote LLM bị kẹt không
được phép chiếm hết worker của local ASR hoặc UI. Đây là quan hệ tài nguyên, không
chỉ là một if quanh exception.

## Quan sát cần ghi

- request ID và operation ID;
- deadline ban đầu và deadline còn lại;
- stage timeout;
- attempt number và lý do retry;
- response status/provider;
- side effect committed/unknown nếu biết;
- tổng latency và outcome cuối.

Tôi không log toàn bộ transcript hoặc API key để debug network. Metadata đủ để
reproduce behavior mà không biến trace thành kho dữ liệu nhạy cảm.

## Failure cases tôi tự test

1. timeout trước khi server nhận request;
2. server commit nhưng response bị drop;
3. 429 có `Retry-After` dài hơn deadline;
4. retry cùng idempotency key;
5. retry khác key do bug caller;
6. request bị hủy khi UI đóng;
7. stream có vài token rồi connection reset;
8. duplicate response đến ngoài thứ tự.

## Cách tôi nhớ

Retry không phải “gọi lại cho đến khi xanh”. Nó là một quyết định dựa trên trạng
thái side effect, deadline và khả năng lặp an toàn. Một hệ thống tốt nói rõ
`unknown whether committed` còn hơn báo `failed` rồi âm thầm chạy lại.

## Câu hỏi còn mở

- provider remote nào hỗ trợ idempotency thực sự và giữ key bao lâu?
- streaming answer nên resume hay bắt đầu lượt mới?
- cancel có cần gửi signal tới upstream hay chỉ dừng render local?

## Bài tập

Viết một client mock có response drop sau commit, chạy retry có và không có key,
sau đó đếm số side effect. Nếu test chỉ assert final HTTP 200, nó sẽ bỏ qua bug
duplicate; invariant đúng là một intent chỉ tạo một side effect.
