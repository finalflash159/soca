---
type: life_decision
area: privacy
status: current
created: 2026-07-19
updated: 2026-07-29
confidence: high
tags: [privacy, local, remote, memory, api-key, data-flow]
source_kind: redacted-personal-note
---

# Ranh giới riêng tư khi dùng assistant

## Vì sao tôi không dùng chữ “local-first” như một khẩu hiệu

Tôi có thể lưu API key ở máy nhưng vẫn gửi transcript ra provider. Tôi có thể
chạy retrieval local nhưng gửi cả retrieved note vào prompt remote. Hai việc đó
không mâu thuẫn, nhưng nếu UI chỉ ghi `remote` thì tôi không biết dữ liệu nào vừa
đi qua boundary.

Vì vậy trước khi bật remote, tôi muốn trả lời được ba câu: provider/model nào sẽ
nhận request, transcript nào được gửi, và context nào được đưa vào request. Nếu
không trả lời được, tôi coi config là chưa đủ rõ để dùng.

## Ma trận dữ liệu tôi áp dụng

| Dữ liệu | Mặc định | Có thể rời máy? | Điều kiện |
| --- | --- | --- | --- |
| câu chat hiện tại | RAM + provider local | có | user bật provider remote |
| transcript voice | RAM + engine local | có | cùng consent với chat |
| knowledge snippet | index local | có | chỉ khi prompt remote cần evidence |
| working memory | session local | có | không gửi nếu turn không cần |
| core/pinned memory | local state | có | phải relevant với prompt |
| API key | secure local config | không như nội dung | không in vào prompt/log |
| vector/index metadata | cache private | không | chỉ local filesystem |

Tôi không dùng “API key đã lưu” để suy ra “transcript an toàn”. Key chỉ là quyền
gọi provider; chính payload mới là dữ liệu cần consent.

## Quy tắc vận hành

1. local là default cho chat và voice.
2. remote phải hiện provider, model, reasoning state và max output.
3. API key hiển thị masked; khi thay hoặc xóa, ô nhập và persisted config cùng đổi.
4. key không đi vào Markdown, exception, trace hay screenshot debug.
5. knowledge và memory được đánh dấu là data không tin cậy trong prompt.
6. memory dài hạn chỉ nhận proposal, không tự ghi từ một câu model nói.
7. khi không có realtime tool, assistant nói rõ chưa kiểm tra hiện tại.
8. path ngoài vault, side effect chưa approve hoặc provider mơ hồ đều phải dừng.

## Những lần tôi đã kiểm tra

| Tình huống | Điều phải nhìn thấy |
| --- | --- |
| chọn OpenRouter | status nói rõ transcript gửi OpenRouter/model cụ thể |
| voice sau khi đổi provider | voice dùng cùng setting hoặc UI nói rõ nếu khác |
| hỏi note riêng tư | citation/path hiển thị, không dump cả file vào UI |
| xóa API key | input trống, config trống, không còn bản sao trong history |
| provider lỗi | lỗi provider không bị đổi thành câu trả lời thành công |
| index crash | cache mới không làm mất generation trước |

## Threat model tôi dùng ở mức desktop

Tôi không giả định máy luôn an toàn tuyệt đối. Người khác có thể đọc file nếu
permission sai; log có thể vô tình chứa prompt; extension có thể chụp terminal;
provider có thể lưu request theo chính sách riêng. Vì vậy index có nội dung note
phải private, trace phải redact, và remote phải là hành động nhìn thấy được.

Tôi chưa giải quyết toàn bộ threat model của hệ điều hành, nhưng ít nhất không
để status nói “không cloud” trong khi model đang remote. Chữ trên UI là một phần
của security boundary vì nó quyết định user có hiểu hành động hay không.

## Điều còn để mở

- retention transcript remote phụ thuộc policy từng provider và chưa tự kiểm chứng;
- approval UI cho memory proposal cần hiển thị evidence trước khi ghi;
- trace debug cần mặc định ẩn query nhạy cảm thay vì chỉ ẩn API key;
- nếu voice/chat dùng provider khác nhau, status cần hiện ở từng lượt.

## Lịch sử quyết định

| Ngày | Quyết định | Lý do |
| --- | --- | --- |
| 19/07 | local-first | privacy và khả năng chạy offline |
| 24/07 | hiển thị provider/model trong status | “remote” quá mơ hồ |
| 28/07 | áp dụng remote setting cho chat + voice | tránh UI hứa một config nhưng runtime dùng config khác |
| 29/07 | tách key boundary khỏi transcript boundary | key local không làm payload local |

## Kết luận

Tôi chấp nhận dùng remote khi lợi ích chất lượng đáng giá và tôi biết dữ liệu đi
đâu. Tôi không chấp nhận một default hoặc một status khiến mình tưởng request
đang local trong khi nó không phải.
