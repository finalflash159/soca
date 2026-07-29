---
type: life_decision
area: personal-knowledge
status: current
created: 2026-07-27
updated: 2026-07-29
tags: [knowledge, notes, retrieval, provenance, personal-workflow]
source_kind: sanitized-life-vault-simulation
---

# Vì sao tôi tách learning, life và journal trong vault

## Bối cảnh

Tôi từng để mọi thứ vào một thư mục `notes` rồi mong assistant tự hiểu sự khác
nhau giữa một định nghĩa tôi đang học, một quyết định đã chốt và một chuyện đã
xảy ra hôm qua. Cấu trúc đó tiện lúc bắt đầu nhưng làm retrieval trả lời lẫn
loại evidence. Một journal có thể nhắc tôi đã đọc Bayes; nó không thay thế note
giải thích Bayes. Một grocery plan nói dự định mua; nó không chứng minh receipt.

Tôi tách vault thành ba loại chính:

- `learning`: điều tôi đang hiểu, còn có thể sửa, có ví dụ và câu hỏi mở;
- `life`: quyết định, ledger, sức khỏe phổ thông và ranh giới sử dụng;
- `journal`: sự kiện theo ngày, trạng thái hoàn thành và phản tỉnh tại thời điểm đó.

## Quy tắc evidence tôi muốn giữ

Mỗi note ghi `type`, `status`, ngày tạo/cập nhật và `source_kind`. Date giúp phân
biệt “đã xảy ra” với “định làm”, nhưng không biến mọi câu có ngày thành sự thật
khách quan. `actual`, `planned`, `draft` và `unknown` phải được viết rõ trong
finance hoặc journal.

Một note có thể liên kết note khác, nhưng link không làm hai note cùng loại. Khi
assistant trả lời, tôi muốn citation chỉ tới note thật sự hỗ trợ câu đó và nói rõ
nếu nguồn là journal, decision hay learning. Nếu không có nguồn đủ mạnh, câu trả
lời nên dừng ở “chưa tìm thấy” thay vì nối các mẩu gần nghĩa.

## Vì sao không để project trong showcase

Project notes dễ kéo retrieval về một sản phẩm cụ thể và làm corpus giống tài liệu
được dựng để test code. Tôi vẫn có thể học systems/ML/LLM trong `learning`, nhưng
không muốn `life/project` giả lập một dự án không có lịch sử thật. Các decision và
journal hiện tại đủ để test câu hỏi assistant-like mà không cần vỏ project.

## Cách tôi cập nhật

Tôi viết raw thought trước, sau đó thêm metadata và link. Không sửa trực tiếp
history để làm câu chuyện đẹp hơn; nếu nhận ra mình hiểu sai, tạo section
“correction” hoặc note mới có ngày. Với số tiền, tôi chỉ đổi `actual` sau khi có
receipt. Với sức khỏe, tôi ghi quan sát và câu hỏi cho chuyên gia, không tự biến
note thành chẩn đoán.

Khi một note lớn, tôi tách theo câu hỏi mà tôi sẽ thực sự hỏi assistant. Tôi không
tách mỗi vài dòng thành một file vì làm mất ngữ cảnh và khiến retrieval trả snippet
rụng rời. Ngược lại, một file quá dài không có heading cũng khó đọc và khó cite.

## Điều tôi dùng vault để kiểm tra

- assistant có phân biệt “tôi đã mua” và “tôi dự kiến mua” không;
- câu hỏi “tuần trước tôi học gì” có ưu tiên journal trước learning không;
- câu hỏi “tôi hiểu embedding thế nào” có lấy learning note không;
- câu hỏi sức khỏe có giữ safety boundary không;
- query không có evidence có được trả lời thẳng là chưa có note không;
- sửa một note có làm hit cũ biến mất khỏi index không.

## Quyết định và giới hạn

Đây là corpus demo đã sanitize, không phải nhật ký riêng tư thật. Tôi giữ các
chi tiết vừa đủ để kiểm tra provenance, planned/actual và ambiguity, nhưng không
đưa secret, địa chỉ, thông tin y tế thật hay giao dịch thật vào repository.

Tôi không dùng corpus này làm benchmark release. Benchmark phải có dataset độc
lập, split và nhãn rõ; showcase chỉ dùng cho smoke test, UI demo và kiểm tra tay.

## Tóm tắt

Tách folder không phải để làm cây thư mục đẹp. Nó là một phần của semantics:
retrieval cần biết loại evidence, runtime cần biết confidence và user cần biết
assistant đang nhắc lại note nào. Nếu cấu trúc không biểu đạt được cách tôi nhớ,
thì thêm model mạnh hơn cũng chỉ làm câu trả lời trôi chảy hơn, chưa chắc đúng hơn.
