---
type: life_decision
area: personal-knowledge
status: current
created: 2026-07-27
updated: 2026-07-29
confidence: high
review_after: 2026-09-01
tags: [knowledge, notes, retrieval, provenance, personal-workflow]
source_kind: redacted-personal-note
---

# Vì sao tôi tách learning, life và journal trong vault

## Điều buộc tôi phải đổi cấu trúc

Trước đây tôi để định nghĩa, log công việc, quyết định và danh sách mua hàng vào
một thư mục `notes`. Lúc số file còn ít, tôi nhớ bằng tên file. Khi số note tăng,
assistant nhìn thấy cùng một keyword ở nhiều nơi nhưng không biết loại evidence
nào đang trả lời câu hỏi.

Một journal ngày 20/07 có thể chứng minh hôm đó tôi đã học Bayes. Nó không thay
thế cho note giải thích công thức. Một grocery plan nói tôi định mua rau; nó không
chứng minh tôi đã mua. Một decision nói lý do chọn TTS; nó không phải benchmark
đo latency. Tôi cần biểu diễn các khác biệt này bằng folder và metadata, không bắt
LLM đoán từ văn phong.

## Cấu trúc tôi chốt

| Khu vực | Câu hỏi nó trả lời tốt | Không được dùng thay cho |
| --- | --- | --- |
| `learning/` | Tôi hiểu khái niệm này thế nào? | bằng chứng tôi đã làm việc đó |
| `life/decisions/` | Tôi đã chọn gì, vì sao, khi nào xem lại? | log thực thi đầy đủ |
| `life/journal/` | Ngày đó đã xảy ra chuyện gì? | định nghĩa kỹ thuật chuẩn |
| `life/finance/` | Khoản nào planned, khoản nào actual? | suy đoán thu nhập |
| `life/health/` | Tôi đang quan sát gì, cần hỏi gì? | chẩn đoán hoặc toa điều trị |
| `sources/` | Corpus có quy ước và nguồn nào? | sự thật về đời sống cá nhân |

## Quy tắc evidence

Mỗi note phải có `type`, `status`, ngày và `source_kind`. Với những note có số
liệu, tôi thêm trạng thái của số đó: `actual`, `planned`, `estimate` hoặc
`unknown`. Tôi không muốn câu “tháng này tôi đã tiêu X” được tổng hợp từ một
plan chỉ vì plan có con số X.

Ngày tạo và ngày cập nhật cũng không giống ngày sự kiện. Journal giữ ngày sự kiện;
`updated` chỉ nói lần cuối tôi sửa note. Khi sửa một quyết định, tôi thêm change
log với ngày và lý do thay đổi, không âm thầm viết lại lịch sử.

## Cách tôi viết một note mới

1. Ghi raw thought trước, chưa cố làm câu chữ đẹp.
2. Chọn loại evidence dựa trên câu hỏi note sẽ trả lời.
3. Thêm ví dụ cụ thể hoặc record có thể kiểm tra.
4. Ghi rõ điều chưa biết và việc tiếp theo.
5. Link sang note liên quan nhưng không làm mất loại của note hiện tại.
6. Đọc lại bằng query paraphrase, không chỉ search đúng title.

Tôi tránh tách một chủ đề thành nhiều file vài dòng. Nếu câu hỏi cần bối cảnh,
ví dụ và failure case, chúng ở cùng một file để chunk vẫn giữ được ý. Ngược lại,
ledger dài sẽ tách khỏi budget vì mỗi dòng giao dịch cần cập nhật độc lập.

## Điều tôi kiểm tra ở assistant

- “Tôi đã học Bayes khi nào?” phải ưu tiên journal.
- “Bayes hoạt động thế nào?” phải ưu tiên learning.
- “Tôi đã mua gì?” phải lấy receipt ledger, không lấy grocery plan.
- “Tôi định mua gì tuần cuối tháng?” phải giữ nhãn planned.
- “Tôi đã quyết chọn TTS nào?” phải lấy decision và ngày hiệu lực.
- “Tôi có bệnh gì không?” phải dừng ở safety boundary, không suy luận từ journal.
- Query không có note phải trả `insufficient`, không chọn một file gần nghĩa.

## Chi phí và điều tôi chấp nhận

Cấu trúc này làm tôi phải viết metadata và cross-link nhiều hơn. Đổi lại, khi
retrieval sai, tôi biết sai ở loại evidence nào. Tôi chấp nhận vài giây dọn note
thay vì có một vault ngắn nhưng mọi câu trả lời đều trộn lẫn planned và actual.

Tôi cũng chấp nhận có note chưa hoàn chỉnh. `draft` hoặc `unknown` trung thực hơn
một đoạn văn trôi chảy nhưng không có record. Assistant phải giữ trạng thái đó khi
tóm tắt, không biến “tôi sẽ đo lại” thành “đã đo đạt”.

## Lịch sử quyết định

| Ngày | Thay đổi | Lý do |
| --- | --- | --- |
| 20/07 | tách journal khỏi learning | câu hỏi “đã học chưa” cần ngày sự kiện |
| 23/07 | tách finance plan khỏi ledger | tránh nhầm dự tính với khoản đã chi |
| 27/07 | bỏ folder project khỏi showcase | không muốn checklist sản phẩm lấn vào life |
| 29/07 | thêm confidence và review date | quyết định hiện tại không phải chân lý vĩnh viễn |

## Khi nào tôi xem lại

Tôi sẽ xem lại sau khi dùng vault đủ một tháng hoặc khi retrieval vẫn thường xuyên
trộn journal với learning. Nếu vấn đề là model không đọc metadata, tôi sửa
pipeline/context trước; không vội gom folder lại cho dễ index.

## Kết luận hiện tại

Tách folder là một quyết định về semantics, không phải trang trí cây thư mục.
Tôi muốn assistant nói được không chỉ “nội dung gì”, mà còn “đây là loại bằng
chứng nào, xảy ra lúc nào, còn chắc đến đâu”.
