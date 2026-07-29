---
type: learning_note
domain: fundamentals
topic: probability-and-statistics
status: active
created: 2026-07-18
updated: 2026-07-28
tags: [probability, bayes, statistics, reasoning]
source_kind: personal-study-note
---

# Xác suất và định lý Bayes — cách tôi đổi góc nhìn

## Tôi từng nghĩ gì

Trước đây tôi hay nhìn xác suất như một con số cố định gắn với sự kiện. Nếu
nghe “xét nghiệm có độ chính xác 95%”, tôi gần như đọc nó thành “kết quả này có
95% là đúng”. Cách đọc đó bỏ qua quần thể ban đầu và loại lỗi của phép đo.

Sau này tôi hiểu xác suất thường là mức độ tin tưởng của tôi trước một thông tin
chưa chắc chắn. Khi có dữ liệu mới, mức độ tin tưởng phải được cập nhật, nhưng
không được nhảy thẳng đến kết luận chỉ vì một tín hiệu nổi bật.

## À, hóa ra Bayes là một cách cập nhật niềm tin

Tôi ghi công thức như sau:

`P(A|B) = P(B|A) × P(A) / P(B)`

Tôi đọc từng phần bằng câu hỏi tự nhiên:

| Phần | Tôi tự hỏi |
| --- | --- |
| `A` | giả thuyết tôi đang quan tâm là gì? |
| `B` | bằng chứng mới quan sát được là gì? |
| `P(A)` | trước bằng chứng, A có phổ biến không? |
| `P(B|A)` | nếu A đúng thì B có dễ xảy ra không? |
| `P(B)` | B có thể xảy ra trong mọi trường hợp với tần suất nào? |
| `P(A|B)` | sau khi thấy B, tôi nên tin A đến mức nào? |

Điểm mấu chốt với tôi là `P(A|B)` và `P(B|A)` là hai câu hỏi khác nhau. Một
loại thuốc có hiệu quả khi bệnh nhân đúng bệnh không có nghĩa một người uống
thuốc rồi chắc chắn mắc bệnh. Một hệ thống bắt được tín hiệu đúng cũng không có
nghĩa mọi tín hiệu nó bắt đều là tín hiệu thật.

## Ví dụ 10.000 người

Tôi dùng bảng đếm người vì nó dễ nhìn hơn một công thức dài. Giả sử:

- tỷ lệ nền của bệnh là 1%;
- xét nghiệm bắt được 90% người bệnh;
- xét nghiệm dương tính giả ở 5% người không bệnh.

Trong 10.000 người:

1. khoảng 100 người thật sự mắc bệnh;
2. trong 100 người đó, khoảng 90 người dương tính thật;
3. khoảng 9.900 người không mắc bệnh;
4. trong 9.900 người đó, khoảng 495 người dương tính giả;
5. tổng dương tính là khoảng 585 người;
6. trong 585 người dương tính, chỉ khoảng 90 người bệnh;
7. xác suất hậu nghiệm là `90 / 585`, xấp xỉ 15,4%.

Tôi không dùng con số 15,4% để kết luận cho một cá nhân. Nó chỉ cho thấy base
rate nhỏ làm số dương tính giả lớn hơn trực giác ban đầu.

## Cách tôi giải một bài Bayes

### Bước 1 — viết câu hỏi bằng tiếng người

Tôi không bắt đầu bằng `P(...)`. Tôi viết “tôi muốn biết khả năng A sau khi thấy
B” để không đảo chiều điều kiện.

### Bước 2 — vẽ bảng trường hợp

Tôi tách true positive, false positive, true negative và false negative. Nếu
không điền được bốn ô này thì tôi chưa hiểu dữ liệu.

### Bước 3 — kiểm tra mẫu số

Mẫu số của “xác suất mắc bệnh khi dương tính” là toàn bộ người dương tính, gồm
cả dương tính thật và dương tính giả. Đây là nơi tôi thường quên nhất.

### Bước 4 — nói rõ giả định

Các phép tính ngắn thường ngầm giả định mẫu đại diện, các phép đo độc lập và
định nghĩa “dương tính” không thay đổi giữa nhóm. Nếu giả định không đúng, con
số đẹp vẫn có thể dẫn đến kết luận sai.

## Các dạng xác suất tôi hay nhầm

- xác suất trước và sau khi nhìn thấy bằng chứng;
- xác suất có điều kiện và xác suất đồng thời;
- xác suất của một cá nhân và tần suất trong quần thể;
- độ chính xác của model và xác suất prediction đúng;
- correlation và khả năng một biến gây ra biến khác;
- “chưa có bằng chứng” và “bằng chứng không tồn tại”.

## Bayes trong machine learning

Tôi nhìn classifier như một hệ thống tạo evidence, không phải oracle. Prior có
thể nằm trong dữ liệu huấn luyện; likelihood phụ thuộc cách model phản ứng với
feature; posterior là mức tin tưởng sau khi kết hợp cả hai.

Calibration quan trọng vì hai model có cùng accuracy nhưng một model có xác suất
đầu ra đáng tin hơn. Nếu model nói 0,8, tôi muốn nhóm các dự đoán 0,8 đúng gần
80% trong dài hạn, không chỉ muốn score cao.

## Cách tôi giải thích cho người khác

Tôi ví prior như ý kiến ban đầu, evidence như một người bạn đưa thêm thông tin,
posterior như ý kiến sau khi nghe bạn. Nếu người bạn thường cung cấp tín hiệu
nhiễu, chỉ nghe một câu chưa đủ để đổi ý hoàn toàn.

Tôi cũng ví retrieval trong RAG như việc lấy sách từ thư viện. Sách được lấy ra
chưa phải câu trả lời; nó chỉ là bằng chứng tiềm năng. Cần đọc đúng đoạn và kiểm
tra đoạn đó có thực sự trả lời câu hỏi hay không.

## Một ví dụ không phải y tế

Tôi thấy một log có chữ “timeout”. Giả thuyết A là provider chậm, nhưng cũng có
thể do DNS, queue, model warm-up hoặc client đặt deadline quá ngắn. Nếu trước đó
provider ổn định, prior cho “provider chậm” chưa chắc cao. Tôi cần thêm latency,
retry count và stage trace trước khi cập nhật kết luận.

## Checklist tự kiểm tra

1. Tôi đang hỏi `A|B` hay `B|A`?
2. Bằng chứng này có base rate hoặc mẫu số nào?
3. Có loại false positive/false negative nào không?
4. Tôi đang suy luận cho quần thể hay một cá nhân?
5. Có giả định độc lập hoặc đại diện nào bị giấu không?
6. Kết quả là xác suất, score hay decision threshold?
7. Nếu bỏ một evidence, kết luận còn giữ được không?

## Câu hỏi tôi còn để mở

- calibration curve và Brier score liên hệ với Bayes như thế nào trong sản phẩm;
- khi nhiều evidence phụ thuộc nhau, tôi nên mô hình hóa dependency ra sao;
- làm sao nói uncertainty rõ mà không làm người nghe tưởng hệ thống không có ích;
- trong RAG, evidence status nên chuyển thành xác suất hay typed decision.

## Ghi chú nguồn

Đây là note tự diễn giải để học, với ví dụ số minh họa. Các ví dụ sức khỏe không
phải tư vấn y tế; khi hỏi về triệu chứng cần xem ranh giới ở
`wiki/life/health/safety-boundaries.md`.

## Worked example: bảng đếm 10.000 người

Giả sử tỷ lệ nền của một tình trạng là 1%. Trong 10.000 người có 100 người thật
sự có tình trạng đó. Nếu test có sensitivity 90%, khoảng 90 người bệnh dương
tính. Nếu specificity 95%, 5% của 9.900 người không bệnh vẫn dương tính, tức
495 false positive. Tổng positive là 585, nên xác suất thật sự có tình trạng khi
đã positive là 90/585, xấp xỉ 15,4%, không phải 90% hay 95%.

Con số này không phải khuyến nghị y tế. Tôi dùng nó vì bảng đếm làm rõ mẫu số:
“test positive” là điều kiện quan sát, còn “có tình trạng” là hypothesis. Nếu tỷ
lệ nền thay đổi, posterior đổi dù test giữ nguyên sensitivity/specificity.

## Ba câu tôi tự hỏi khi đọc một claim

1. Claim đang nói `P(evidence | hypothesis)` hay `P(hypothesis | evidence)`?
2. Prior đến từ population nào, có phù hợp với nhóm đang xét không?
3. False positive/negative và cách lấy mẫu có làm lệch interpretation không?

Nếu người nói không nêu population, tôi đánh dấu uncertainty thay vì điền prior
bằng cảm giác. Nếu data được chọn sau khi nhìn outcome, posterior có thể trông
đẹp một cách giả tạo.

## Base rate và product decision

Trong classifier của assistant, một intent hiếm nhưng rủi ro cao không nên chỉ
được quyết bằng accuracy. Tôi cần confusion matrix và chi phí false positive/
false negative. Với knowledge gate, false positive có thể làm LLM trả lời lạc đề;
false negative làm mất một lượt retrieval. Threshold hợp lý tùy hậu quả.

Điều này nối với note evaluation: threshold phải được calibrate trên tập giữ lại,
không chỉnh đến khi showcase “trông đúng”. Tập test cần có query không thuộc
knowledge, typo, code-mix và các query gần nghĩa nhưng khác domain.

## Một phản ví dụ tôi hay dùng

Nếu tôi hỏi “Bayes có phải cứ có test đúng 95% là posterior 95% không?”, câu trả
lời không nên chỉ lặp công thức. Cần nói precision phụ thuộc prevalence. Nếu tôi
hỏi “vault của tôi ghi gì về Bayes?”, lúc đó phải ưu tiên evidence từ note và
không bịa một ví dụ ngoài note rồi gắn citation.

## Cách tôi ghi uncertainty

Tôi phân biệt ba câu: “note nói X”, “tôi suy ra Y từ X” và “tôi chưa có dữ liệu
cho Z”. Ba câu có mức evidence khác nhau. Assistant cũng nên giữ distinction:
citation chứng minh note chứa câu nào, không tự động chứng minh mọi diễn giải
mới của model.

## Bài tập tự kiểm tra

- tính posterior khi prior là 10% với cùng sensitivity/specificity;
- đổi specificity và xem false positive thay đổi ra sao;
- viết một ví dụ Simpson’s paradox bằng bảng đếm nhỏ;
- kiểm tra calibration curve thay vì chỉ accuracy;
- tạo query Bayes có typo và query “Bayes trong note của tôi”;
- viết một câu trả lời abstain khi không có note Bayes trong vault.

## Điều tôi đã đổi trong cách học

Trước đây tôi cố nhớ công thức trước. Bây giờ tôi dựng bảng đếm, đặt tên các
biến cố, viết direction của conditional probability rồi mới rút gọn đại số. Cách
này chậm hơn ở ví dụ đầu nhưng giúp tôi phát hiện lúc đang đảo điều kiện.
