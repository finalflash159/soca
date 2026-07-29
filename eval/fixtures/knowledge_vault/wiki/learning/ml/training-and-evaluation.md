---
type: learning_note
domain: ml
topic: training-and-evaluation
status: active
created: 2026-07-21
updated: 2026-07-28
tags: [ml, training, validation, leakage, metrics, calibration]
source_kind: personal-study-note
---

# ML: train, validation và evaluation — đừng tự chấm bài mình đã học thuộc

## Cú nhầm đầu tiên

Tôi từng thấy metric tăng rồi nghĩ model tốt hơn. Sau đó tôi hỏi: metric trên bộ
nào, split thế nào, có leakage không, variance bao nhiêu và failure class nào bị
che đi? Một con số đẹp trên dữ liệu đã nhìn thấy không đại diện cho generalization.

Tôi ví train set là bài tập đã làm, validation là đề luyện chưa chấm cuối cùng,
test là đề thi chỉ mở một lần. Nếu cứ nhìn test trong lúc chỉnh model, test không
còn là test sạch.

## Ba tập dữ liệu

Train dùng để cập nhật weight. Validation dùng để chọn hyperparameter, threshold,
checkpoint và decision. Test chỉ dùng khi quyết định cuối đã đóng.

Tôi còn tách theo người dùng, thời gian, topic hoặc source nếu production sẽ gặp
distribution khác. Random split có thể làm một user xuất hiện ở cả train và test,
khiến metric giả cao.

## Leakage

Leakage không chỉ là copy exact answer. Nó có thể là:

- paraphrase cùng một trajectory nằm ở hai split;
- future feature lọt vào record quá khứ;
- normalization dùng thống kê của toàn bộ corpus;
- duplicate document khác tên file;
- benchmark query được viết từ chính note train;
- threshold tune trực tiếp trên release test.

Tôi lưu provenance, split và hash của case. Dataset demo không được trộn vào
quality benchmark chỉ vì nó dễ pass.

## Classification metric

Accuracy dễ hiểu nhưng giấu class imbalance. Precision trả lời “dự đoán positive
đúng bao nhiêu?”, recall trả lời “bắt được bao nhiêu positive thật?”. F1 là cân
bằng harmonic, không phải lúc nào cũng là mục tiêu đúng.

Trong router, false positive tool có thể nguy hiểm hơn false negative vì gọi sai
capability. Trong retrieval, bỏ qua evidence đúng và nhận distractor là hai lỗi
khác nhau; không gộp thành một accuracy.

## Threshold và calibration

Score model thường không phải xác suất. Threshold 0,8 ở model này không tương
đương 0,8 ở model khác. Tôi chọn threshold trên dev, kiểm tra confidence bucket,
then evaluate một lần trên held-out test.

Calibration tốt nghĩa nhóm prediction 0,7 có outcome đúng gần 70% trong dài hạn.
Reliability diagram, Expected Calibration Error và Brier score giúp phát hiện
score tự tin nhưng sai.

## Retrieval evaluation

Với retrieval, tôi không chỉ hỏi “top1 đúng không?”. Tôi lưu relevance judgment
theo query và document/chunk:

| Metric | Tôi đọc là |
| --- | --- |
| Recall@5 | evidence đúng có xuất hiện trong 5 hit? |
| MRR@10 | hit đúng đầu tiên đứng ở đâu? |
| nDCG@10 | nhiều mức relevance được xếp ra sao? |
| Precision@3 | top3 có bao nhiêu distractor? |
| abstention | query không đáp án có dừng đúng không? |

## Model grader và human sample

Model grader rẻ hơn human nhưng có bias và variance. Tôi dùng code grader cho
route/tool/citation/budget, model grader cho faithfulness/relevance và human
sample để calibrate grader. Không cho model tự chấm bài của chính nó là bằng chứng
duy nhất.

## Ablation

Muốn biết dense có ích, tôi so current lexical, dense, hybrid và reranker trên
cùng query/corpus/hardware. Mỗi trial ghi commit, model, seed nếu có, cache state,
latency và RSS.

Một ablation tốt chỉ thay một biến. Nếu vừa đổi chunker, embedder, threshold và
prompt rồi metric tăng, tôi không biết phần nào tạo ra cải thiện.

## Variance và confidence interval

LLM answer có generation variance; một lượt pass không đủ. Tôi chạy nhiều trial
cho case quan trọng, báo mean/median và khoảng biến thiên. Với retrieval exact,
variance thấp hơn nhưng index generation và model download vẫn phải ghi.

## Đánh giá lỗi

Tôi lưu failure taxonomy thay vì chỉ log pass/fail:

- no-hit nhưng đáng lẽ có evidence;
- irrelevant-hit accepted;
- evidence đúng nhưng answer unsupported;
- citation thiếu hoặc trỏ ID không tồn tại;
- answer abstain quá nhiều;
- tool đúng nhưng terminal goal sai;
- latency/RSS vượt resource gate.

## Checklist trước khi tin metric

1. dataset có nguồn và license rõ không?
2. split có leakage/paraphrase family không?
3. benchmark có vô tình dùng demo không?
4. query không đáp án có đủ hard negative không?
5. threshold được tune trên dev hay test?
6. metric có phản ánh risk product không?
7. có transcript và artifact để reproduce không?

## Tóm tắt

Evaluation không phải bước trang trí sau code. Nó là cách tôi biết model tốt hơn ở
đâu, tệ hơn ở đâu và có đang học thuộc bộ demo hay không. Không có split,
provenance và failure taxonomy thì metric chỉ là cảm giác được viết thành số.

## Split không chỉ là chia ngẫu nhiên

Random split có thể làm cùng một template, người nói hoặc tài liệu xuất hiện ở cả
train và test. Khi đó metric cao nhưng model chưa gặp distribution mới. Tôi cân
nhắc split theo user, time, document, domain và difficulty. Với memory/knowledge,
split theo source document quan trọng để tránh chunk leakage.

Nếu dữ liệu có temporal drift, test gần thời điểm deploy mới nói được performance
thực tế. Nếu query tiếng Việt có typo/code-mix, các nhóm đó cần xuất hiện đủ ở
eval; không để happy path chiếm 95% rồi gọi average representative.

## Metric và decision

Retrieval có recall@k, precision@k, MRR/nDCG tùy mục tiêu. Answer có groundedness,
citation correctness, abstention quality và latency. Một hệ có recall cao nhưng
đưa nhiều distractor vào prompt có thể làm answer giảm.

Tôi ghi metric theo slice, không chỉ micro average. No-answer precision quan trọng
với safety: hệ từ chối đúng câu không có evidence có thể tốt hơn hệ trả lời đầy
đủ nhưng lạc đề. Threshold thay đổi cần đánh đổi được ghi trong decision note.

## Leakage và contamination

Leakage có thể đến từ duplicate, near-duplicate, answer nằm trong metadata,
prompt template chứa label hoặc benchmark đã nằm trong training data. Tôi hash/
normalize để tìm duplicate, kiểm source date và giữ provenance. Nếu không biết
dataset có contamination, ghi unknown thay vì nói eval sạch.

## Calibration và confidence

Confidence của classifier/threshold không tự nhiên là xác suất. Reliability curve,
ECE hoặc binning giúp biết score 0,8 có consistent không. Với retrieval, tôi có
thể calibrate score + margin trên labeled set; với LLM answer, citation validator
chỉ nói claim có label hợp lệ, chưa chứng minh entailment hoàn toàn.

## Failure taxonomy tôi dùng

- miss: evidence đúng có trong corpus nhưng không retrieve;
- false positive: hit liên quan bề mặt nhưng không support;
- stale: index/manifest không phản ánh file mới;
- synthesis: LLM bỏ qua hoặc diễn giải quá context;
- abstain sai: có evidence đủ nhưng hệ từ chối;
- citation lỗi: label thiếu, sai hoặc trích nguồn không support;
- ops: timeout, empty completion, resource hoặc provider failure.

Mỗi failure cần query, corpus revision, model/config, trace và expected behavior.
Không chỉ ghi “model trả lời sai” vì remediation của retrieval và synthesis khác
nhau.

## Release gate thực tế

Trước khi gọi model tốt hơn, tôi chạy regression trên bộ cố định, blind/adversarial
set và real-flow smoke. Tôi ghi mean/p95 latency, token, memory, cold/warm, error
rate và output empty. Nếu benchmark pass nhưng cold process không tải được model,
release gate vẫn fail.

## Bài tập

- viết 20 query no-answer không lộ keyword “không có”;
- tạo near-duplicate khác heading;
- giữ một test set không được dùng khi calibrate threshold;
- chạy evaluator leak sau mỗi lần sửa corpus;
- review từng failure theo taxonomy;
- lưu decision cùng model revision và command reproduce.
