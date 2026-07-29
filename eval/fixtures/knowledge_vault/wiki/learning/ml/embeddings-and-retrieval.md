---
type: learning_note
domain: ml
topic: embeddings-and-information-retrieval
status: active
created: 2026-07-20
updated: 2026-07-28
tags: [ml, embeddings, retrieval, vector, sparse, dense, rag]
source_kind: personal-study-note
---

# ML: embedding và retrieval — từ “gần nhau” đến “đúng câu hỏi”

## Tôi từng nghĩ embedding là gì

Tôi từng tưởng embedding biến câu thành một tọa độ để câu giống nghĩa tự động
nằm cạnh nhau. Thực tế vector chỉ là biểu diễn học được dưới một objective. “Gần”
phụ thuộc model, pooling, normalization, domain và cách tạo query/passage.

Embedding không tự biết đoạn nào là bằng chứng. Nó chỉ cho tôi một tín hiệu để
chọn candidate. Câu hỏi sau đó vẫn cần relevance gate và kiểm tra groundedness.

## Cosine và hình học

Với vector đã normalize, dot product gần với cosine similarity. Tôi hình dung mỗi
văn bản là một mũi tên; hướng thể hiện pattern ngữ nghĩa, độ dài bị loại bỏ khi
normalize. Hai câu có từ khác nhau nhưng cùng pattern có thể gần nhau; hai câu
có một từ chung nhưng ngữ cảnh khác có thể vẫn xa.

Tôi không so raw score giữa hai model. Cosine của model A và score reranker của
model B không cùng thang đo. Threshold phải gắn với backend/model fingerprint.

## Sparse và dense

Sparse giữ tín hiệu token, phrase, title và IDF. Nó mạnh với tên riêng, mã lỗi,
file path và cụm từ hiếm. Dense mạnh với paraphrase nhưng dễ kéo note “cùng chủ
đề” mà không có câu trả lời cụ thể.

Tôi dùng hai người tìm:

- sparse là người nhớ đúng từ khóa;
- dense là người hiểu cách diễn đạt gần nghĩa.

Hai danh sách chạy độc lập. Dense không được đặt sau lexical pre-gate, nếu không
paraphrase miss ngay từ cửa vào.

## Chunking

Chunk quá dài làm bằng chứng lẫn nhiều ý và tốn context. Chunk quá ngắn mất ngữ
cảnh, title hoặc điều kiện. Tôi ưu tiên cắt theo heading/đoạn, giữ line range,
path và title trong record.

Một chunk ID nên phụ thuộc path, line range và text. Mtime chỉ là tín hiệu thay
đổi; content hash mới quyết định có cần embed lại hay không.

## Pipeline tôi tự vẽ

```text
query
  → normalize nhưng giữ raw query
  → sparse candidates
  → dense candidates
  → backend-local score handling
  → fusion / optional rerank
  → floor + margin
  → dedup/diversity
  → evidence bundle
  → LLM context
```

“Top-k” chỉ là giới hạn candidate. Sau floor có thể còn zero hit và đó là kết
quả hợp lệ.

## RRF tôi hiểu thế nào

Reciprocal Rank Fusion cộng đóng góp theo rank để kết hợp nhiều retriever mà
không so raw score. Chunk xuất hiện top ở nhiều backend được ưu tiên. Nhưng RRF
không biết chunk có trả lời câu hỏi hay không; nó chỉ ổn định việc hợp nhất.

Nếu sparse top1 là một hard negative và dense cũng bị hút theo, fusion có thể
đánh giá sai rất tự tin. Vì thế evidence gate cần xem coverage, dense floor,
margin và query class.

## Relevance và abstention

Tôi tách bốn trạng thái:

- supported: tín hiệu đủ và top có separation;
- weak: có candidate nhưng margin thấp hoặc calibration chưa chắc;
- insufficient: đã tra nhưng không có evidence đủ liên quan;
- unavailable: index/model không sẵn sàng.

Không dùng “hit_count > 0” làm supported. Một note có chữ `provider` không trả lời
được câu hỏi provider của ONNX nếu snippet không có mapping đó.

## Đánh giá retrieval

Recall@k trả lời “evidence đúng có lọt vào top-k không?”. MRR quan tâm vị trí
đầu tiên. nDCG xử lý nhiều mức relevance. Precision@k giúp đo contamination.

Với RAG thực tế, tôi còn đo:

- distractor contamination;
- unsupported abstention;
- citation precision/recall;
- answer faithfulness;
- p50/p95 latency;
- index build và incremental update;
- peak RSS và disk size.

Một model tăng Recall nhưng làm contamination tăng gấp đôi chưa chắc là nâng cấp.

## Dense model chưa provision

Nếu embedding model không có local, sparse fallback phải báo degraded. Không tạo
vector zero, không coi cache cũ của model khác là tương thích. Manifest cần lưu
model ID, dimension, prefix, pooling và source digest.

## Query tiếng Việt

Normalize dấu giúp match lexical nhưng không nên thay raw query. Từ ghép, tên
riêng và lỗi ASR cần được kiểm bằng held-out set. Tôi không giải quyết mọi query
bằng regex; mỗi rule phải có lý do, test và artifact calibrate.

## Hard negative tôi tự tạo

- query Bayes và nhật ký có nhắc Bayes nhưng không giải thích;
- query ONNX provider và note context có chữ provider;
- query ngân sách tháng 07 và review chỉ nói cách theo dõi;
- query TTS local và privacy decision chỉ nói ranh giới remote;
- query thời tiết khi vault không có realtime source.

Hard negative tốt không phải đoạn vô nghĩa. Nó là đoạn đọc lên thấy hợp chủ đề
nhưng vẫn không đủ để trả lời câu hỏi.

## Cách tôi debug một hit lạ

1. xem query token sau normalize;
2. xem source backend và score local;
3. đọc title, path và snippet, không chỉ title;
4. kiểm tra phrase/coverage nào làm nó lọt;
5. so top1/top2 margin;
6. xác định lỗi retriever, gate hay prompt;
7. thêm case vào held-out hard-negative set nếu lỗi có tính lặp.

## Câu hỏi mở

- khi nào RRF thua reranker multilingual;
- cách đo semantic similarity cho tiếng Việt có code-mix;
- chunk theo section hay theo token tốt hơn với note cá nhân dài;
- ANN có đáng cho corpus laptop hay exact search đủ rồi;
- có nên lưu query embedding trong trace hay chỉ lưu fingerprint để riêng tư.

## Tóm tắt

Embedding giúp tôi tìm “gần nghĩa”, không đảm bảo tìm “đúng điều cần chứng minh”.
RAG tốt là retrieval, relevance, evidence, prompt và verification đi cùng nhau;
đổi model embedding riêng lẻ không giải quyết được pipeline sai.

## Query và passage không phải cùng một loại text

Một query thường ngắn, nhiều ý định và có từ “của tôi”, “tuần trước”, “tại sao”.
Passage là đoạn note có tiêu đề, ngày, ví dụ và caveat. Nếu model được train với
prefix khác nhau, tôi phải dùng đúng instruction/prefix cho query và passage. Tôi
không kết luận model kém chỉ từ một lần encode sai format.

Chunking cũng là một phần của embedding. Cắt giữa tiêu đề và điều kiện áp dụng
có thể làm vector mất nghĩa. Chunk quá dài trộn nhiều topic; quá ngắn mất chủ ngữ
và tạo citation rời rạc. Tôi muốn chunk giữ heading, source path, line range và
đủ context để người đọc kiểm chứng.

## Vì sao lexical và dense cần hỗ trợ nhau

Lexical giỏi tên riêng, mã lỗi, số, path và exact phrase. Dense giỏi paraphrase,
typo nhẹ và câu hỏi diễn đạt khác note. Hybrid không có nghĩa lấy union mọi hit;
nó cần fuse score, áp threshold và kiểm margin/relevance. Một dense hit nghe gần
nghĩa nhưng khác domain vẫn là false positive.

Ví dụ “ngân sách tuần cuối” có thể kéo budget, grocery plan và ledger. Câu trả lời
phải phân biệt planned/actual, không chọn top-1 chỉ vì score. Tôi muốn metadata
retrieval backend, sparse/dense/fusion score và reason được giữ trong trace.

## Score không phải xác suất đúng

Cosine 0,8 không có nghĩa 80% câu trả lời đúng. Score phụ thuộc model, normalize,
domain và corpus. Tôi calibrate bằng query có nhãn: relevant, adjacent, irrelevant,
no-answer. Threshold cần theo dõi recall và false positive, đặc biệt với query
out-of-scope.

Top-1 score cao nhưng margin với top-2 thấp là tín hiệu ambiguity. Tôi có thể đưa
hai evidence cho LLM với instruction so sánh, hoặc hỏi clarification. Không nên
giấu ambiguity bằng cách tự chọn một file.

## Chẩn đoán một lượt retrieval sai

1. query đã normalize đúng unicode/typo chưa;
2. chunk có chứa answer hay chỉ chứa keyword;
3. lexical miss nhưng dense có rescue được không;
4. dense score có cao vì boilerplate lặp không;
5. fusion/rerank có làm mất hit tốt không;
6. evidence gate có abstain khi tất cả yếu không;
7. LLM có trích ngoài context dù retrieval đúng không.

Tôi log fingerprint và score, không log raw private text mặc định. Khi debug
corpus, tôi có thể xem path/chunk cụ thể; khi chạy vault thật, retention cần theo
privacy policy.

## Incremental indexing

Document digest và chunk id giúp reuse vector không đổi. Khi sửa heading ở đầu
file, boundary có thể đổi nhiều chunk; tôi ghi số chunk reused/new/deleted để biết
chi phí thực. Khi model/dimension/normalization đổi, cache cũ không được dùng mù.

Dense matrix vài chục chunk có thể exact search. Tôi chỉ cân nhắc FAISS/HNSW khi
benchmark cho thấy latency hoặc memory cần, và vẫn phải có manifest model/schema,
rebuild, atomic generation và garbage collection.

## Bài tập tôi muốn chạy

- typo tiếng Việt và query code-mix;
- query answerable, adjacent và no-answer cùng keyword;
- chunk title-only so với chunk có context;
- thay model embedding nhưng giữ query set;
- thêm/sửa/xóa document rồi kiểm stale hit;
- đo recall@k, MRR, no-answer precision và citation support;
- kiểm hybrid có thật sự cứu lexical miss mà không tăng false positive.
