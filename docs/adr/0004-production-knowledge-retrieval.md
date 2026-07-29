# ADR 0004: Production knowledge retrieval

## Quyết định

SoCa dùng đúng một profile retrieval production:

- sparse: BM25 Lucene;
- dense: `AITeamVN/Vietnamese_Embedding_v2`, revision
  `18b44161e041bf1d3a333ab5144b5b7b93f914d2`, 1024 chiều;
- fusion: min-max linear score fusion, dense weight `0.75`;
- vector search: exact deterministic NumPy top-k;
- reranker: không dùng.

Model, tokenizer và dense generation đều được khóa bằng SHA-256. Dense
generation phải khớp corpus revision, source digest và toàn bộ embedding
fingerprint.

## Bằng chứng

TVPL 1.000 query cho winner Recall@5 `0.9161`, MRR@10 `0.8275` và nDCG@10
`0.8487`. BM25 chỉ đạt `0.7001 / 0.5970 / 0.6278`; hybrid BGE-M3 đạt
`0.8953 / 0.8020 / 0.8244`.

Ba seed 250.000 vector thật 1024 chiều xác nhận NumPy và FAISS Flat có Recall@10
và ordered-top-k match `1.0`. FAISS chỉ giảm p95 khoảng 2,6–2,9 ms, không đạt
gate 2× và không đáng thêm native dependency/lifecycle. Reranker có latency
p95 ở mức giây và không cải thiện ổn định trên ViRe.

Chi tiết dataset, resource, run ID và lệnh tái lập nằm trong
`BENCHMARKS.md` và `docs/10-vietnamese-rag-model-selection.md`.

## Failure contract

Production không tự đổi sang sparse, model khác, fusion khác, stale generation
hoặc vector backend khác. Thiếu dependency/model, checksum/dimension sai,
generation absent/stale/failed/corrupt phải hiện lỗi. Empty corpus hoặc zero hit
là kết quả hợp lệ riêng biệt.

Rollback chỉ do operator gọi và chỉ tới previous generation có cùng corpus
revision/source digest. Candidate thua chỉ tồn tại trong eval/research.

## Hệ quả

SoCa chấp nhận khoảng 70 ms p95 query embedding và model khoảng 2,1 GB để đổi
lấy chất lượng tiếng Việt cao hơn. Thay model, fusion weight, reranker hoặc
vector backend bắt buộc chạy lại release benchmark và cập nhật ADR.
