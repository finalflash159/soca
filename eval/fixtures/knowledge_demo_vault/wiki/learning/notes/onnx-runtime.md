# Ghi chú học tập — ONNX Runtime trong SoCa

#learning #notes #onnx #onnx-runtime #machine-learning

Ngày ghi: 2026-07-21
Slice: learning_notes
Provenance: repository_fact — tóm tắt từ `BENCHMARKS.md` và `docs/09-hybrid-rag-memory.md`.

## Điều cần nhớ

ONNX Runtime là lớp chạy model ONNX trong pipeline local của SoCa. Trên máy
Apple, benchmark của dự án kiểm tra `CoreMLExecutionProvider` và
`CPUExecutionProvider`; CPU là fallback khi một node không chạy được trên
CoreML.

Trong voice pipeline, PhoWhisper dùng các graph ONNX encoder/decoder cho ASR.
Ở knowledge layer, dense retriever cũng có thể dùng embedding model ONNX;
retrieval sẽ quay về sparse-only nếu dense backend không khả dụng.

## Liên hệ với RAG

ONNX Runtime không tự quyết định note nào được đưa vào câu trả lời. Nó chỉ
chạy phần embedding/dense retrieval. Sau đó SoCa hợp nhất sparse và dense bằng
RRF, rồi mới tạo context có citation cho LLM.

## Nguồn trong repo

- `BENCHMARKS.md`: provider CoreML/CPU và benchmark ONNX của ASR.
- `docs/09-hybrid-rag-memory.md`: dense retriever, fallback và RRF.
