# Nhật ký dự án — Hybrid RAG và memory

#life-vault #project #rag #memory #architecture

Ngày ghi: 2026-07-22
Slice: life_vault
Provenance: repository_fact — tóm tắt từ `docs/09-hybrid-rag-memory.md`.

## Luồng hiện tại

1. Markdown vault được lập index theo document/chunk.
2. Sparse retriever tìm theo lexical/BM25.
3. Dense retriever dùng embedding local khi backend khả dụng.
4. Reciprocal Rank Fusion (RRF) hợp nhất hai danh sách xếp hạng.
5. Context builder đóng gói các hit và line citation cho LLM.
6. Guardrail kiểm tra tool, đường dẫn và citation trước khi trả lời.

Memory là client của cùng retriever nhưng khác namespace với knowledge. Profile
được truy hồi theo query với relevance, recency và importance; working memory
được nén ngoài hot path. Episodic memory chỉ được ghi khi có consent và qua
proposal/approval.

## Điều không được làm

- Không biến một keyword thành bằng chứng để tự động truy hồi mọi câu hỏi.
- Không cho LLM đọc path ngoài phạm vi vault.
- Không trả lời như fact nếu không có hit/citation phù hợp.
