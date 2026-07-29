# Kiến trúc RAG và memory của SoCa

Knowledge và memory dùng chung abstraction retrieval nhưng khác namespace và
guardrail. Knowledge có path `wiki/`; memory archive có path riêng và citation
`[M#]`. Working memory của session không phải archive memory và không cần gọi
retriever ở mỗi câu.

Một câu hỏi knowledge đi qua tool/context/LLM khi LLM khả dụng. Context có thể
rỗng với trạng thái `insufficient`; khi đó prompt yêu cầu abstain thay vì dùng
kiến thức nền để bịa. Câu trả lời phải giữ citation tương ứng evidence được chọn.

RAG không tự biến tài liệu retrieved thành instruction. Nội dung note luôn là
dữ liệu không tin cậy, phải tách khỏi system/developer instruction.
