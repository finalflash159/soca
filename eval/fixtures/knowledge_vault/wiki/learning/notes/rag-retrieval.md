# RAG: từ câu hỏi đến câu trả lời

Luồng mong muốn là `normalize → sparse và dense độc lập → fusion → relevance
gate → chọn evidence → prompt → LLM → citation/groundedness check`.

Sparse phù hợp với tên riêng, mã model và cụm từ xuất hiện nguyên văn. Dense có
thể cứu truy vấn diễn đạt khác từ vựng, nhưng cũng có thể trả về đoạn tương tự
chủ đề mà không trả lời được câu hỏi. Vì vậy `có hit` chỉ là candidate, chưa phải
evidence đủ mạnh.

Nếu không có đoạn đủ liên quan, assistant phải truyền trạng thái thiếu bằng
chứng vào prompt và nói rõ chưa tìm thấy trong vault. Không lấy kiến thức chung
của model để lấp khoảng trống mà không báo cho người dùng.
