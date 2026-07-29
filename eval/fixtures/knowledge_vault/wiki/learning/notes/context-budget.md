# Context budget và output reserve

Context window của model bao gồm system prompt, memory, knowledge, câu hỏi và
output reserve. `max_tokens` là giới hạn output, không phải toàn bộ input window.

Vì vậy runtime phải dựng prompt manifest trước khi gọi model: mỗi component có
ước lượng token, priority và trạng thái required/optional. Nếu model chỉ có
context nhỏ, các component ít ưu tiên bị bỏ trước; không cắt system hoặc câu hỏi
hiện tại. Provider report sau lượt gọi dùng để hiệu chỉnh safety margin cho các
lượt sau.

Một prompt đủ nhỏ nhưng output reserve bằng không vẫn có thể làm provider từ
chối. Đây là lý do cần giữ reserve và clamp theo capability của model.
