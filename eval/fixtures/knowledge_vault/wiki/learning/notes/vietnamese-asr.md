# Ghi chú xử lý tiếng Việt trong ASR và terminal

Văn bản sau ASR có thể mất dấu, tách từ hoặc chứa lỗi chính tả. Query retrieval
nên normalize dấu ở lớp tìm kiếm nhưng vẫn giữ nguyên câu người dùng để hiển
thị và làm provenance.

Ở terminal, IME có thể phát chuỗi chỉnh sửa tạm thời qua các sự kiện insert,
delete và composition. UI không được coi mọi delete rỗng là một lệnh xóa logic;
phải đồng bộ state sau composition event và tránh render lại input theo từng
event trung gian.

Note này là ghi chú kỹ thuật, không phải danh sách từ khóa để hardcode intent.
