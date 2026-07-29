# Vòng đời embedding index

Index dense được xem là artifact sinh từ `(vault, content digest, model
fingerprint)`, không phải nguồn dữ liệu gốc. Markdown vẫn là source of truth.

Khi thêm note mới, index coordinator tạo chunk mới và chỉ embed chunk chưa có
content hash. Khi sửa note, chunk có nội dung thay đổi nhận ID mới và vector cũ
không được tái sử dụng cho chunk đó. Khi xóa note, generation kế tiếp bỏ chunk
đã mất rồi swap atomically.

## Kiểm tra vận hành

1. verify manifest và permissions trước khi load;
2. build generation mới trong thư mục tạm;
3. fsync/đóng file rồi mới đổi manifest pointer;
4. giữ generation cũ theo retention để rollback;
5. chạy `inspect` và `gc` khi cần, không xóa vector đang được process dùng.

Dense search chỉ là một tín hiệu; evidence gate phải kiểm tra relevance sau
retrieval, không coi mọi vector top-k là bằng chứng.
