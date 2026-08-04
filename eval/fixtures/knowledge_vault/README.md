# SoCa showcase knowledge vault

Đây là corpus dùng cho smoke test, UI demo và kiểm tra thủ công. Nội dung được
viết theo dạng một vault cá nhân đã được sanitise: learning notes dài theo góc
nhìn người viết, journal theo ngày, finance có actual/planned và health có safety
boundary. Nó không đại diện cho dữ liệu cá nhân thật và không
được dùng làm benchmark chất lượng retrieval/release.

Corpus có các lát chính:

- `wiki/learning/`: ghi chú DSA, systems, ML, DL, LLM, serving và fundamentals.
- `wiki/life/journal/`: nhật ký và weekly review mẫu.
- `wiki/life/finance/`: kế hoạch chi tiêu mẫu.
- `wiki/life/health/`: thông tin sức khỏe phổ thông có ranh giới an toàn.

Các note có metadata, liên kết chéo, lịch sử/ngữ cảnh, ví dụ, câu hỏi follow-up,
planned-vs-actual distinction, tài liệu nguồn và hard negative tự nhiên để việc
demo không chỉ là tìm đúng một keyword.

Khi muốn chạy demo trong vault runtime, copy nội dung fixture này vào vault đã
được UI khởi tạo. Index và vector không nằm trong fixture; chúng được tạo tại
`Knowledge/.soca/knowledge_index/` sau thao tác `Index vault`.
