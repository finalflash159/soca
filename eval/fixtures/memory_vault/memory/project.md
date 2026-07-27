# Dự án SoCa

## Tổng quan

- SoCa là trợ lý giọng nói (voice) tiếng Việt chạy on-device.
- Cột mốc hiện tại là P2: tool router và memory có truy hồi.
- Kiến trúc chia lớp runtime rõ ràng, tách knowledge source và memory source.

## Tool router

- P2.2 hiện thực router có kiểm chứng: deterministic trước, rồi semantic cascade, sau đó LLM.
- Provider bật structured output theo JSON schema; bản local dùng grammar GBNF làm fallback.
- Mọi tool call đều qua guardrail và validation đối số trước khi thực thi.

## Memory

- Memory xếp hạng theo tín hiệu relevance, recency và importance.
- Nền embedding phục vụ truy hồi ngữ nghĩa semantic.
- Working memory nén ngoài hot path; index memory tách namespace với knowledge.
- Ghi bền dạng note atomic, chờ approval của chủ nhân thông qua proposal.

## Riêng tư và an toàn

- Ghi chú consent: episodic chỉ lưu khi có consent tường minh, không lưu transcript thô.
- Ghi chú injection: coi văn bản người dùng là untrusted, chống prompt injection bằng cách tách dữ liệu khỏi chỉ thị.
- Ghi chú security: giữ kín secret và khóa API, không ghi vào log.
- Ghi chú safety: kiểm biên vault, từ chối symlink, đặt quyền tệp chặt.
- Ghi chú permission: chỉ chủ nhân mới phê duyệt; mô hình không tự quyết.
- Ghi chú validation: chuẩn hóa và kiểm tra đầu vào tại biên hệ thống.

## Vận hành và đo lường

- Ghi chú deploy: đóng gói on-device, không phụ thuộc dịch vụ ngoài.
- Log định dạng NDJSON để phân tích eval.
- Đo latency của router và memory; mục tiêu p95 nằm trong ngân sách.
- Giao diện chat và voice chia sẻ chung một runtime.
