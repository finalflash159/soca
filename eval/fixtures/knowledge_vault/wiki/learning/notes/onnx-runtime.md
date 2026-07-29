# ONNX Runtime trong pipeline local

ONNX Runtime là lớp thực thi model ONNX, không phải một model mới. Pipeline local
có thể chọn execution provider theo phần cứng: ưu tiên `CoreMLExecutionProvider`
trên Apple Silicon khi graph tương thích, sau đó dùng `CPUExecutionProvider` làm
fallback.

## Điều cần kiểm tra

- model có input/output shape đúng với graph hay không;
- provider có được runtime báo là available hay chỉ được khai báo trong config;
- fallback CPU có làm latency tăng nhưng vẫn trả kết quả đúng hay không;
- phiên bản ONNX Runtime có tương thích với opset của model.

Note này giải thích khái niệm và checklist vận hành, không ghi kết quả benchmark
cho mọi model.
