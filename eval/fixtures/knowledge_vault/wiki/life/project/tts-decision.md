# Quyết định TTS cho SoCa

## Quyết định hiện tại

Baseline TTS local được giữ làm đường chạy mặc định vì không gửi transcript ra
ngoài và dễ kiểm soát khi demo offline. Valtec là candidate cần kiểm tra thêm về
độ ổn định graph và latency; không coi tên model trong config là bằng chứng model
đã load thành công.

## Vì sao chưa chốt một model duy nhất

- chất lượng phát âm tên riêng và câu dài chưa đồng đều;
- thời gian khởi động model ảnh hưởng trải nghiệm voice;
- barge-in cần dừng audio sạch, không chỉ tạo được waveform;
- kết quả nghe phải được đánh giá trên câu tiếng Việt tự nhiên.

Decision này liên quan TTS, không phải quyết định embedding cho knowledge.
