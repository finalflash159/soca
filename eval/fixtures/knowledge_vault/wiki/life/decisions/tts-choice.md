---
type: life_decision
area: personal-tooling
status: current
created: 2026-07-18
updated: 2026-07-26
tags: [tts, privacy, local, decision]
source_kind: sanitized-life-vault-simulation
---

# Vì sao tôi chọn giọng đọc local

## Bối cảnh

Tôi muốn dùng SoCa bằng giọng nói hằng ngày, không chỉ chạy một câu demo. Vì
voice có transcript, memory và những câu nói riêng tư, lựa chọn TTS không thể
chỉ nhìn chất lượng âm thanh cao nhất trên một đoạn mẫu.

## Các tiêu chí tôi ghi ra trước khi chọn

| Tiêu chí | Trọng lượng cảm nhận | Cách tôi kiểm |
| --- | ---: | --- |
| riêng tư transcript | cao | request có rời máy không |
| tiếng Việt tự nhiên | cao | tên riêng, câu dài, số và viết tắt |
| latency | cao | first audio và total generation |
| barge-in | cao | dừng giữa câu có sạch không |
| ổn định | cao | 20 lượt liên tiếp có lỗi không |
| resource | vừa | cold start, RSS, audio buffer |

## Quyết định

Tôi giữ TTS local làm mặc định. Lý do chính là transcript không phải rời khỏi
máy và lúc demo không phụ thuộc mạng. Đây là default theo privacy/risk, không phải
khẳng định local luôn có chất lượng tốt nhất.

## Trade-off tôi chấp nhận

Giọng local có thể kém tự nhiên hơn remote ở câu dài, dấu câu và tên riêng. Cold
start cũng làm lượt đầu chậm. Tôi chấp nhận điều đó khi câu hỏi chứa dữ liệu
riêng tư; với một lượt đánh giá chất lượng có chủ ý, tôi có thể bật remote và
hiểu rõ transcript sẽ được gửi ra ngoài.

## Cách tôi test

- câu chào và câu ngắn;
- câu tiếng Việt có nhiều dấu;
- số tiền và ngày tháng;
- tên model/code-mix;
- câu dài cần nhiều sentence chunk;
- user cắt lời giữa audio;
- provider/model fail giữa stream.

Tôi ghi first audio, completion, lỗi, model ID và cảm nhận nghe riêng. Không ghi
“đã chọn model tốt nhất” nếu mới nghe một câu.

## Khi nào tôi xem lại quyết định

Tôi sẽ xem lại nếu model local mới giảm rõ cold start mà vẫn giữ privacy, hoặc nếu
barge-in không đạt contract. Remote chỉ trở thành default khi privacy policy,
chi phí và user consent đều thay đổi có chủ ý.

## Liên hệ

Quyết định này liên quan `privacy-boundary.md`, nhưng không phải bằng chứng cho
query về ngân sách hay kiến thức kỹ thuật.

## Cách tôi đánh giá lại lựa chọn

Tôi ghi thử cùng một tập câu: câu ngắn tiếng Việt, tên riêng, số, từ code-mix,
câu dài và đoạn có dấu câu. Tôi nghe các lỗi nuốt âm, dấu thanh, ngắt câu và
thời gian từ lúc text sẵn sàng đến lúc có audio. Một demo một câu rất dễ làm tôi
đánh giá quá cao model.

Tôi tách cold-start khỏi warm latency. Cold-start ảnh hưởng lượt đầu sau khi
khởi động; warm latency ảnh hưởng cảm giác hội thoại liên tục. Tôi cũng ghi RAM
đỉnh, audio queue, khả năng cancel khi barge-in và việc model có giữ file tạm
hay không.

## Khi nào phải đổi quyết định

- lỗi phát âm tên riêng lặp lại trên tập câu thật;
- barge-in làm câu mới bị trễ hoặc audio cũ không dừng;
- model local chiếm tài nguyên khiến ASR/LLM bị starvation;
- license hoặc model update đổi quyền sử dụng;
- remote tốt hơn rõ ràng nhưng privacy/consent chưa có thiết kế phù hợp.

Tôi không đổi chỉ vì một sample nghe hay. Tôi cần artifact của benchmark, môi
trường, model revision và failure examples để quyết định có thể reproduce.
