---
type: health_note
area: safety
status: permanent-boundary
created: 2026-07-10
updated: 2026-07-27
tags: [health, safety, disclaimer, escalation]
source_kind: sanitized-life-vault-simulation
---

# Ranh giới an toàn khi assistant nói về sức khỏe

## Vì sao tôi cần một note riêng

Các note sức khỏe dễ làm câu trả lời nghe có vẻ chắc chắn hơn dữ liệu thật. Tôi
muốn assistant giúp tôi hiểu nguyên tắc chung, nhưng không muốn nó chẩn đoán,
đổi thuốc hoặc biến một ví dụ dinh dưỡng thành lời khuyên cá nhân.

## Phạm vi được phép

Assistant có thể:

- giải thích khái niệm dinh dưỡng phổ thông;
- tóm tắt chính note đã có trong vault;
- giúp tôi lập câu hỏi để hỏi bác sĩ;
- nhắc rằng thông tin cần được cá nhân hóa;
- chỉ ra note nào đang được dùng làm nguồn.

Assistant không được:

- chẩn đoán bệnh từ vài triệu chứng;
- đưa liều thuốc hoặc bảo ngừng thuốc;
- khẳng định thực đơn phù hợp với bệnh nền;
- xem một note cũ là hồ sơ y tế cập nhật;
- nói “chắc chắn không sao” khi thiếu khám/đo lường;
- dùng model knowledge để lấp empty retrieval mà không báo.

## Tín hiệu cần escalation

Nếu người dùng mô tả khó thở, đau ngực, ngất, yếu liệt đột ngột, chảy máu nhiều,
phản ứng dị ứng nặng, lú lẫn cấp tính hoặc tình trạng nguy hiểm khác, assistant
nên khuyên tìm trợ giúp y tế khẩn cấp phù hợp. Note này không thay thế hướng dẫn
của dịch vụ khẩn cấp tại nơi người dùng đang ở.

Với bệnh nền, thai kỳ, trẻ nhỏ, người cao tuổi hoặc thuốc đang dùng, mọi thay đổi
lớn về ăn uống/vận động cần hỏi bác sĩ hoặc chuyên gia được cấp phép.

## Cách tôi muốn câu trả lời được nói

1. nói rõ đây là thông tin chung;
2. trích note nếu có evidence;
3. nêu giới hạn dữ liệu;
4. hỏi thêm hoặc khuyên gặp chuyên gia khi cần;
5. không dùng giọng hoảng sợ cho chuyện bình thường;
6. không dùng giọng chắc chắn cho chuyện chưa đủ dữ kiện.

## Kiểm tra trước khi trả lời

- câu hỏi đang hỏi giáo dục hay hỏi chẩn đoán?
- có người cụ thể, thuốc cụ thể hoặc triệu chứng cấp tính không?
- note nguồn có ngày cập nhật không?
- evidence là health note hay chỉ là note ăn uống chung?
- câu trả lời có vô tình biến ví dụ thành prescription không?

## Ghi chú

Nội dung trong fixture là mô phỏng sanitized. Khi dùng SoCa với dữ liệu thật,
user phải thay boundary này bằng policy được chuyên gia duyệt nếu sản phẩm phục
vụ use case y tế.

## Cách phân biệt câu hỏi chung và câu hỏi cá nhân

“Nguyên tắc chung của bữa ăn cân bằng là gì?” có thể trả lời ở mức giáo dục với
nguồn và disclaimer phù hợp. “Tôi đang dùng thuốc X, nên ăn bao nhiêu món Y?”
đã là câu hỏi cá nhân cần chuyên gia. Cùng một keyword không có nghĩa cùng một
mức rủi ro.

## Các lỗi tôi muốn bắt

- biến note general-reference thành prescription;
- bỏ mất cảnh báo khi câu hỏi có bệnh nền/thuốc/trẻ nhỏ/thai kỳ;
- dùng journal chưa đầy đủ để kết luận xu hướng sức khỏe;
- nói một con số chính xác mà nguồn không có;
- hứa “tôi đã kiểm tra” khi tool chưa chạy;
- không chuyển hướng khi có dấu hiệu khẩn cấp.

## Output tối thiểu an toàn

Assistant cần nói phạm vi mình có thể giúp, trích nguồn nếu có, nêu uncertainty,
khuyên hỏi chuyên gia khi phù hợp và chỉ ra dấu hiệu cần trợ giúp khẩn cấp. Đây
không phải câu chữ cố định cho mọi tình huống; nó là contract để kiểm tra hành vi.
