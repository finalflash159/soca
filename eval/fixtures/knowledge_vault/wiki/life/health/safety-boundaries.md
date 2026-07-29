---
type: health_note
area: assistant-safety
status: current
created: 2026-07-11
updated: 2026-07-29
confidence: high
tags: [health, safety, assistant, disclaimer, escalation]
source_kind: redacted-personal-note
---

# Ranh giới khi assistant nói về sức khỏe

## Tôi muốn assistant giúp ở mức nào

Assistant có thể giúp tôi tìm lại note, tóm tắt nguồn giáo dục chung, sắp xếp
câu hỏi cho chuyên gia và nhắc rằng dữ liệu đang thiếu. Nó không được chẩn đoán,
kê thuốc, tự đổi liều, bảo dừng điều trị hoặc dùng một journal ngắn để kết luận
người cụ thể mắc bệnh gì.

Tôi muốn câu trả lời hữu ích nhưng có giới hạn rõ: “đây là thông tin chung”,
“note không đủ để kết luận”, “nên hỏi chuyên gia” và “nếu có dấu hiệu cấp tính,
hãy tìm trợ giúp ngay” phải xuất hiện khi bối cảnh cần, không phải một câu
disclaimer vô nghĩa dán vào mọi câu.

## Bốn tầng rủi ro

| Tầng | Ví dụ | Hành vi mong muốn |
| --- | --- | --- |
| giáo dục chung | nguyên tắc bữa ăn | trả lời có nguồn và phạm vi |
| tự quan sát | ngủ, vận động, bữa ăn | mô tả, không gắn chẩn đoán |
| cá nhân hóa cao | thuốc, bệnh nền, thai kỳ | hỏi/chuyển chuyên gia |
| cấp tính | khó thở, ngất, đau ngực | hướng tới trợ giúp khẩn cấp |

Từ khóa không đủ để phân tầng một mình. “Đường huyết” trong một câu hỏi chung
khác với “tôi đang dùng thuốc X, đổi bữa ăn thế nào”. Runtime cần đưa query và
evidence vào một policy có thể kiểm tra, không để model tự bỏ qua boundary.

## Checklist trước khi trả lời

1. User hỏi kiến thức chung hay lời khuyên cho một người cụ thể?
2. Note có ngày và nguồn không?
3. Evidence là health note hay chỉ là journal?
4. Có thuốc, bệnh nền, dị ứng, thai kỳ, trẻ nhỏ hoặc người cao tuổi không?
5. Có dấu hiệu cấp tính không?
6. Câu trả lời có biến ví dụ thành prescription không?
7. Có cần nói rõ thiếu dữ liệu và đề nghị câu hỏi cho chuyên gia không?

## Các failure tôi muốn bắt

- trả “ăn món X là khỏi” từ note balanced meals;
- lấy bảng recovery và kết luận nguyên nhân đau;
- biến “tôi sẽ hỏi bác sĩ” thành “bác sĩ đã xác nhận”;
- nói một con số chính xác mà note không có;
- bỏ disclaimer khi user nhắc thuốc hoặc bệnh nền;
- trấn an chắc chắn trong khi có dấu hiệu cần khám sớm;
- nói “tôi đã kiểm tra” dù retrieval chưa chạy hoặc trả empty.

## Cách tôi muốn hiển thị provenance

Nếu câu trả lời dựa trên `balanced-meals.md`, citation phải nói đó là note nguyên
tắc chung. Nếu dựa trên journal, phải gọi đúng là quan sát trong ngày. `M#` của
memory không biến nó thành clinical evidence. Không có nguồn đủ mạnh thì câu trả
lời phải chuyển sang `insufficient`, không bịa cho tròn đoạn.

## Quy tắc riêng cho voice

Voice cần câu ngắn, nói rõ khi đang tìm note và không đọc một đoạn disclaimer dài
làm che mất điểm chính. Nhưng ngắn không có nghĩa bỏ safety. Với dấu hiệu cấp
tính, câu đầu phải hướng người dùng tìm trợ giúp; không đợi assistant “suy luận
xong” mới nói.

## Kết luận

Safety boundary là một contract của sản phẩm. Note này mô tả cách tôi muốn
assistant hành xử; policy runtime và hướng dẫn chuyên môn mới là authority thật.
