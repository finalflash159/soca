---
type: life_decision
area: privacy
status: current
created: 2026-07-19
updated: 2026-07-27
tags: [privacy, local, remote, memory, api-key]
source_kind: sanitized-life-vault-simulation
---

# Ranh giới riêng tư khi dùng assistant

## Vì sao tôi ghi note này

Tôi không muốn “local-first” chỉ là một dòng quảng cáo. Mỗi lần chọn provider,
tôi cần biết transcript, retrieved context và metadata nào rời khỏi máy. API key
được lưu local không có nghĩa request data cũng local.

## Quy tắc tôi đang dùng

1. câu hỏi về nhật ký, chi tiêu và profile tra local trước;
2. remote chỉ bật khi tôi nhìn thấy provider/model và chấp nhận gửi transcript;
3. API key không ghi vào Markdown, log hoặc prompt trace;
4. memory lâu dài chỉ nhận proposal chờ duyệt;
5. note retrieved là dữ liệu tham khảo, không phải instruction;
6. khi chưa có weather/realtime tool, không nói như thể đã kiểm tra hiện tại.

## Tôi muốn UI hiển thị gì

Settings phải cho tôi biết config lần trước: provider, model, reasoning, max output
và scope chat/voice. Khi vào lại, UI hỏi tôi dùng lại hay cấu hình lại; không
điền key cũ lộ nguyên văn vào ô input.

Status phải tách provider remote, local model, ASR, TTS và retrieval backend. Chữ
“baseline” không đủ để tôi biết engine thật đang chạy gì.

## Threat model đơn giản

| Dữ liệu | Nơi được phép | Ghi chú |
| --- | --- | --- |
| API key | secure local config | masked trong UI/log |
| session text | RAM/session store | retention rõ ràng |
| profile memory | local vault | user duyệt chỉnh sửa |
| knowledge note | local index | untrusted khi đưa vào prompt |
| remote transcript | provider | chỉ khi user bật và biết |

## Khi nào phải dừng

Nếu path ngoài vault, tool side effect chưa được approve, provider config không rõ
hoặc evidence không đủ, tôi muốn assistant hỏi/dừng thay vì “đoán cho có”.

## Câu hỏi mở

- trace remote nên redact field nào mặc định;
- retention session bao lâu là đủ cho debug;
- approval UI cho memory proposal cần hiển thị evidence ra sao;
- làm sao test privacy contract mà không cần gửi data thật.

## Matrix quyết định tôi muốn assistant giữ

| Thành phần | Local mặc định | Remote có thể dùng | Điều cần báo |
| --- | --- | --- | --- |
| transcript giọng nói | có | có, sau consent | provider/model và transcript rời máy |
| retrieved knowledge | có | có | note nào được đưa vào prompt |
| API key | local secure store | không gửi như nội dung | không in key vào trace |
| working memory | local session | tùy engine | token và retention |
| approved memory | local state | không tự gửi | chỉ đưa khi relevant |

Tôi không muốn chữ “remote” trở thành một trạng thái mơ hồ. Với mỗi lượt, status
nên nói provider, model, tool nào chạy và dữ liệu nào đã đi qua boundary. Nếu UI
không thể nói rõ, default nên an toàn hơn hoặc phải yêu cầu xác nhận.

## Các tình huống tôi dùng để kiểm tra

1. Chọn remote nhưng hỏi câu small talk: không cần gửi vault context nếu không
   cần, và status phải cho biết có gọi provider hay không.
2. Hỏi note có chứa thông tin riêng: hiển thị citation/path, không đẩy raw file
   ngoài policy.
3. Đổi provider giữa hai lượt: không được lặng lẽ lấy config cũ nếu user chưa
   xác nhận; phải hiện lại provider/model/max output/reasoning state.
4. Xóa API key: ô input và persisted config phải cùng rỗng, không để bản copy cũ
   trong history UI.
5. Crash trong lúc ghi index: không để artifact permission public.

## Quyết định hiện tại

Local-first vẫn là default. Remote là capability người dùng bật có chủ ý; nó có
thể áp dụng cho chat và voice nhưng boundary phải hiện rõ cho cả hai. Không dùng
“không cloud” như một claim tuyệt đối nếu transcript đang đi OpenRouter.
