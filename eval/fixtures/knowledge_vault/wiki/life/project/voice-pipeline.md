# Voice pipeline và knowledge

Voice và chat phải dùng cùng execution semantics sau khi ASR tạo ra text frame:
guardrail, route, retrieval, context, LLM, verify và terminal outcome. Khác biệt
chỉ nằm ở adapter audio, streaming và TTS.

Voice không nên tự gọi một retrieval gate khác rồi truy vấn lần hai. Nếu đã có
query embedding và retrieval batch, controller truyền lại batch đó cho context
builder. Progress UI chỉ mô tả stage thật như `retrieve`, `compose` hoặc `verify`,
không dùng timer giả để làm người dùng tưởng task đã hoàn tất.
