# Context budget calibration

Ngày chạy: 2026-07-29  
Branch: fix/model-context-budget

Đây là release smoke/calibration, không phải benchmark chất lượng RAG. Không
dùng knowledge demo làm dữ liệu đo.

## Contract đã kiểm tra

- Local model capability lấy từ LLM_MODEL_REGISTRY.
- Remote model capability lấy từ provider catalog và lưu vào LlmSettings.
- Tokenizer adapter ưu tiên engine.count_tokens(), fallback về UTF-8/4 chỉ khi
  engine không có tokenizer hoặc tokenizer lỗi.
- PromptAssembler giữ system, current input, answer prefix và evidence bắt
  buộc; optional memory/evidence bị loại theo priority nếu không đủ input budget.
- Output reserve được clamp bởi model_max_output_tokens và phần context còn
  lại sau required prompt.
- Runtime/structured planner dùng admission margin mặc định 128 token; margin
  tự tăng nếu provider quan sát delta dương lớn hơn margin hiện tại.
- Manifest ghi prompt_hash, component token count, safety margin, effective
  output, provider-reported prompt/completion token và delta calibration.

## Matrix và static gate

| Context window | Kết quả |
|---:|---|
| 2,048 | required input giữ nguyên, output clamp |
| 4,096 | required input giữ nguyên |
| 16,384 | required input giữ nguyên |
| 32,768 | required input giữ nguyên |

Kết quả test:

~~~
uv run pytest -q
1078 passed, 3 skipped, 3 warnings
~~~

Ruff, Pyright và git diff --check đều đạt.

UI gate: npm run typecheck, 49 Vitest tests và npm run build đều đạt.

## Real remote smoke

Provider/model: OpenRouter / google/gemini-3.5-flash-lite  
Catalog capability: context 1,048,576, max output 65,536  
Requested output: 2,048

### Free-chat

- used_llm=True, blocked=False, route free_chat.
- Manifest prompt estimate: 328 tokens.
- Provider-reported prompt: 344 tokens.
- Calibration delta: +16.
- Có prompt hash và output có nội dung.

### Knowledge → context → LLM

Query: wiki: định lý Bayes

- route knowledge_llm;
- used_tool=True, tool knowledge.search;
- used_llm=True, blocked=False;
- knowledge component 318 tokens và required=True;
- manifest prompt estimate 670, provider report 741, delta +71;
- response có nội dung và citation.

Post-review-fix knowledge smoke vẫn đạt với margin `128`: route
`knowledge_llm`, tool + LLM đều chạy, prompt estimate `641`, provider report
`714`, delta `+73`, knowledge required và prompt hash đều có mặt.

### Controlled workflow planner

Goal: `Cho tôi biết giờ hiện tại`

- remote structured planner chạy thật, sau đó `local_time.now` chạy thật;
- terminal `succeeded`, một model call, một tool call, bảy transitions;
- planner prompt estimate 164, provider report 156, delta -8;
- planner output reserve `256` và prompt hash có mặt.

Smoke này đồng thời phát hiện retrieval quality còn việc cho Phase 4: truy vấn
Bayes vẫn kéo thêm hai hit không liên quan (tts-decision.md và
safety-boundaries.md). Chúng không gây overflow, nhưng không được dùng kết quả
này để kết luận RAG đã đạt chất lượng release.

## Giới hạn cần ghi nhớ

Nếu persisted remote settings chưa có capability và catalog chưa fetch xong,
manifest sẽ báo context_window=null; runtime không thể biết một hard limit
không được provider công bố. UI phải hoàn tất catalog refresh trước khi coi
model capability là ready; known capability thì không gửi prompt vượt budget.
