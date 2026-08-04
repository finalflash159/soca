---
type: learning_note
domain: software-engineering
topic: debugging-and-testing
status: active
created: 2026-07-25
updated: 2026-07-28
tags: [software, debugging, testing, regression, observability]
source_kind: personal-study-note
---

# Software engineering: debug bằng bằng chứng, test bằng contract

## Tôi từng làm thế nào

Tôi từng thấy lỗi rồi sửa dòng gần nhất cho test xanh. Cách đó có thể xóa triệu
chứng nhưng không giải thích nguyên nhân. Giờ tôi cố ghi lại reproduction, expected
behavior, actual behavior, môi trường và boundary trước khi chạm code.

## Debug loop

```text
reproduce
  → thu hẹp input
  → quan sát state/trace
  → lập giả thuyết
  → thử một thay đổi nhỏ
  → xác nhận nguyên nhân
  → regression test
  → kiểm side effect
```

Nếu thay nhiều thứ cùng lúc, tôi mất khả năng biết giả thuyết nào đúng. Nếu lỗi
không reproducible, tôi thu telemetry: commit, OS, Python/Node version, model,
cache state, timing và input shape.

## Test không chỉ là code coverage

Unit test bảo vệ hàm nhỏ. Integration test bảo vệ boundary giữa index, runtime,
tool và LLM adapter. End-to-end test kiểm user-visible outcome. Real smoke với
model/provider thật kiểm wiring mà fake không thể chứng minh.

Coverage cao vẫn có thể không test failure mode. Tôi ưu tiên contract:

- input rỗng bị xử lý đúng;
- tool failure không bị trả như success;
- citation ID không tồn tại bị reject;
- prompt budget không overflow;
- cancellation không append partial turn;
- retry không nhân side effect.

## Regression test tốt

Regression nên tái hiện bug bằng input nhỏ nhất, tên nói được behavior và assert
outcome quan trọng. Không assert toàn bộ transcript nếu chỉ cần assert route/tool/
terminal/citation; snapshot quá rộng làm test dễ gãy mà không bảo vệ đúng.

## Property và metamorphic test

Property test kiểm invariant trên nhiều input. Ví dụ normalize dấu không làm đổi
path identity, chunk ID ổn định với cùng text, compact không làm mất turn ngoài
phần được phép.

Metamorphic test kiểm quan hệ: thêm dấu tiếng Việt không được làm lexical search
đột nhiên mất document; đổi thứ tự candidate không đổi stable tie-break; thêm
unrelated note không làm answerable query đổi evidence đúng thành note lạ.

## Debug retrieval

Tôi giữ query gốc và normalized query. Sau đó xem candidate trước gate, candidate
đã reject, backend signal, floor, margin và selected evidence. Nếu chỉ nhìn top
answer, tôi không phân biệt được lỗi retriever với lỗi model.

## Debug LLM output

Tôi kiểm prompt hash, manifest component, model ID, effective max output, finish
reason và raw usage. Empty output có thể do reasoning mandatory, output reserve,
provider refusal hoặc adapter parse sai. Tăng max token chỉ là một giả thuyết,
không phải fix mặc định.

## UI và terminal

UI progress phải phản ánh stage thật: retrieve, compose, generate, verify. Không
dùng timer giả. Input IME có provisional event; test phải mô phỏng composition,
delete, Tab palette và resize để phát hiện layout jump.

## Release checklist cá nhân

1. diff chỉ gồm scope đã yêu cầu;
2. không có secret/log nhạy cảm;
3. unit + integration xanh;
4. real local/remote smoke nếu logic chạm provider;
5. branch/commit/PR rõ;
6. Qodo review đã có comment thật, không chỉ summary;
7. fix review xong thì CI green rồi merge;
8. sau merge checkout main và pull trước phase tiếp.

## Câu hỏi mở

- test nào nên chạy mỗi commit, test nào nightly;
- khi nào fake adapter làm người phát triển quá tự tin;
- trace retention bao lâu là đủ mà vẫn riêng tư;
- làm sao hiển thị failure cho user mà không lộ chain-of-thought.

## Tóm tắt

Debug là quá trình cập nhật giả thuyết bằng observation. Test là cách đóng đinh
contract để bug không quay lại. Tôi muốn test bảo vệ quyết định và outcome, không
chỉ làm con số coverage đẹp.

## Reproduction tối thiểu

Tôi ghi command, branch/commit, input, config, environment, expected và actual.
Nếu lỗi phụ thuộc IME, terminal hoặc provider, tôi ghi cả điều kiện đó. “Thỉnh
thoảng fail” chưa phải reproduction; tôi cần frequency, log/trace và cách reset.

Tôi thử giảm input nhưng không xóa chính điều kiện làm lỗi. Với lỗi retrieval,
giữ query, corpus revision, index generation và embedding model. Với lỗi UI,
giữ terminal size, key sequence và snapshot/state transition.

## Unit, integration và real flow

Unit test kiểm pure decision nhanh. Integration test kiểm tool/runtime/source
thật ở boundary. Real-flow smoke chạy process, model/provider thật và resource
thật. Ba tầng trả lời ba câu khác nhau; unit xanh không chứng minh provider remote,
tokenizer, model load hay terminal IME đang ổn.

Tôi ghi rõ test nào mock LLM. Mock hữu ích để kiểm branching/retry, nhưng không
được gọi là “LLM thật”. Real smoke có thể tốn tiền/thời gian nên chạy có chủ ý,
redact output và lưu metadata không lưu secret.

## Regression cho bug khó chịu

Với input composition, test chuỗi `insert/update/delete`, IME marked text, paste,
backspace và resize terminal. Với compact, test dưới ngưỡng, đủ lượt, empty
summary, timeout, retry và giữ dữ liệu gốc. Với RAG, test lexical miss/dense
rescue, distractor, no-answer, citation sai và stale index.

## Observability test

Mỗi phase cần reason/status typed để test không phải parse câu UI. Tôi muốn assert
progress sequence, trace flags, tool calls, evidence metadata và final outcome.
Snapshot text chỉ dùng cho rendering; contract thật nằm ở state.

## Failure budget và rollback

Không sửa bằng cách nới timeout vô hạn, nuốt exception hoặc disable test. Nếu cần
temporary fallback, ghi reason, scope và expiry. Commit nhỏ giúp bisect; benchmark
artifact giúp biết change làm p95/recall/memory xấu đi.

## Checklist trước commit

- đọc diff và phát hiện file ngoài scope;
- chạy formatter/linter/type checker liên quan;
- unit + integration của boundary;
- real-flow nếu đổi engine/tool/prompt;
- kiểm secrets/permission/artifact;
- ghi failure còn lại, không tuyên bố pass quá mức;
- commit message nói outcome, không nói “fix stuff”.
