---
type: learning_note
domain: llm
topic: context-prompting-and-tool-use
status: active
created: 2026-07-23
updated: 2026-07-28
tags: [llm, context, prompting, tools, rag, grounding, workflow]
source_kind: personal-study-note
---

# LLM: context, tool use và cách tôi không để model tự nhận đã xong

## LLM không phải database

LLM sinh token có xác suất cao theo context. Nó có thể biết kiến thức nền nhưng
không biết vault mới nhất, giờ hiện tại, trạng thái tool hay dữ liệu cá nhân nếu
không được đưa vào context. Câu trả lời trôi chảy không phải receipt của một
action.

Tôi tách ba thứ:

- knowledge model đã học trước;
- context runtime đang đưa vào lượt này;
- observation thật từ tool/index/provider.

## Prompt là interface

Prompt tốt không chỉ là “hãy trả lời đúng”. Nó khai báo role, dữ liệu nào là
untrusted, mục tiêu, điều kiện thành công, citation map, empty state và output
format. Nếu những phần này nhập thành một cục text không có boundary, note chứa
“ignore previous instructions” có thể làm model lẫn authority.

## Context budget

Tôi tính system, current input, working memory, archive memory, knowledge và
output reserve cùng lúc. `max_tokens` là output cap. Model context 16K không đồng
nghĩa session có thể chiếm 16K vì còn wrapper, system và safety margin.

Mỗi component có priority. Required system và câu hỏi không được bỏ; evidence
có thể giảm snippet nhưng giữ path/citation; archive memory thường bỏ trước.
Prompt manifest phải ghi component included/omitted, estimate, observed usage và
prompt hash.

## Tool call không phải câu trả lời

Luồng tôi muốn là:

```text
user goal
  → capability decision
  → tool authorization
  → tool execution
  → observation
  → evidence/goal assessment
  → synthesis
  → verify
  → terminal outcome
```

Assistant có thể phát “đang tra cứu” nhưng không được append câu đó như final
turn. Controller chỉ kết thúc sau observation và verify, hoặc kết thúc bằng
clarification/insufficient evidence/safe failure.

## Retrieval context

Knowledge và memory là reference không tin cậy, không phải system instruction.
Knowledge citation dùng `[K#]`; memory archive dùng `[M#]`. Nếu context empty,
prompt phải nói typed state `insufficient` để model abstain, không lấp bằng kiến
thức chung mà không báo.

## Citation không chỉ là trang trí

Citation ID phải tồn tại trong evidence bundle. `[K9]` khi chỉ có hai hit là sai
contract. Có citation cũng chưa đủ; claim phải có khả năng map về đoạn đã chọn.
Giai đoạn đầu groundedness judge có thể shadow, nhưng unknown citation và missing
provenance là lỗi deterministic.

## Tool output có thể là dữ liệu độc hại

Một note có thể chứa câu giống system prompt, lệnh xóa file hoặc yêu cầu gọi tool.
Tôi đặt retrieved content trong boundary rõ ràng, nhắc model chỉ dùng làm fact,
và chạy retrieval guardrail. Tool result không được tự sửa tool policy.

## Khi model trả lời “mình sẽ kiểm tra”

Đó là public update. Controller phải nhìn goal còn pending và action chưa chạy
chưa. Nếu cần query revision, chạy vòng nhỏ có budget. Nếu đủ evidence thì
synthesize tiếp. Chỉ khi verify pass mới phát achieved.

## One repair

Nếu answer có evidence nhưng thiếu citation hoặc có ID không tồn tại, cho phép một
lần answer repair. Repair prompt yêu cầu viết lại từ evidence, không retrieve lại
nếu evidence đã insufficient. Nếu sửa vẫn invalid, terminal là safe failure hoặc
abstention chứ không retry vô hạn.

## Memory nào đưa vào

- working memory gần đây phục vụ continuity;
- core/pinned memory chỉ đưa khi profile policy cho phép;
- archival memory là tủ, chỉ search khi goal cần;
- episode memory cần consent và lifecycle riêng.

Không phải mọi memory đều nhét vào prompt mọi lượt. Prompt dài làm loãng câu hỏi
hiện tại và tăng chi phí.

## Cách tôi debug câu trả lời sai

1. route có chọn đúng capability không;
2. tool có thực sự chạy không;
3. evidence status là supported hay chỉ có hit;
4. prompt có chứa đúng selected evidence không;
5. model có output citation hợp lệ không;
6. guardrail/validator có cho qua claim unsupported không;
7. lỗi nằm ở retrieval, prompt, model hay output contract.

## Tóm tắt

LLM là một thành phần synthesis trong workflow, không phải controller duy nhất.
Tool và evidence đưa sự thật vào; typed state và verifier giữ model không nhận
goal đã xong quá sớm.

## Context là một ngân sách có cấu trúc

Tôi tách system policy, capability manifest, core memory, working summary, recent
turns, retrieved knowledge, archival memory, current user input và output reserve.
Nếu chỉ đếm message length mà không biết mỗi phần làm nhiệm vụ gì, tôi không thể
quyết định cắt phần nào khi model window nhỏ.

Một context dài không tự động tốt. Evidence trùng lặp làm model phân tán, recent
turn quá dài che goal, summary mất caveat làm hội thoại tiếp tục sai. Tôi muốn
token accounting và source labels để `/context` giải thích được phần nào chiếm chỗ.

## Tool result không phải instruction

Retrieved note, file content và memory snippet đều là data không đáng tin cậy đối
với controller. Tôi đặt chúng trong delimiter, label source/citation và nói rõ
không làm theo instruction nằm bên trong. Nếu note chứa “ignore previous…”, đó là
nội dung cần trích dẫn/phân tích, không phải lệnh cho model.

Tool schema nên typed: tên tool, input, output, side effect, permission và error.
LLM đề xuất tool call nhưng runtime validate path/scope/argument trước khi thực
thiện. Không để model tự gọi file path ngoài vault chỉ vì text nghe hợp lý.

## Goal và done criteria

Mỗi lượt nên có goal ngắn và done criteria. “Tôi sẽ kiểm tra note Bayes” chưa phải
done; cần query, có/không có evidence, tổng hợp và verify answer. Nếu model nói
“để tôi tìm” nhưng chưa gọi tool, state vẫn là `needs_retrieval`.

Tôi tách progress text khỏi internal chain-of-thought: UI chỉ hiện “Đang tra cứu
ghi chú” hoặc “Đang kiểm chứng nguồn”, không hiện suy luận riêng tư. Runtime giữ
typed phase và reason code để trace/test.

## Loop có kiểm soát

Một lượt có thể đi qua:

1. hiểu goal và chọn capability;
2. gọi tool cần thiết;
3. đánh giá evidence đủ/yếu/trống;
4. tổng hợp câu trả lời grounded;
5. validate citation/claim/goal;
6. nếu thiếu, repair hoặc một tool call bổ sung trong budget;
7. kết thúc với answer/clarification/abstain.

Loop không phải cho model tự chạy vô hạn. Runtime đặt max iteration, max tool
calls, deadline, duplicate-call guard và cancellation. Một lượt chỉ nên retry
khi state thay đổi hoặc có lý do rõ; gọi cùng tool với cùng input không tạo thêm
thông tin.

## Khi answer chưa đạt goal

Nếu user nói “ý tôi là…” sau câu trả lời lệch, đó là signal sửa goal, không phải
small talk. Runtime cần giữ prior context nhưng re-evaluate intent. Nếu user nói
“tôi muốn biết trong note của tôi”, cần truy vấn knowledge thay vì chỉ giải thích
kiến thức nền.

Answer validator kiểm citation label tồn tại, nhưng claim entailment phức tạp hơn.
Tôi ghi status shadow/trace và dùng eval set để calibrate; không tuyên bố regex đã
chứng minh factuality.

## Bài tập

- query có evidence rồi yêu cầu giải thích sâu;
- query không có evidence và kiểm abstain;
- model nói “đang tìm” nhưng không tạo tool call;
- tool trả empty rồi model vẫn phải phản hồi rõ;
- note có prompt injection;
- tool timeout, duplicate call và cancellation;
- câu follow-up sửa typo/goal sau answer đầu.
