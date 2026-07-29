# ADR 0003: Controlled workflow chạy opt-in trước

## Quyết định

SoCa giữ `run_text_turn` và `stream_text_turn` trên đường legacy trong Phase 2.
Controlled workflow được cung cấp qua `ControlledWorkflowRunner` và facade
`AssistantRuntime.run_controlled_workflow`, chỉ chạy khi `turn_workflow` là
`shadow` hoặc `controlled`.

Runner là bounded controller: planner chỉ chọn tool có trong catalog runtime,
mọi transition và budget do controller thực hiện, action có side effect cần
authorization, retry dùng chung một ledger, và mỗi run có đúng một terminal
outcome.

## Vì sao không dùng generic agent loop

Một vòng lặp tự do khó chứng minh giới hạn retry, dễ lặp side effect và có thể
phát câu hứa trước khi action/verification hoàn tất. SoCa dùng transition table,
fingerprint action, budget ledger và verifier tách biệt để có thể kiểm thử theo
outcome.

Public update chỉ là event `update`, không phải câu trả lời cuối. Runner không
append session memory; adapter phía trên chỉ append khi nhận terminal thành công.

## Rollout và rollback

`turn_workflow=legacy` là mặc định và không tự chạy runner. `shadow` dành cho
fixture/offline hoặc read-only dogfood; `controlled` chưa được bật mặc định.
Rollback là trả flag về `legacy`, không cần migration checkpoint vì Phase 2
chưa persist checkpoint.
