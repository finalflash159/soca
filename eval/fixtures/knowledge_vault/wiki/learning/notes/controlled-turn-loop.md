# Controlled turn loop

Một lượt xử lý không kết thúc chỉ vì assistant đã phát câu thông báo như “mình
sẽ kiểm tra”. Public update là sự kiện tiến trình; terminal outcome chỉ được
phát sau khi controller đã có observation và verify.

Các cạnh lặp được giới hạn: revise query tối đa một lần, repair structured output
một lần cho mỗi model call và answer repair tối đa một lần. Mọi retry dùng chung
ledger của lượt, có fingerprint để tránh gọi lại side effect với cùng input.

Các kết thúc hợp lệ gồm achieved, clarification, insufficient evidence, safe
failure, budget exhausted và cancelled. Không lưu public update như một câu trả
lời hoàn tất trong working memory.
