"""Seed a practical coaching knowledge pack into a local Markdown vault."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()

COACH_INDEX_BLOCK = """## Coach Tập Luyện Và Dinh Dưỡng

- [[coach/coach-dashboard|Coach Dashboard]]
- [[coach/todo-system|Todo Hôm Nay Và Checklist Sức Khỏe]]
- [[coach/daily-checkin|Daily Check-in]]
- [[coach/training-time-blocks|Khung Giờ Tập Hằng Ngày]]
- [[coach/training-week-template|Training Week Template]]
- [[coach/monthly-training-checklist|Checklist Tập Luyện Theo Tháng]]
- [[coach/weigh-in-checkin|Cân Ký Và Check-in]]
- [[coach/nutrition-coach-principles|Nguyên Tắc Dinh Dưỡng Và Protein Cho Người Tập Luyện]]
- [[coach/safety-boundaries|Ranh Giới An Toàn]]
- [[sources/coach-fitness-sources|Nguồn Coach Fitness]]
"""

COACH_FILES: Mapping[str, str] = {
    "wiki/coach/coach-dashboard.md": """# Coach Dashboard

#coach #dashboard #training #nutrition

Trang này là hub để SoCa hỗ trợ người dùng như một trợ lý coach tập luyện và dinh dưỡng. Đây không phải chẩn đoán y khoa; nếu có đau ngực, chóng mặt, chấn thương, bệnh nền, rối loạn ăn uống, hoặc thay đổi cân nặng bất thường, SoCa nên khuyên người dùng gặp chuyên gia phù hợp.

## Mục tiêu vận hành

- Giúp người dùng nhớ việc cần làm trong ngày.
- Nhắc khung giờ tập luyện nhất quán.
- Theo dõi checklist tập theo tuần/tháng.
- Theo dõi cân ký theo xu hướng, không ám ảnh từng con số.
- Gợi ý bữa ăn đơn giản, đủ đạm, rau, tinh bột và chất béo tốt.
- Hỏi lại khi thiếu dữ liệu cá nhân thay vì bịa chỉ tiêu.

## Cấu trúc knowledge liên quan

- [[coach/todo-system|Todo System]]
- [[coach/daily-checkin|Daily Check-in]]
- [[coach/training-time-blocks|Khung Giờ Tập Hằng Ngày]]
- [[coach/training-week-template|Training Week Template]]
- [[coach/monthly-training-checklist|Checklist Tập Luyện Theo Tháng]]
- [[coach/weigh-in-checkin|Cân Ký Và Check-in]]
- [[coach/nutrition-coach-principles|Nguyên Tắc Dinh Dưỡng Coach]]
- [[coach/safety-boundaries|Ranh Giới An Toàn]]

## Câu hỏi SoCa nên hỏi khi bắt đầu

- Mục tiêu hiện tại là tăng cơ, giảm mỡ, giữ cân, tăng sức bền hay xây thói quen?
- Tuần này có thể tập mấy buổi, mỗi buổi bao nhiêu phút?
- Có đau/chấn thương/bệnh nền nào cần tránh không?
- Có dụng cụ nào: gym, tạ đơn, dây kháng lực, xà, hay chỉ bodyweight?
- Cân ký muốn theo dõi hằng ngày hay 1-2 lần mỗi tuần?

## Quy tắc trả lời

- Ưu tiên kế hoạch nhỏ, rõ, làm được ngay.
- Không ép lịch quá nặng nếu người dùng chưa có dữ liệu phục hồi.
- Không đưa mục tiêu calo/macro cá nhân nếu chưa có cân nặng, mục tiêu, lịch tập và bối cảnh sức khỏe.
- Khi dùng kiến thức từ vault, trích nguồn [K1], [K2] nếu runtime yêu cầu.
""",
    "wiki/coach/todo-system.md": """# Todo Hôm Nay Và Checklist Sức Khỏe

#coach #todo #planning #daily #checklist #tap-luyen #dinh-duong

Todo của SoCa nên giúp người dùng giảm mơ hồ trong ngày, không biến thành danh sách quá dài. Mỗi ngày chỉ nên có 3 nhóm: việc chính, sức khỏe, và check-in.

## Daily Todo Format

```text
Ngày:
Mức năng lượng: thấp / vừa / tốt

Top 3 việc chính:
- [ ] Việc 1
- [ ] Việc 2
- [ ] Việc 3

Sức khỏe:
- [ ] Tập luyện hoặc đi bộ
- [ ] Ăn đủ một nguồn đạm trong 2-3 bữa
- [ ] Có rau/trái cây trong ngày
- [ ] Uống nước đều
- [ ] Ngủ đúng giờ mục tiêu

Check-in:
- [ ] Cân ký hoặc ghi weekly average nếu đến lịch
- [ ] Ghi cảm giác cơ thể sau tập
- [ ] Chốt việc còn lại cho ngày mai
```

## Quy tắc ưu tiên

- Nếu người dùng quá tải: chọn 1 việc chính + 1 việc sức khỏe.
- Nếu bỏ lỡ buổi tập: không phạt, chuyển sang phiên rút gọn 10-20 phút.
- Nếu ngày bận: ưu tiên đi bộ, mobility, hoặc bài full-body ngắn.
- Nếu thiếu ngủ nhiều: giảm cường độ tập, ưu tiên kỹ thuật và phục hồi.

## Các thao tác SoCa có thể hỗ trợ

- Lập checklist ngày theo 3 nhóm: việc chính, sức khỏe, check-in.
- Rút gọn ngày quá tải thành 1 việc chính và 1 việc sức khỏe.
- Chuyển buổi tập đầy đủ thành phiên 10-20 phút khi lịch bận.
- Nhắc phần còn thiếu vào cuối ngày mà không phán xét.
""",
    "wiki/coach/daily-checkin.md": """# Daily Check-in

#coach #checkin #daily #habit

Daily check-in là cuộc kiểm tra ngắn để SoCa biết hôm nay nên đẩy, giữ nhịp hay giảm tải. Không cần dài; mục tiêu là duy trì dữ liệu và thói quen.

## Morning Check-in

```text
Sáng nay:
- Ngủ: ___ giờ
- Năng lượng: 1-5
- Cân nặng nếu có: ___ kg
- Đau/mỏi bất thường: có / không
- Khung giờ có thể tập: ___
- Việc quan trọng nhất hôm nay: ___
```

## Pre-workout Check-in

```text
Trước tập:
- Thời gian có thật: ___ phút
- Mức năng lượng: thấp / vừa / tốt
- Cơ đang đau: ___
- Mục tiêu buổi này: kỹ thuật / volume / cardio / phục hồi
```

## Evening Check-in

```text
Tối nay:
- Đã tập: có / không
- Điểm ăn uống: 1-5
- Đủ đạm: có / chưa rõ / không
- Rau/trái cây: có / không
- Điều làm tốt hôm nay: ___
- Một chỉnh nhỏ cho ngày mai: ___
```

## Cách SoCa phản hồi

- Nếu đủ dữ liệu: đề xuất hành động tiếp theo.
- Nếu thiếu dữ liệu: hỏi tối đa 1-2 câu quan trọng.
- Nếu người dùng thất bại một ngày: phản hồi trung tính, quay lại kế hoạch tối thiểu.
""",
    "wiki/coach/training-time-blocks.md": """# Khung Giờ Tập Hằng Ngày

#coach #training #time-block #habit

Khung giờ tập nên được xem như lịch hẹn với bản thân. Với người bận, tính nhất quán quan trọng hơn việc chọn giờ hoàn hảo.

## Ba khung giờ phổ biến

### Sáng

Phù hợp nếu người dùng muốn hoàn thành sớm và ít bị việc khác chen ngang.

Checklist:

- [ ] Chuẩn bị đồ tập từ tối hôm trước.
- [ ] Khởi động kỹ vì cơ thể còn cứng.
- [ ] Buổi tập có thể ngắn hơn nhưng đều.

### Chiều hoặc sau giờ học/làm

Phù hợp nếu năng lượng tốt hơn sau khi ăn trong ngày.

Checklist:

- [ ] Đặt giờ bắt đầu cụ thể.
- [ ] Ăn nhẹ trước tập nếu đói.
- [ ] Không kéo dài quá muộn làm ảnh hưởng giấc ngủ.

### Tối

Phù hợp khi ban ngày bận, nhưng nên giảm kích thích sát giờ ngủ.

Checklist:

- [ ] Ưu tiên kỹ thuật, mobility, đi bộ hoặc buổi nhẹ nếu đã muộn.
- [ ] Tránh caffeine muộn.
- [ ] Chừa thời gian hạ nhịp trước khi ngủ.

## Quy tắc chọn giờ

- Chọn giờ có xác suất làm được cao nhất, không chọn giờ "lý tưởng" nhưng hay bỏ.
- Nếu bỏ lỡ khung chính, dùng khung dự phòng 10-20 phút.
- Một ngày chỉ cần hoàn thành phiên tối thiểu vẫn được tính là giữ streak.
""",
    "wiki/coach/training-week-template.md": """# Training Week Template

#coach #training #weekly #strength #cardio

Người trưởng thành thường nên có cả vận động aerobic và bài tăng cường cơ. WHO/CDC khuyến nghị tối thiểu khoảng 150 phút vận động aerobic mức vừa mỗi tuần và ít nhất 2 ngày tập tăng cường cơ cho các nhóm cơ chính. Với người mới hoặc đang quay lại, SoCa nên bắt đầu thấp hơn rồi tăng dần.

## Template 3 buổi mỗi tuần

Phù hợp khi bận hoặc mới quay lại.

```text
Thứ 2: Full-body strength A
Thứ 3: Đi bộ nhanh / cardio nhẹ 20-30 phút
Thứ 4: Nghỉ hoặc mobility
Thứ 5: Full-body strength B
Thứ 6: Đi bộ nhanh / cardio nhẹ 20-30 phút
Thứ 7: Full-body strength C hoặc hoạt động ngoài trời
Chủ nhật: Check-in tuần + nghỉ
```

## Template 4 buổi mỗi tuần

Phù hợp khi đã có nhịp tập ổn.

```text
Thứ 2: Upper body
Thứ 3: Lower body
Thứ 4: Cardio nhẹ / mobility
Thứ 5: Upper body hoặc full-body nhẹ
Thứ 6: Lower body hoặc conditioning
Thứ 7: Đi bộ dài / thể thao nhẹ
Chủ nhật: Check-in tuần + nghỉ
```

## Cấu trúc một buổi strength

```text
Khởi động: 5-10 phút
Main lift / compound: 2-4 sets
Phụ trợ: 2-4 bài, mỗi bài 2-3 sets
Core hoặc mobility: 5-10 phút
Ghi log: bài, set, rep, mức khó RPE/RIR
```

## Tăng tiến an toàn

- Ưu tiên form trước khi tăng tải.
- Khi đạt đầu trên của rep range ở hầu hết set và form vẫn ổn, tăng nhẹ tải hoặc tăng rep.
- Không cần tập tới failure mọi set; nên chừa 1-3 reps dự phòng ở bài compound nếu chưa có kinh nghiệm.
- Nếu đau nhói, chóng mặt hoặc đau bất thường: dừng buổi tập và đánh giá lại.
""",
    "wiki/coach/monthly-training-checklist.md": """# Checklist Tập Luyện Theo Tháng

#coach #monthly #training-log #checklist #tap-luyen

Checklist tháng giúp SoCa nhìn xu hướng thay vì chỉ phản ứng theo từng ngày. Người dùng có thể copy bảng này mỗi tháng.

## Mục tiêu tháng

```text
Tháng:
Mục tiêu chính:
- [ ] Giữ lịch tập tối thiểu ___ buổi/tuần
- [ ] Đi bộ/cardio ___ phút/tuần
- [ ] Cân ký/check-in theo lịch
- [ ] Cải thiện 1 bài tập chính: ___
```

## Checklist tuần

| Tuần | Strength | Cardio/đi bộ | Mobility | Cân ký | Check-in tuần | Ghi chú |
|---|---:|---:|---:|---|---|---|
| Tuần 1 | __ buổi | __ phút | __ buổi | __ lần | [ ] | |
| Tuần 2 | __ buổi | __ phút | __ buổi | __ lần | [ ] | |
| Tuần 3 | __ buổi | __ phút | __ buổi | __ lần | [ ] | |
| Tuần 4 | __ buổi | __ phút | __ buổi | __ lần | [ ] | |
| Tuần 5 nếu có | __ buổi | __ phút | __ buổi | __ lần | [ ] | |

## Review cuối tháng

```text
Điều làm tốt:
-

Điều làm khó giữ nhịp:
-

Số buổi tập thực tế:
- Strength:
- Cardio:
- Mobility:

Cân nặng xu hướng:
- Đầu tháng:
- Cuối tháng:
- Weekly average thay đổi:

Điều chỉnh tháng sau:
-
```

## Quy tắc đánh giá

- Đánh giá theo tỷ lệ hoàn thành, không theo cảm xúc một ngày.
- Nếu đạt 70-80% kế hoạch mà vẫn phục hồi tốt, đó là tháng ổn.
- Nếu liên tục bỏ buổi, giảm kế hoạch tối thiểu trước khi tăng tham vọng.
""",
    "wiki/coach/weigh-in-checkin.md": """# Cân Ký Và Check-in

#coach #weigh-in #bodyweight #tracking

Cân nặng dao động theo nước, muối, glycogen, giấc ngủ, stress và thời điểm cân. Vì vậy SoCa nên ưu tiên xu hướng tuần/tháng, không phản ứng mạnh với từng ngày.

## Cách cân nhất quán

- Dùng cùng một cân.
- Cân cùng thời điểm, tốt nhất là buổi sáng sau khi đi vệ sinh và trước khi ăn/uống.
- Ghi điều kiện đặc biệt: ăn mặn, ngủ ít, tập chân nặng, stress, đi xa.
- Nếu cân hằng ngày gây căng thẳng hoặc ám ảnh, chuyển sang 1-2 lần mỗi tuần.

## Daily weigh-in format

```text
Ngày:
Cân nặng: ___ kg
Ngủ: ___ giờ
Ăn mặn hôm qua: có / không
Tập nặng hôm qua: có / không
Ghi chú:
```

## Weekly check-in format

```text
Tuần:
Số lần cân:
Weekly average:
So với tuần trước:
Vòng eo nếu đo:
Ảnh tiến độ nếu có:
Nhận xét:
```

## Cách SoCa diễn giải

- Nếu tăng/giảm trong 1 ngày: nhắc rằng đó có thể là nước và biến động bình thường.
- Nếu xu hướng 2-4 tuần đi ngược mục tiêu: đề xuất chỉnh nhẹ ăn uống, vận động hoặc giấc ngủ.
- Nếu giảm cân quá nhanh hoặc mệt mỏi kéo dài: khuyên giảm tốc độ và cân nhắc gặp chuyên gia.

## Ngưỡng an toàn

CDC nhấn mạnh giảm cân bền vững thường là nhịp chậm và ổn định. Không nên đặt mục tiêu giảm cực nhanh chỉ vì một tuần cân nặng đứng yên.
""",
    "wiki/coach/nutrition-coach-principles.md": """# Nguyên Tắc Dinh Dưỡng Và Protein Cho Người Tập Luyện

#coach #nutrition #protein #meal-planning #tap-luyen #dinh-duong-nguoi-tap

SoCa nên đưa gợi ý dinh dưỡng theo nguyên tắc đơn giản, có thể làm được, và không biến thành đơn thuốc. Nếu người dùng có bệnh nền, thuốc đang dùng, tiền sử rối loạn ăn uống, hoặc mục tiêu thi đấu, nên khuyên hỏi chuyên gia.

## Khung bữa ăn thực dụng

Một bữa chính nên cố gắng có:

- một nguồn đạm;
- một phần rau/củ/trái cây;
- một phần tinh bột phù hợp mức vận động;
- một lượng chất béo tốt vừa phải;
- nước hoặc đồ uống ít đường.

## Đạm cho người tập luyện

ISSN nêu rằng nhiều người tập luyện có thể cần khoảng 1.4-2.0 g protein/kg/ngày để hỗ trợ thích nghi tập luyện, tùy mục tiêu và bối cảnh. Đây không phải chỉ tiêu mặc định cho mọi người. SoCa nên hỏi cân nặng, mục tiêu, bệnh nền và khẩu phần hiện tại trước khi tính số cụ thể.

## Timing đơn giản

- Không cần ám ảnh "cửa sổ vàng".
- Nếu tiện, chia đạm đều trong ngày.
- Sau tập có thể ăn một bữa có đạm và tinh bột, nhất là khi buổi sau gần hoặc tập nặng.
- Nếu tập lúc đói làm giảm hiệu suất, dùng bữa nhẹ trước tập.

## Mẫu trả lời thực dụng

"Bạn ưu tiên mỗi bữa có một nguồn đạm như trứng, cá, thịt nạc, đậu phụ hoặc sữa chua; thêm rau/trái cây và tinh bột vừa đủ theo mức vận động. Nếu hôm nay có tập nặng, bữa sau tập nên có cả đạm và tinh bột."

## Không nên làm

- Không tự kê calo quá thấp.
- Không khuyên cắt toàn bộ tinh bột nếu không có lý do.
- Không khuyên supplement như bắt buộc.
- Không phán xét cơ thể hoặc cân nặng.
""",
    "wiki/coach/safety-boundaries.md": """# Ranh Giới An Toàn

#coach #safety #boundaries

SoCa là trợ lý hỗ trợ thói quen, không thay thế bác sĩ, huấn luyện viên cá nhân hoặc chuyên gia dinh dưỡng lâm sàng.

## Khi cần khuyên gặp chuyên gia

- Đau ngực, khó thở bất thường, ngất, chóng mặt mạnh khi tập.
- Đau nhói, sưng, mất chức năng vận động hoặc chấn thương kéo dài.
- Có bệnh tim mạch, thận, tiểu đường, gout, rối loạn ăn uống, đang dùng thuốc ảnh hưởng cân nặng/chuyển hóa.
- Giảm cân quá nhanh, mệt mỏi kéo dài, mất ngủ nặng, ám ảnh cân nặng.
- Muốn chế độ ăn rất cực đoan hoặc cắt nhóm thực phẩm lớn.

## Cách phản hồi an toàn

- Công nhận mục tiêu của người dùng.
- Nói rõ giới hạn.
- Gợi ý bước an toàn nhỏ: giảm cường độ, nghỉ, theo dõi triệu chứng, hỏi chuyên gia.
- Không đưa chẩn đoán.

## Ví dụ

"Mình chưa thể đánh giá nguyên nhân đau này qua chat. Nếu đau nhói, sưng, tê hoặc ảnh hưởng vận động, bạn nên dừng bài đó và gặp chuyên gia y tế/huấn luyện viên để kiểm tra. Hôm nay mình có thể giúp bạn đổi sang buổi phục hồi nhẹ."
""",
    "wiki/sources/coach-fitness-sources.md": """# Nguồn Coach Fitness

#sources #coach #fitness #nutrition

Các note coach trong vault này được viết theo hướng thực dụng, an toàn, và có kiểm soát. Chúng không thay thế tư vấn y tế cá nhân.

## Nguồn chính

- WHO, Physical activity fact sheet: khuyến nghị người trưởng thành vận động aerobic 150-300 phút mức vừa mỗi tuần hoặc tương đương, cộng với hoạt động tăng cường cơ ít nhất 2 ngày/tuần.
- CDC, Adult physical activity guidelines: người trưởng thành cần tối thiểu 150 phút vận động mức vừa mỗi tuần và 2 ngày tăng cường cơ.
- CDC, Steps for Losing Weight: giảm cân bền vững thường theo nhịp chậm và ổn định, khoảng 1-2 pounds mỗi tuần.
- American Heart Association, weigh-in discussion: nếu cân thường xuyên, nên dùng cùng cân và cùng thời điểm; nếu cân hằng ngày gây ảnh hưởng tâm lý, có thể giảm tần suất.
- Cleveland Clinic, best time to weigh yourself: cân buổi sáng sau khi đi vệ sinh và trước khi ăn/uống để nhất quán.
- International Society of Sports Nutrition, protein and exercise position stand: người tập luyện thường có thể cần khoảng 1.4-2.0 g protein/kg/ngày; mỗi liều protein thực tế thường khoảng 20-40 g tùy cơ thể và bữa ăn.
- ACSM resistance training position stand / public summaries: tập sức mạnh nên tăng tiến dần, ưu tiên kỹ thuật, nhóm cơ chính và lịch phù hợp khả năng phục hồi.

## Cách dùng nguồn

- Dùng các con số như khung tham khảo, không áp cứng nếu thiếu dữ liệu cá nhân.
- Trước khi đưa chỉ tiêu cụ thể, SoCa nên kiểm tra mục tiêu, cân nặng, lịch tập, bệnh nền và mức kinh nghiệm.
- Khi có dấu hiệu nguy cơ, ưu tiên ranh giới an toàn trong [[coach/safety-boundaries|Ranh Giới An Toàn]].
""",
}


@dataclass(frozen=True)
class SeedResult:
    root: Path
    created: tuple[Path, ...]
    updated: tuple[Path, ...]
    skipped: tuple[Path, ...]


def seed_coach_knowledge(root: str | Path, *, force: bool = False) -> SeedResult:
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []
    updated: list[Path] = []
    skipped: list[Path] = []

    for relative_path, content in COACH_FILES.items():
        path = root_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            skipped.append(path)
            continue

        existed = path.exists()
        path.write_text(content, encoding="utf-8")
        if existed:
            updated.append(path)
        else:
            created.append(path)

    index_path = root_path / "wiki" / "index.md"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_existed = index_path.exists()
    if index_existed:
        index_text = index_path.read_text(encoding="utf-8")
    else:
        index_text = "# Index\n"

    if "## Coach Tập Luyện Và Dinh Dưỡng" not in index_text:
        index_text = index_text.rstrip() + "\n\n" + COACH_INDEX_BLOCK + "\n"
        index_path.write_text(index_text, encoding="utf-8")
        if index_existed:
            updated.append(index_path)
        else:
            created.append(index_path)
    elif force:
        before, _, after = index_text.partition("## Coach Tập Luyện Và Dinh Dưỡng")
        next_heading = after.find("\n## ")
        if next_heading >= 0:
            index_text = before.rstrip() + "\n\n" + COACH_INDEX_BLOCK + after[next_heading:] + "\n"
        else:
            index_text = before.rstrip() + "\n\n" + COACH_INDEX_BLOCK + "\n"
        index_path.write_text(index_text, encoding="utf-8")
        updated.append(index_path)

    return SeedResult(
        root=root_path,
        created=tuple(created),
        updated=tuple(dict.fromkeys(updated)),
        skipped=tuple(skipped),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed coach knowledge pages into a local vault.")
    parser.add_argument(
        "--vault",
        type=Path,
        default=Path.home() / "KnowledgeVault",
        help="Knowledge vault root.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing coach pages.")
    return parser


def print_result(result: SeedResult) -> None:
    table = Table(title="Coach Knowledge Seed")
    table.add_column("Item")
    table.add_column("Count", justify="right")
    table.add_row("Created", str(len(result.created)))
    table.add_row("Updated", str(len(result.updated)))
    table.add_row("Skipped", str(len(result.skipped)))
    console.print(table)
    console.print(f"[green]Vault:[/green] {result.root}")

    if result.created:
        console.print("\n[bold]Created files[/bold]")
        for path in result.created:
            console.print(f"- {path.relative_to(result.root)}")

    if result.updated:
        console.print("\n[bold]Updated files[/bold]")
        for path in result.updated:
            console.print(f"- {path.relative_to(result.root)}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = seed_coach_knowledge(args.vault, force=args.force)
    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
