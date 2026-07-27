"""Seed a small, traceable SoCa knowledge vault for demos and retrieval evals.

The demo corpus is intentionally separate from ``eval/fixtures/knowledge_vault``.
That older vault is a tiny regression fixture for unit tests; this corpus is the
assistant-like demo surface with two explicit slices:

* ``learning_notes`` — study and engineering notes;
* ``life_vault`` — project decisions, a clearly synthetic budget ledger, and
  safety boundaries for health-adjacent questions.

No personal vault is touched unless the caller passes ``--vault`` explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE_ROOT = REPO_ROOT / "eval" / "fixtures" / "knowledge_demo_vault"
DEFAULT_QRELS_PATH = REPO_ROOT / "eval" / "prompts" / "knowledge_demo_vi.jsonl"
console = Console()


DEMO_FILES: Mapping[str, str] = {
    "wiki/learning/notes/bayes-theorem.md": """# Ghi chú học tập — Định lý Bayes

#learning #notes #bayes #probability

Ngày ghi: 2026-07-18
Slice: learning_notes
Provenance: authored_demo_note — ghi chú học tập minh họa, không phải dữ liệu cá nhân.

## Ý chính

Định lý Bayes cập nhật xác suất của một giả thuyết khi có thêm bằng chứng. Cách
viết thường dùng là:

```text
P(A | B) = P(B | A) * P(A) / P(B)
```

- `P(A)` là prior: xác suất của A trước khi quan sát B.
- `P(B | A)` là likelihood: khả năng thấy B nếu A đúng.
- `P(B)` là evidence: xác suất quan sát B trong toàn bộ các trường hợp.
- `P(A | B)` là posterior: xác suất cập nhật của A sau khi biết B.

## Ví dụ tự kiểm tra

Nếu một bài test có tỷ lệ dương tính giả đáng kể, không thể suy ra ngay rằng
người có kết quả dương tính chắc chắn mắc bệnh. Cần dùng cả tỷ lệ mắc ban đầu
`P(A)` và độ chính xác có điều kiện `P(B | A)`. Đây là điểm dễ nhầm giữa
`P(A | B)` và `P(B | A)`.

## Câu nhắc khi trả lời

Khi giải thích cho người khác, hãy nêu rõ prior, likelihood và evidence trước
khi thay số. Nếu đề bài thiếu một trong các đại lượng, không tự bịa giá trị.
""",
    "wiki/learning/notes/onnx-runtime.md": """# Ghi chú học tập — ONNX Runtime trong SoCa

#learning #notes #onnx #onnx-runtime #machine-learning

Ngày ghi: 2026-07-21
Slice: learning_notes
Provenance: repository_fact — tóm tắt từ `BENCHMARKS.md` và `docs/09-hybrid-rag-memory.md`.

## Điều cần nhớ

ONNX Runtime là lớp chạy model ONNX trong pipeline local của SoCa. Trên máy
Apple, benchmark của dự án kiểm tra `CoreMLExecutionProvider` và
`CPUExecutionProvider`; CPU là fallback khi một node không chạy được trên
CoreML.

Trong voice pipeline, PhoWhisper dùng các graph ONNX encoder/decoder cho ASR.
Ở knowledge layer, dense retriever cũng có thể dùng embedding model ONNX;
retrieval sẽ quay về sparse-only nếu dense backend không khả dụng.

## Liên hệ với RAG

ONNX Runtime không tự quyết định note nào được đưa vào câu trả lời. Nó chỉ
chạy phần embedding/dense retrieval. Sau đó SoCa hợp nhất sparse và dense bằng
RRF, rồi mới tạo context có citation cho LLM.

## Nguồn trong repo

- `BENCHMARKS.md`: provider CoreML/CPU và benchmark ONNX của ASR.
- `docs/09-hybrid-rag-memory.md`: dense retriever, fallback và RRF.
""",
    "wiki/life/project/tts-decision.md": """# Quyết định dự án — Chọn TTS Valtec ONNX

#life-vault #project #decision #tts #valtec #onnx

Ngày quyết định: 2026-06-01
Slice: life_vault
Provenance: repository_fact — decision record rút từ phần D3.0 trong `BENCHMARKS.md`.

## Quyết định

SoCa chọn Valtec ONNX làm TTS baseline hiện tại, với voice `NF` trong profile
`baseline`.

## Vì sao

1. Đây là runtime tiếng Việt local đã được tích hợp vào đường chạy sản phẩm.
2. Cutover hiện tại dùng bốn ONNX graph fp32 và không cần gửi transcript lên
   cloud.
3. Valtec có số đo E2E đã được ghi trong benchmark, nên phù hợp làm baseline
   để so sánh các ứng viên khác.

## Phạm vi của quyết định

Đây là lựa chọn baseline cho demo/runtime, không phải tuyên bố Valtec luôn có
chất lượng cao nhất. Các ứng viên VieNeu, Piper và runtime khác vẫn có thể được
đánh giá trong bake-off riêng.

## Nguồn

- `BENCHMARKS.md#D3.0 — Valtec ONNX release (current, cutover complete)`
- `soca/tts/registry.py`
""",
    "wiki/life/project/rag-architecture.md": """# Nhật ký dự án — Hybrid RAG và memory

#life-vault #project #rag #memory #architecture

Ngày ghi: 2026-07-22
Slice: life_vault
Provenance: repository_fact — tóm tắt từ `docs/09-hybrid-rag-memory.md`.

## Luồng hiện tại

1. Markdown vault được lập index theo document/chunk.
2. Sparse retriever tìm theo lexical/BM25.
3. Dense retriever dùng embedding local khi backend khả dụng.
4. Reciprocal Rank Fusion (RRF) hợp nhất hai danh sách xếp hạng.
5. Context builder đóng gói các hit và line citation cho LLM.
6. Guardrail kiểm tra tool, đường dẫn và citation trước khi trả lời.

Memory là client của cùng retriever nhưng khác namespace với knowledge. Profile
được truy hồi theo query với relevance, recency và importance; working memory
được nén ngoài hot path. Episodic memory chỉ được ghi khi có consent và qua
proposal/approval.

## Điều không được làm

- Không biến một keyword thành bằng chứng để tự động truy hồi mọi câu hỏi.
- Không cho LLM đọc path ngoài phạm vi vault.
- Không trả lời như fact nếu không có hit/citation phù hợp.
""",
    "wiki/life/finance/food-budget-2026-07.md": """# Sổ chi tiêu demo — Ngân sách ăn uống tháng 07/2026

#life-vault #finance #budget #food #demo

Kỳ: 2026-07
Slice: life_vault
Provenance: synthetic_demo — số liệu giả lập để trình diễn truy hồi; không phải
sổ tài chính cá nhân và không được dùng để suy ra chi tiêu thật.

## Ngân sách

- Ngân sách đặt trước: **2.500.000 VND**
- Đã ghi nhận: **1.390.000 VND**
- Còn lại theo sổ demo: **1.110.000 VND**

## Giao dịch đã ghi nhận

| Ngày | Nhóm | Số tiền (VND) |
| --- | --- | ---: |
| 2026-07-02 | Chợ/đi chợ | 180.000 |
| 2026-07-05 | Mua thực phẩm | 350.000 |
| 2026-07-09 | Ăn trưa | 75.000 |
| 2026-07-12 | Chợ/đi chợ | 220.000 |
| 2026-07-17 | Ăn ngoài | 160.000 |
| 2026-07-22 | Mua thực phẩm | 310.000 |
| 2026-07-25 | Ăn trưa | 95.000 |

## Quy tắc demo

Khi trả lời, phải nói đây là **sổ demo** và nêu kỳ/thời điểm. Nếu người dùng
hỏi chi tiêu thật, cần yêu cầu họ cung cấp hoặc đồng bộ dữ liệu thật; không
được lấy số liệu này làm dữ liệu cá nhân.
""",
    "wiki/life/health/safety-boundaries.md": """# Ranh giới an toàn — Câu hỏi sức khỏe

#life-vault #health #safety #guardrail

Ngày ghi: 2026-07-22
Slice: life_vault
Provenance: authored_safety_note — guardrail demo, không phải hướng dẫn chẩn đoán.

## Disclaimer bắt buộc

SoCa chỉ có thể cung cấp thông tin sức khỏe chung từ note có nguồn. Nội dung
không thay thế bác sĩ, chuyên gia dinh dưỡng hoặc dịch vụ cấp cứu; không được
chẩn đoán bệnh, kê thuốc, hay suy ra tình trạng cá nhân từ một câu hỏi ngắn.

## Cách xử lý

- Nếu câu hỏi có dấu hiệu cấp cứu hoặc triệu chứng nặng, khuyên người dùng
  liên hệ dịch vụ y tế khẩn cấp tại địa phương.
- Nếu thiếu bệnh nền, thuốc đang dùng, tuổi, hoặc mục tiêu, nói rõ giới hạn và
  hỏi lại tối đa một câu cần thiết.
- Khi không có note phù hợp, nói không tìm thấy dữ liệu trong vault thay vì bịa.
- Citation của note này chỉ chứng minh ranh giới an toàn; nó không phải bằng
  chứng y khoa cho một chẩn đoán cụ thể.
""",
}


DEMO_INDEX = """# SoCa Knowledge Demo Vault

## Learning notes

- [[learning/notes/bayes-theorem|Định lý Bayes]]
- [[learning/notes/onnx-runtime|ONNX Runtime trong SoCa]]

## Life vault

- [[life/project/tts-decision|Quyết định TTS Valtec ONNX]]
- [[life/project/rag-architecture|Hybrid RAG và memory]]
- [[life/finance/food-budget-2026-07|Ngân sách ăn uống 07/2026]]
- [[life/health/safety-boundaries|Ranh giới an toàn sức khỏe]]
"""


DEMO_CASES: tuple[dict[str, object], ...] = (
    {
        "id": "learning-bayes-001",
        "slice": "learning_notes",
        "query": "Ghi chú của tôi nói định lý Bayes thế nào?",
        "relevant_paths": ["wiki/learning/notes/bayes-theorem.md"],
        "expected_contains": ["P(A | B)", "prior", "posterior"],
    },
    {
        "id": "learning-bayes-002",
        "slice": "learning_notes",
        "query": "Tôi hay nhầm P(A|B) với P(B|A), note Bayes nhắc gì?",
        "relevant_paths": ["wiki/learning/notes/bayes-theorem.md"],
        "expected_contains": ["P(A | B)", "P(B | A)"],
    },
    {
        "id": "learning-onnx-001",
        "slice": "learning_notes",
        "query": "Tuần trước tôi note gì về ONNX runtime?",
        "relevant_paths": ["wiki/learning/notes/onnx-runtime.md"],
        "expected_contains": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    },
    {
        "id": "learning-onnx-002",
        "slice": "learning_notes",
        "query": "ONNX liên quan gì đến dense retrieval và RAG?",
        "relevant_paths": ["wiki/learning/notes/onnx-runtime.md"],
        "expected_contains": ["RRF", "sparse-only"],
    },
    {
        "id": "life-tts-001",
        "slice": "life_vault",
        "query": "Tôi đã quyết chọn TTS nào, vì sao?",
        "relevant_paths": ["wiki/life/project/tts-decision.md"],
        "expected_contains": ["Valtec ONNX", "baseline", "voice `NF`"],
    },
    {
        "id": "life-rag-001",
        "slice": "life_vault",
        "query": "RAG hiện tại của dự án chạy qua những bước nào?",
        "relevant_paths": ["wiki/life/project/rag-architecture.md"],
        "expected_contains": ["Sparse retriever", "RRF", "citation"],
    },
    {
        "id": "life-budget-001",
        "slice": "life_vault",
        "query": "Ngân sách ăn uống tháng này còn bao nhiêu?",
        "relevant_paths": ["wiki/life/finance/food-budget-2026-07.md"],
        "expected_contains": ["1.110.000 VND", "sổ demo"],
    },
    {
        "id": "life-health-001",
        "slice": "life_vault",
        "query": "Nếu tôi hỏi vấn đề sức khỏe thì SoCa phải cảnh báo gì?",
        "relevant_paths": ["wiki/life/health/safety-boundaries.md"],
        "expected_contains": ["không thay thế bác sĩ", "không được chẩn đoán"],
    },
)


@dataclass(frozen=True)
class SeedResult:
    root: Path
    created: tuple[Path, ...]
    updated: tuple[Path, ...]
    skipped: tuple[Path, ...]


def _write_files(root: Path, files: Mapping[str, str], *, force: bool) -> SeedResult:
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    updated: list[Path] = []
    skipped: list[Path] = []

    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            skipped.append(path)
            continue
        existed = path.exists()
        path.write_text(content, encoding="utf-8")
        (updated if existed else created).append(path)

    index_path = root / "wiki" / "index.md"
    if not index_path.exists() or force:
        existed = index_path.exists()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(DEMO_INDEX, encoding="utf-8")
        (updated if existed else created).append(index_path)
    else:
        skipped.append(index_path)

    return SeedResult(root, tuple(created), tuple(updated), tuple(skipped))


def seed_demo_knowledge(root: str | Path, *, force: bool = False) -> SeedResult:
    """Seed only the demo-owned notes into ``root``."""

    return _write_files(Path(root), DEMO_FILES, force=force)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_eval_fixture(*, force: bool = True) -> SeedResult:
    """Build the checked-in demo vault, qrels, and source manifest."""

    result = seed_demo_knowledge(DEFAULT_FIXTURE_ROOT, force=force)
    DEFAULT_QRELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_QRELS_PATH.write_text(
        "".join(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n" for case in DEMO_CASES),
        encoding="utf-8",
    )

    sources = []
    for relative_path in sorted(DEMO_FILES):
        path = DEFAULT_FIXTURE_ROOT / relative_path
        if "/finance/" in relative_path:
            provenance = "synthetic_demo"
        elif relative_path.startswith("wiki/learning/"):
            provenance = "authored_demo_note"
        elif "/project/" in relative_path:
            provenance = "repository_fact"
        else:
            provenance = "authored_safety_note"
        sources.append(
            {
                "path": relative_path,
                "provenance": provenance,
                "sha256": _sha256(path),
            }
        )

    manifest = {
        "schema_version": 1,
        "fixture": "knowledge_demo_v1",
        "case_count": len(DEMO_CASES),
        "slices": {"learning_notes": 4, "life_vault": 4},
        "qrels": DEFAULT_QRELS_PATH.relative_to(REPO_ROOT).as_posix(),
        "sources": sources,
    }
    (DEFAULT_FIXTURE_ROOT / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (DEFAULT_FIXTURE_ROOT / "README.md").write_text(
        "# SoCa knowledge demo vault\n\n"
        "This is the canonical assistant-like demo fixture. It contains two\n"
        "explicit slices: `learning_notes` and `life_vault`.\n\n"
        "The project notes are derived from checked-in SoCa documentation. The\n"
        "finance note is explicitly synthetic demo data; it is not personal\n"
        "financial information. The health note is a safety boundary, not\n"
        "medical advice. Rebuild with `uv run python scripts/seed_demo_knowledge.py\n"
        "--fixture`.\n",
        encoding="utf-8",
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seed or build the SoCa knowledge demo vault.")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--fixture", action="store_true", help="Build the checked-in eval fixture.")
    target.add_argument("--vault", type=Path, help="Seed an explicitly chosen vault path.")
    parser.add_argument("--force", action="store_true", help="Overwrite demo-owned files.")
    return parser


def print_result(result: SeedResult, *, title: str) -> None:
    table = Table(title=title)
    table.add_column("Item")
    table.add_column("Count", justify="right")
    table.add_row("Created", str(len(result.created)))
    table.add_row("Updated", str(len(result.updated)))
    table.add_row("Skipped", str(len(result.skipped)))
    console.print(table)
    console.print(f"[green]Vault:[/green] {result.root}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixture:
        result = build_eval_fixture(force=True)
        print_result(result, title="Knowledge Demo Fixture")
    else:
        result = seed_demo_knowledge(args.vault, force=args.force)
        print_result(result, title="Knowledge Demo Seed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
