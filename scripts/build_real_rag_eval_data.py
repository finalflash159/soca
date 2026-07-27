"""Build reproducible Vietnamese retrieval fixtures from attributed sources.

The generated fixture combines the CC BY-SA XQuAD Vietnamese split with a
project-vault slice copied from this repository's checked-in documentation.
Every generated file has a source manifest entry and a content digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from urllib.request import urlopen

REPO_ROOT = Path(__file__).resolve().parents[1]
XQUAD_URL = "https://raw.githubusercontent.com/google-deepmind/xquad/master/xquad.vi.json"
XQUAD_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"
PROJECT_LICENSE_URL = "https://github.com/finalflash159/shrike-7/blob/main/LICENSE"
PROJECT_DOCS = (
    ("docs/05-assistant-runtime.md", "assistant-runtime.md"),
    ("docs/07-tui.md", "tui.md"),
    ("BENCHMARKS.md", "benchmarks.md"),
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_name(title: str) -> str:
    value = "".join(char if char.isalnum() else "_" for char in title).strip("_")
    return value[:96] or "untitled"


def build(output_root: Path, *, refresh_download: bool = True) -> None:
    fixture_root = output_root / "real_rag_vault"
    wiki_root = fixture_root / "wiki"
    project_root = wiki_root / "project"
    xquad_root = wiki_root / "xquad_vi"
    prompt_path = output_root.parent / "prompts" / "real_rag_vi.jsonl"
    manifest_path = fixture_root / "SOURCE_MANIFEST.json"

    if fixture_root.exists():
        shutil.rmtree(fixture_root)
    wiki_root.mkdir(parents=True)
    project_root.mkdir()
    xquad_root.mkdir()
    prompt_path.parent.mkdir(parents=True, exist_ok=True)

    if refresh_download:
        with urlopen(XQUAD_URL, timeout=60) as response:
            xquad_bytes = response.read()
    else:
        raise ValueError("refresh_download=False requires a checked-in source cache")
    xquad = json.loads(xquad_bytes.decode("utf-8"))
    if not isinstance(xquad, dict) or not isinstance(xquad.get("data"), list):
        raise ValueError("unexpected XQuAD payload")

    manifest: list[dict[str, str]] = []
    cases: list[dict[str, object]] = []
    for article in xquad["data"]:
        title = article.get("title")
        paragraphs = article.get("paragraphs")
        if not isinstance(title, str) or not isinstance(paragraphs, list):
            raise ValueError("invalid XQuAD article")
        article_path = xquad_root / f"{_safe_name(title)}.md"
        article_lines = [f"# {title}", ""]
        for paragraph in paragraphs:
            context = paragraph.get("context") if isinstance(paragraph, dict) else None
            if not isinstance(context, str) or not context.strip():
                continue
            article_lines.extend([context.strip(), ""])
        article_bytes = "\n".join(article_lines).encode("utf-8")
        article_path.write_bytes(article_bytes)
        manifest.append(
            {
                "path": article_path.relative_to(fixture_root).as_posix(),
                "source": f"{XQUAD_URL}#{title}",
                "license": "CC BY-SA 4.0",
                "license_url": XQUAD_LICENSE_URL,
                "sha256": _digest(article_bytes),
            }
        )
        for paragraph in paragraphs:
            if not isinstance(paragraph, dict):
                continue
            qas = paragraph.get("qas", [])
            if not isinstance(qas, list):
                continue
            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                question = qa.get("question")
                case_id = qa.get("id")
                if not isinstance(question, str) or not question.strip() or not isinstance(case_id, str):
                    continue
                cases.append(
                    {
                        "id": f"xquad-{case_id}",
                        "slice": "learning_notes",
                        "query": question.strip(),
                        "relevant_paths": [
                            article_path.relative_to(fixture_root).as_posix()
                        ],
                        "source": "XQuAD Vietnamese",
                        "answer": str((qa.get("answers") or [{}])[0].get("text", "")),
                    }
                )

    for relative_source, target_name in PROJECT_DOCS:
        source_path = REPO_ROOT / relative_source
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        target_path = project_root / target_name
        shutil.copyfile(source_path, target_path)
        data = target_path.read_bytes()
        relative_target = target_path.relative_to(fixture_root).as_posix()
        manifest.append(
            {
                "path": relative_target,
                "source": f"repository:{relative_source}",
                "license": "MIT",
                "license_url": PROJECT_LICENSE_URL,
                "sha256": _digest(data),
            }
        )

    project_cases = (
        ("project-runtime-001", "Tôi đã thiết kế runtime text và voice đi qua lớp nào?", "wiki/project/assistant-runtime.md"),
        ("project-ui-001", "TUI của dự án dùng giao thức nào để nói chuyện với engine?", "wiki/project/tui.md"),
        ("project-bench-001", "Các benchmark chính của dự án đo những chỉ số nào?", "wiki/project/benchmarks.md"),
    )
    for case_id, query, path in project_cases:
        cases.append(
            {
                "id": case_id,
                "slice": "life_vault_project",
                "query": query,
                "relevant_paths": [path],
                "source": "checked-in project documentation",
            }
        )

    with prompt_path.open("w", encoding="utf-8") as handle:
        for case in cases:
            handle.write(json.dumps(case, ensure_ascii=False, sort_keys=True) + "\n")

    manifest_payload = {
        "schema_version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "xquad_source_sha256": _digest(xquad_bytes),
        "case_count": len(cases),
        "sources": manifest,
        "qrels": prompt_path.relative_to(REPO_ROOT).as_posix(),
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (fixture_root / "README.md").write_text(
        "# Real Vietnamese retrieval fixture\n\n"
        "This fixture is reproducibly built by `scripts/build_real_rag_eval_data.py`. "
        "The `learning_notes` slice is XQuAD Vietnamese (CC BY-SA 4.0). The "
        "`life_vault_project` slice uses checked-in project documentation as a "
        "traceable project-vault substitute; no private personal data is included.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "eval" / "fixtures",
    )
    args = parser.parse_args()
    build(args.output_root)


if __name__ == "__main__":
    main()
