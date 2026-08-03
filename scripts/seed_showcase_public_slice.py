from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = REPO_ROOT / "eval" / "fixtures" / "real_rag_vault"
DEFAULT_TARGET = REPO_ROOT / "eval" / "fixtures" / "knowledge_vault"
XQUAD_SOURCE_URL = "https://github.com/google-deepmind/xquad/blob/master/xquad.vi.json"
XQUAD_LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def seed(source_root: Path, target_root: Path) -> int:
    source_root = source_root.resolve()
    target_root = target_root.resolve()
    source_manifest = json.loads(
        (source_root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    sources = [
        item
        for item in source_manifest.get("sources", [])
        if isinstance(item, dict)
        and str(item.get("path", "")).startswith("wiki/xquad_vi/")
    ]
    if not sources:
        raise ValueError("source fixture has no XQuAD Vietnamese slice")

    target_slice = target_root / "wiki" / "xquad_vi"
    target_slice.mkdir(parents=True, exist_ok=True)
    copied: list[dict[str, str]] = []
    for item in sorted(sources, key=lambda row: str(row["path"])):
        relative = Path(str(item["path"]))
        source_path = source_root / relative
        target_path = target_root / relative
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if _sha256(source_path) != item.get("sha256"):
            raise ValueError(f"source digest mismatch: {relative}")
        shutil.copyfile(source_path, target_path)
        copied.append(
            {
                "path": relative.as_posix(),
                "source": str(item.get("source", "")),
                "license": str(item.get("license", "CC BY-SA 4.0")),
                "license_url": str(item.get("license_url", XQUAD_LICENSE_URL)),
                "sha256": _sha256(target_path),
            }
        )

    manifest = {
        "schema_version": 1,
        "dataset": "XQuAD Vietnamese",
        "dataset_source": XQUAD_SOURCE_URL,
        "license": "CC BY-SA 4.0",
        "license_url": XQUAD_LICENSE_URL,
        "source_fixture": str(source_root.relative_to(REPO_ROOT)),
        "source_sha256": source_manifest.get("xquad_source_sha256"),
        "document_count": len(copied),
        "sources": copied,
    }
    (target_root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (target_root / "wiki" / "sources" / "xquad-vietnamese.md").write_text(
        "---\n"
        "type: source_note\n"
        "scope: public-reference\n"
        "status: attributed\n"
        "dataset: XQuAD Vietnamese\n"
        "license: CC BY-SA 4.0\n"
        "---\n\n"
        "# XQuAD Vietnamese reference slice\n\n"
        "Đây là 48 bài đọc tiếng Việt lấy từ XQuAD Vietnamese để vault demo có "
        "nội dung tham khảo thật, đa chủ đề và đủ dài cho việc tìm kiếm theo "
        "ngữ nghĩa. Đây không phải dữ liệu cá nhân và không thay thế release "
        "benchmark.\n\n"
        f"- Dataset: [{XQUAD_SOURCE_URL}]({XQUAD_SOURCE_URL})\n"
        f"- License: [CC BY-SA 4.0]({XQUAD_LICENSE_URL})\n"
        "- Digest và từng file: `SOURCE_MANIFEST.json`\n"
        "- Qrels/release benchmark vẫn nằm trong `eval/fixtures/real_rag_vault`; "
        "không dùng slice này để tự chấm chất lượng model.\n",
        encoding="utf-8",
    )
    return len(copied)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy the attributed public XQuAD slice into the showcase vault."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    args = parser.parse_args()
    count = seed(args.source, args.target)
    print(f"Copied {count} attributed public documents into {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
