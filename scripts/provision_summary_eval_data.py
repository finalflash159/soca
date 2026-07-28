"""Provision a pinned, public real-data suite for summary-model evaluation.

The generated JSONL files are local benchmark inputs under
``eval/data/summary_public`` and are intentionally gitignored. Synthetic
SoCa-state fixtures remain separate; this script never runs in the product
runtime and never downloads user data.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import stat
import tarfile
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from datasets import load_dataset
from huggingface_hub import hf_hub_download

from scripts.download_summary_models import load_hf_token_from_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "eval" / "data" / "summary_public"


@dataclass(frozen=True)
class PublicSummaryDatasetSpec:
    key: str
    repo: str
    revision: str
    license: str
    language: str
    role: str
    paper_url: str


PUBLIC_SUMMARY_DATASETS: dict[str, PublicSummaryDatasetSpec] = {
    "vsolscsum_vi": PublicSummaryDatasetSpec(
        key="vsolscsum_vi",
        repo="https://github.com/nguyenlab/VSoLSCSum-Dataset",
        revision="f2fc07026917436ff092765baa01acce1997d145",
        license="CC-BY-4.0 per SEACrowd loader; upstream repository has no standalone LICENSE",
        language="vi",
        role="primary Vietnamese human-annotated social-context summarization",
        paper_url="https://aclanthology.org/W16-5405/",
    ),
    "seahorse_vi": PublicSummaryDatasetSpec(
        key="seahorse_vi",
        repo="tasksource/seahorse_summarization_evaluation",
        revision="c641c22879b601fb2edaa0a3f0edf3a68e5f5d23",
        license="CC (dataset card); retain source-dataset licenses",
        language="vi",
        role="Vietnamese human-rated quality/reference sanity",
        paper_url="https://aclanthology.org/2023.emnlp-main.584/",
    ),
    "wiki_lingua_vi": PublicSummaryDatasetSpec(
        key="wiki_lingua_vi",
        repo="esdurmus/wiki_lingua",
        revision="ea3db3510cbd34d0f8dc612419ae40e4732f3b40",
        license="CC-BY-3.0",
        language="vi",
        role="Vietnamese crowdsourced how-to summarization sanity",
        paper_url="https://aclanthology.org/2020.findings-emnlp.360/",
    ),
    "xlsum_vi": PublicSummaryDatasetSpec(
        key="xlsum_vi",
        repo="csebuetnlp/xlsum",
        revision="30fece425f9a3866e04321773ca7a80056d55ca6",
        license="CC-BY-NC-SA-4.0",
        language="vi",
        role="Vietnamese professionally edited news-summary sanity",
        paper_url="https://aclanthology.org/2021.findings-acl.413/",
    ),
    "dialogsum_en": PublicSummaryDatasetSpec(
        key="dialogsum_en",
        repo="knkarthick/dialogsum",
        revision="a968e7aee0602e257935f1321a02e4287f7d5848",
        license="CC-BY-NC-SA-4.0",
        language="en",
        role="secondary human-written dialogue-structure control; not Vietnamese evidence",
        paper_url="https://aclanthology.org/2021.findings-acl.449/",
    ),
}

_VSOLSCSUM_FILENAME = "VSoSLCSum.xml"
_VSOLSCSUM_SHA256 = "54c48065a1a90a3cb87dd7fe2889f91a383bdf2bbba7ca86194ff0b0bb2d1f70"
_XLSUM_FILENAME = "data/vietnamese_XLSum_v2.0.tar.bz2"


def _private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(stat.S_IRWXU)


def _make_tree_private(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"public summary data tree must not contain symlinks: {path}")
        if path.is_dir():
            path.chmod(stat.S_IRWXU)
        elif path.is_file():
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    root.chmod(stat.S_IRWXU)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    count = 0
    with path.open("w", encoding="utf-8") as destination:
        for row in rows:
            destination.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return count, _sha256(path)


def _stable_sample(rows: Iterable[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    """Select the lowest stable hashes without retaining the whole public corpus."""
    heap: list[tuple[int, str, dict[str, Any]]] = []
    for row in rows:
        identifier = str(row["id"])
        score = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big")
        item = (-score, identifier, row)
        if len(heap) < limit:
            heapq.heappush(heap, item)
        elif item > heap[0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, key=lambda value: (-value[0], value[1]))]


def parse_vsolscsum_xml(payload: bytes) -> Iterator[dict[str, Any]]:
    root = ET.fromstring(payload)
    for post in root.findall(".//post"):
        post_id = str(post.get("id") or "")
        title = post.findtext("title", default="")
        summary = [
            value
            for sentence in post.findall(".//summary/sentences/sentence")
            if (value := sentence.findtext("content", default="").strip())
        ]
        document = [
            value
            for sentence in post.findall(".//document/sentences/sentence")
            if (value := sentence.findtext("content", default="").strip())
        ]
        comments = [
            value
            for sentence in post.findall(".//comments/comment/sentences/sentence")
            if (value := sentence.findtext("content", default="").strip())
        ]
        if post_id and summary and document:
            yield {
                "id": f"vsolscsum:{post_id}",
                "dataset": "vsolscsum_vi",
                "language": "vi",
                "kind": "social_context",
                "title": title,
                "source": "\n".join([*document, *comments]),
                "reference": " ".join(summary),
            }


def _decode_literal_unicode(value: str) -> str:
    return re.sub(
        r"\\u([0-9a-fA-F]{4})",
        lambda match: chr(int(match.group(1), 16)),
        value,
    ).replace("\\n", "\n")


def _seahorse_rows(limit: int) -> list[dict[str, Any]]:
    spec = PUBLIC_SUMMARY_DATASETS["seahorse_vi"]
    stream = load_dataset(spec.repo, split="validation", revision=spec.revision, streaming=True)
    seen: set[str] = set()

    def rows() -> Iterator[dict[str, Any]]:
        for row in stream:
            identifier = str(row["gem_id"])
            if row["worker_lang"] != "vi" or row["model"] != "reference" or identifier in seen:
                continue
            seen.add(identifier)
            yield {
                "id": f"seahorse:{identifier}",
                "dataset": spec.key,
                "language": "vi",
                "kind": "human_rated_reference",
                "source": _decode_literal_unicode(str(row["article"])),
                "reference": _decode_literal_unicode(str(row["summary"])),
            }

    return _stable_sample(rows(), limit=limit)


def _wiki_lingua_rows(limit: int) -> list[dict[str, Any]]:
    spec = PUBLIC_SUMMARY_DATASETS["wiki_lingua_vi"]
    stream = load_dataset(
        spec.repo,
        "vietnamese",
        split="train",
        revision=spec.revision,
        streaming=True,
    )

    def rows() -> Iterator[dict[str, Any]]:
        for row in stream:
            article = row["article"]
            documents = [str(value).strip() for value in article["document"] if str(value).strip()]
            summaries = [str(value).strip() for value in article["summary"] if str(value).strip()]
            if not documents or not summaries:
                continue
            yield {
                "id": f"wiki_lingua:{row['url']}",
                "dataset": spec.key,
                "language": "vi",
                "kind": "how_to",
                "source": "\n".join(documents),
                "reference": " ".join(summaries),
            }

    return _stable_sample(rows(), limit=limit)


def _dialogsum_rows(limit: int) -> list[dict[str, Any]]:
    spec = PUBLIC_SUMMARY_DATASETS["dialogsum_en"]
    stream = load_dataset(spec.repo, split="validation", revision=spec.revision, streaming=True)
    return _stable_sample(
        (
            {
                "id": f"dialogsum:{row['id']}",
                "dataset": spec.key,
                "language": "en",
                "kind": "dialogue",
                "source": str(row["dialogue"]),
                "reference": str(row["summary"]),
            }
            for row in stream
            if row["dialogue"] and row["summary"]
        ),
        limit=limit,
    )


def _xlsum_rows(root: Path, limit: int) -> list[dict[str, Any]]:
    spec = PUBLIC_SUMMARY_DATASETS["xlsum_vi"]
    archive = Path(
        hf_hub_download(
            repo_id=spec.repo,
            repo_type="dataset",
            revision=spec.revision,
            filename=_XLSUM_FILENAME,
            local_dir=str(root / "_raw" / "xlsum"),
        )
    )
    with tarfile.open(archive, "r:bz2") as bundle:
        member = bundle.getmember("./vietnamese_val.jsonl")
        extracted = bundle.extractfile(member)
        if extracted is None:
            raise RuntimeError("XL-Sum Vietnamese validation split is missing")

        def rows() -> Iterator[dict[str, Any]]:
            for raw_line in extracted:
                row = json.loads(raw_line)
                if row.get("text") and row.get("summary"):
                    yield {
                        "id": f"xlsum:{row['id']}",
                        "dataset": spec.key,
                        "language": "vi",
                        "kind": "news",
                        "title": str(row.get("title", "")),
                        "source": str(row["text"]),
                        "reference": str(row["summary"]),
                    }

        return _stable_sample(rows(), limit=limit)


def _vsolscsum_rows(root: Path) -> list[dict[str, Any]]:
    spec = PUBLIC_SUMMARY_DATASETS["vsolscsum_vi"]
    raw_dir = root / "_raw" / "vsolscsum"
    _private_directory(raw_dir)
    url = (
        "https://raw.githubusercontent.com/nguyenlab/VSoLSCSum-Dataset/"
        f"{spec.revision}/{_VSOLSCSUM_FILENAME}"
    )
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - pinned HTTPS source
        payload = response.read()
    with tempfile.NamedTemporaryFile(dir=raw_dir, delete=False) as temporary:
        temporary.write(payload)
        temporary_path = Path(temporary.name)
    if _sha256(temporary_path) != _VSOLSCSUM_SHA256:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError("VSoLSCSum integrity verification failed")
    destination = raw_dir / _VSOLSCSUM_FILENAME
    temporary_path.replace(destination)
    destination.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return list(parse_vsolscsum_xml(payload))


def provision(*, root: Path, sample_size: int) -> dict[str, Any]:
    _private_directory(root)
    load_hf_token_from_dotenv()
    datasets = {
        "vsolscsum_vi": _vsolscsum_rows(root),
        "seahorse_vi": _seahorse_rows(sample_size),
        "wiki_lingua_vi": _wiki_lingua_rows(sample_size),
        "xlsum_vi": _xlsum_rows(root, sample_size),
        "dialogsum_en": _dialogsum_rows(sample_size),
    }
    artifacts: dict[str, dict[str, Any]] = {}
    for key, rows in datasets.items():
        path = root / f"{key}.jsonl"
        count, digest = _write_jsonl(path, rows)
        artifacts[key] = {"path": path.name, "rows": count, "sha256": digest}
    manifest = {
        "version": 1,
        "sample_policy": "lowest_sha256_id",
        "sample_size_per_large_dataset": sample_size,
        "datasets": {key: asdict(value) for key, value in PUBLIC_SUMMARY_DATASETS.items()},
        "artifacts": artifacts,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _make_tree_private(root)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Provision pinned public summary-eval data.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--sample-size", type=int, default=64)
    args = parser.parse_args(argv)
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    manifest = provision(root=args.output_root.resolve(), sample_size=args.sample_size)
    print(json.dumps(manifest["artifacts"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
