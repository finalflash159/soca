from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from soca.knowledge.base import KnowledgeDocument, KnowledgeHit

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DEFAULT_EXCLUDE_DIRS = (".obsidian", ".trash", "private")
DEFAULT_EXCLUDE_FILES = ("index.md", "log.md")
DEFAULT_INCLUDE_GLOBS = ("**/*.md",)


@dataclass(frozen=True)
class SearchScoringConfig:
    title_weight: float = 6.0
    tag_weight: float = 5.0
    path_weight: float = 4.0
    body_weight: float = 1.0
    title_phrase_weight: float = 14.0
    tag_phrase_weight: float = 10.0
    path_phrase_weight: float = 8.0
    body_phrase_weight: float = 4.0
    wiki_link_phrase_weight: float = 5.0
    title_bigram_weight: float = 4.0
    tag_bigram_weight: float = 3.0
    path_bigram_weight: float = 2.0
    body_bigram_weight: float = 1.0
    max_term_frequency: int = 8


def fold_accents(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    decomposed = unicodedata.normalize("NFD", text)
    return "".join(char for char in decomposed if unicodedata.category(char) != "Mn")


def tokenize_terms(text: str) -> tuple[str, ...]:
    return tuple(TOKEN_RE.findall(fold_accents(text).lower()))


def tokenize(text: str) -> set[str]:
    return set(tokenize_terms(text))


def normalized_phrase(text: str) -> str:
    return " ".join(tokenize_terms(text))


def is_low_value_snippet_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return stripped.startswith("#")


class MarkdownVaultKnowledgeSource:
    def __init__(
        self,
        root: str | Path,
        exclude_dirs: tuple[str, ...] = DEFAULT_EXCLUDE_DIRS,
        exclude_files: tuple[str, ...] = DEFAULT_EXCLUDE_FILES,
        include_globs: tuple[str, ...] = DEFAULT_INCLUDE_GLOBS,
        max_file_bytes: int = 256 * 1024,
        scoring: SearchScoringConfig | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Knowledge vault not found: {self.root}")

        self.exclude_dirs = exclude_dirs
        self.exclude_files = exclude_files
        self.include_globs = include_globs
        self.max_file_bytes = max_file_bytes
        self.scoring = scoring or SearchScoringConfig()

    def _resolve_relative_path(self, path: str) -> Path:
        candidate = (self.root / path).resolve()

        try:
            candidate.relative_to(self.root)
        except ValueError as exc:
            raise ValueError(f"Path {path} is outside of the knowledge vault") from exc

        if candidate.suffix.lower() != ".md":
            raise ValueError(f"Only markdown files are supported: {path}")

        if self._is_excluded(candidate):
            raise ValueError(f"Path is excluded by configuration: {path}")

        return candidate

    def _is_excluded(self, path: Path) -> bool:
        rel_parts = path.relative_to(self.root).parts
        return any(part.startswith(".") or part in self.exclude_dirs for part in rel_parts)

    def read(self, path: str) -> KnowledgeDocument:
        file_path = self._resolve_relative_path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {path}")

        text = file_path.read_text(encoding="utf-8")
        rel_path = file_path.relative_to(self.root).as_posix()

        return KnowledgeDocument(
            id=rel_path,
            path=rel_path,
            title=self._extract_title(text, fallback=file_path.stem),
            text=text,
            tags=self._extract_tags(text),
        )

    def _extract_title(self, text: str, fallback: str) -> str:
        for line in text.splitlines():
            if line.startswith("# "):
                return line[2:].strip()
        return fallback

    def _extract_tags(self, text: str) -> tuple[str, ...]:
        tags = re.findall(r"(?<!\w)#([\w/-]+)", text)
        return tuple(sorted(set(tags)))

    def _iter_markdown_files(self) -> list[Path]:
        files: list[Path] = []

        for pattern in self.include_globs:
            for path in self.root.glob(pattern):
                if not path.is_file():
                    continue
                if path.name in self.exclude_files:
                    continue
                if self._is_excluded(path):
                    continue
                if path.stat().st_size > self.max_file_bytes:
                    continue
                if path.suffix.lower() != ".md":
                    continue

                files.append(path)
        return sorted(files)

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        query_terms = tokenize_terms(query)
        query_term_set = set(query_terms)
        if not query_term_set:
            return []

        docs: list[KnowledgeDocument] = []
        for path in self._iter_markdown_files():
            docs.append(self.read(path.relative_to(self.root).as_posix()))

        document_frequencies = self._document_frequencies(docs)
        query_phrase = " ".join(query_terms)
        query_bigrams = set(zip(query_terms, query_terms[1:], strict=False))
        hits: list[KnowledgeHit] = []

        for doc in docs:
            score = self._score(
                query_term_set=query_term_set,
                query_phrase=query_phrase,
                query_bigrams=query_bigrams,
                document_frequencies=document_frequencies,
                document_count=len(docs),
                doc=doc,
            )
            if score <= 0:
                continue

            hits.append(
                KnowledgeHit(
                    document=doc,
                    score=score,
                    snippet=self._make_snippet(doc.text, query_term_set),
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.document.path))
        return hits[:limit]

    def _document_frequencies(self, docs: list[KnowledgeDocument]) -> Counter[str]:
        frequencies: Counter[str] = Counter()
        for doc in docs:
            frequencies.update(set(self._document_terms(doc)))
        return frequencies

    def _document_terms(self, doc: KnowledgeDocument) -> tuple[str, ...]:
        tag_text = " ".join(doc.tags)
        path_text = doc.path.replace("/", " ").replace("-", " ")
        return tokenize_terms(f"{doc.title} {path_text} {tag_text} {doc.text}")

    def _score(
        self,
        *,
        query_term_set: set[str],
        query_phrase: str,
        query_bigrams: set[tuple[str, str]],
        document_frequencies: Counter[str],
        document_count: int,
        doc: KnowledgeDocument,
    ) -> float:
        title_terms = tokenize_terms(doc.title)
        path_terms = tokenize_terms(doc.path.replace("/", " ").replace("-", " "))
        tag_terms = tokenize_terms(" ".join(doc.tags))
        body_terms = tokenize_terms(doc.text)

        title_counter = Counter(title_terms)
        path_counter = Counter(path_terms)
        tag_counter = Counter(tag_terms)
        body_counter = Counter(body_terms)
        scoring = self.scoring

        return (
            self._field_score(
                query_term_set,
                title_counter,
                document_frequencies,
                document_count,
                weight=scoring.title_weight,
            )
            + self._field_score(
                query_term_set,
                tag_counter,
                document_frequencies,
                document_count,
                weight=scoring.tag_weight,
            )
            + self._field_score(
                query_term_set,
                path_counter,
                document_frequencies,
                document_count,
                weight=scoring.path_weight,
            )
            + self._field_score(
                query_term_set,
                body_counter,
                document_frequencies,
                document_count,
                weight=scoring.body_weight,
            )
            + self._phrase_score(query_phrase, doc, title_terms, tag_terms, path_terms, body_terms)
            + self._bigram_score(query_bigrams, title_terms, weight=scoring.title_bigram_weight)
            + self._bigram_score(query_bigrams, tag_terms, weight=scoring.tag_bigram_weight)
            + self._bigram_score(query_bigrams, path_terms, weight=scoring.path_bigram_weight)
            + self._bigram_score(query_bigrams, body_terms, weight=scoring.body_bigram_weight)
        )

    def _field_score(
        self,
        query_terms: set[str],
        field_counter: Counter[str],
        document_frequencies: Counter[str],
        document_count: int,
        *,
        weight: float,
    ) -> float:
        score = 0.0
        for term in query_terms:
            count = field_counter.get(term, 0)
            if count == 0:
                continue

            term_frequency = 1.0 + math.log(min(count, self.scoring.max_term_frequency))
            idf = math.log((document_count + 1.0) / (document_frequencies.get(term, 0) + 0.5)) + 1.0
            score += weight * term_frequency * idf
        return score

    def _phrase_score(
        self,
        query_phrase: str,
        doc: KnowledgeDocument,
        title_terms: tuple[str, ...],
        tag_terms: tuple[str, ...],
        path_terms: tuple[str, ...],
        body_terms: tuple[str, ...],
    ) -> float:
        if not query_phrase or " " not in query_phrase:
            return 0.0

        title_phrase = " ".join(title_terms)
        tag_phrase = " ".join(tag_terms)
        path_phrase = " ".join(path_terms)
        body_phrase = " ".join(body_terms)

        score = 0.0
        if query_phrase in title_phrase:
            score += self.scoring.title_phrase_weight
        if query_phrase in tag_phrase:
            score += self.scoring.tag_phrase_weight
        if query_phrase in path_phrase:
            score += self.scoring.path_phrase_weight
        if query_phrase in body_phrase:
            score += self.scoring.body_phrase_weight

        for link_text in re.findall(r"\[\[([^\]|]+)(?:\|([^\]]+))?\]\]", doc.text):
            alias_or_path = link_text[1] or link_text[0]
            if query_phrase in normalized_phrase(alias_or_path):
                score += self.scoring.wiki_link_phrase_weight
        return score

    def _bigram_score(self, query_bigrams: set[tuple[str, str]], field_terms: tuple[str, ...], *, weight: float) -> float:
        if not query_bigrams or len(field_terms) < 2:
            return 0.0
        field_bigrams = set(zip(field_terms, field_terms[1:], strict=False))
        return weight * len(query_bigrams & field_bigrams)

    def _make_snippet(self, text: str, query_terms: set[str], max_chars: int = 500) -> str:
        lines = text.splitlines()
        best_index: int | None = None
        best_score = 0

        for index, line in enumerate(lines):
            if is_low_value_snippet_line(line):
                continue

            line_terms = tokenize(line)
            score = len(query_terms & line_terms)
            if score > best_score:
                best_index = index
                best_score = score

        if best_index is not None:
            start = max(0, best_index - 2)
            end = min(len(lines), best_index + 5)
            snippet = "\n".join(
                candidate for candidate in lines[start:end] if not is_low_value_snippet_line(candidate)
            )
            return snippet if len(snippet) <= max_chars else snippet[:max_chars] + "..."

        for line in lines:
            if not is_low_value_snippet_line(line):
                snippet = line.strip()
                return snippet if len(snippet) <= max_chars else snippet[:max_chars] + "..."

        return ""
