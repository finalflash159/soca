from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from types import MappingProxyType

from soca.knowledge.base import KnowledgeDocument, KnowledgeHit

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
DEFAULT_EXCLUDE_DIRS = (".obsidian", ".trash", "private")
DEFAULT_EXCLUDE_FILES = ("index.md", "log.md")
DEFAULT_INCLUDE_GLOBS = ("**/*.md",)


def _validate_include_globs(patterns: tuple[str, ...]) -> None:
    for pattern in patterns:
        if not isinstance(pattern, str) or not pattern or "\\" in pattern:
            raise ValueError("include globs must be non-empty POSIX strings")
        parts = Path(pattern).parts
        if Path(pattern).is_absolute() or ".." in parts:
            raise ValueError("include globs must stay relative to the vault")


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
    min_out_of_vault_score_ratio: float = 1.3
    min_out_of_vault_query_coverage: float = 0.35

    def __post_init__(self) -> None:
        ratio = self.min_out_of_vault_score_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("min_out_of_vault_score_ratio must be a number")
        if not math.isfinite(float(ratio)) or float(ratio) < 1.0:
            raise ValueError("min_out_of_vault_score_ratio must be finite and at least 1")
        coverage = self.min_out_of_vault_query_coverage
        if isinstance(coverage, bool) or not isinstance(coverage, (int, float)):
            raise ValueError("min_out_of_vault_query_coverage must be a number")
        if not 0.0 <= float(coverage) <= 1.0:
            raise ValueError("min_out_of_vault_query_coverage must be between 0 and 1")


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
        _validate_include_globs(self.include_globs)
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

    def iter_paths(self) -> tuple[str, ...]:
        return tuple(path.relative_to(self.root).as_posix() for path in self._iter_markdown_files())

    def _has_symlink_component(self, candidate: Path) -> bool:
        current = self.root
        for part in candidate.relative_to(self.root).parts:
            current = current / part
            if current.is_symlink():
                return True
        return False

    def _iter_markdown_files(self) -> list[Path]:
        files_by_path: dict[str, Path] = {}

        for pattern in self.include_globs:
            for candidate in self.root.glob(pattern):
                try:
                    candidate.relative_to(self.root)
                    if self._has_symlink_component(candidate):
                        continue
                    path = candidate.resolve(strict=True)
                    path.relative_to(self.root)
                except (FileNotFoundError, OSError, ValueError):
                    continue

                if not path.is_file():
                    continue
                if path.name in self.exclude_files:
                    continue
                if self._is_excluded(path):
                    continue
                if path.suffix.lower() != ".md":
                    continue
                try:
                    if path.stat().st_size > self.max_file_bytes:
                        continue
                except OSError:
                    continue

                relative = path.relative_to(self.root).as_posix()
                files_by_path[relative] = path

        return [files_by_path[path] for path in sorted(files_by_path)]

    def search(self, query: str, limit: int = 5) -> list[KnowledgeHit]:
        documents = tuple(self.read(path) for path in self.iter_paths())
        return self._search_documents(query, documents, limit=limit)

    def _search_documents(
        self,
        query: str,
        documents: tuple[KnowledgeDocument, ...],
        *,
        limit: int,
    ) -> list[KnowledgeHit]:
        return self._search_lexical_snapshot(
            query,
            prepare_lexical_snapshot(documents),
            limit=limit,
        )

    def _search_lexical_snapshot(
        self,
        query: str,
        snapshot: LexicalSnapshot,
        *,
        limit: int,
    ) -> list[KnowledgeHit]:
        query_terms = tokenize_terms(query)
        query_term_set = set(query_terms)
        if not query_term_set:
            return []

        query_phrase = " ".join(query_terms)
        query_bigrams = set(zip(query_terms, query_terms[1:], strict=False))
        hits: list[KnowledgeHit] = []

        for entry in snapshot.entries:
            score = self._score_prepared(
                query_term_set=query_term_set,
                query_phrase=query_phrase,
                query_bigrams=query_bigrams,
                document_frequencies=snapshot.document_frequencies,
                document_count=len(snapshot.entries),
                entry=entry,
            )
            if score <= 0:
                continue
            hits.append(
                KnowledgeHit(
                    document=entry.document,
                    score=score,
                    snippet=self._make_snippet(
                        entry.document.text,
                        query_term_set,
                    ),
                    retrieval_backend="lexical_custom",
                    sparse_score=score,
                )
            )

        hits.sort(key=lambda hit: (-hit.score, hit.document.path))
        if not self._has_confident_query_match(query_terms, snapshot, hits):
            return []
        return hits[:limit]

    def _has_confident_query_match(
        self,
        query_terms: tuple[str, ...],
        snapshot: LexicalSnapshot,
        hits: list[KnowledgeHit],
    ) -> bool:
        """Reject weak matches when the query contains out-of-vault evidence.

        The decision is derived from the current vault's document frequencies and
        score distribution. It intentionally has no vocabulary-specific rules:
        an unknown term is only meaningful because this vault cannot support it,
        and a result is accepted when it clearly separates from its competitors.
        """
        if not hits:
            return False

        has_out_of_vault_term = any(
            snapshot.document_frequencies.get(term, 0) == 0
            for term in set(query_terms)
        )
        if not has_out_of_vault_term:
            return True

        best_by_document: dict[str, float] = {}
        for hit in hits:
            current = best_by_document.get(hit.document.path, 0.0)
            best_by_document[hit.document.path] = max(current, hit.score)

        query_term_set = set(query_terms)
        document_count = len(snapshot.entries)
        query_weights = {
            term: math.log(
                (document_count + 1.0)
                / (snapshot.document_frequencies.get(term, 0) + 0.5)
            )
            + 1.0
            for term in query_term_set
        }
        total_query_weight = sum(query_weights.values())
        coverage_by_document: dict[str, float] = {}
        if total_query_weight > 0.0:
            for entry in snapshot.entries:
                document_terms = set(
                    chain(
                        entry.title_terms,
                        entry.tag_terms,
                        entry.path_terms,
                        entry.body_terms,
                    )
                )
                coverage = sum(
                    weight
                    for term, weight in query_weights.items()
                    if term in document_terms
                ) / total_query_weight
                current = coverage_by_document.get(entry.document.path, 0.0)
                coverage_by_document[entry.document.path] = max(current, coverage)

        best_path = max(best_by_document, key=best_by_document.__getitem__)
        if (
            coverage_by_document.get(best_path, 0.0)
            < self.scoring.min_out_of_vault_query_coverage
        ):
            return False

        scores = sorted(best_by_document.values(), reverse=True)
        if len(scores) < 2:
            return True

        return (
            scores[0] / scores[1] >= self.scoring.min_out_of_vault_score_ratio
        )

    def _score_prepared(
        self,
        *,
        query_term_set: set[str],
        query_phrase: str,
        query_bigrams: set[tuple[str, str]],
        document_frequencies: Mapping[str, int],
        document_count: int,
        entry: PreparedLexicalDocument,
    ) -> float:
        scoring = self.scoring
        return (
            self._field_score(
                query_term_set,
                entry.title_counter,
                document_frequencies,
                document_count,
                weight=scoring.title_weight,
            )
            + self._field_score(
                query_term_set,
                entry.tag_counter,
                document_frequencies,
                document_count,
                weight=scoring.tag_weight,
            )
            + self._field_score(
                query_term_set,
                entry.path_counter,
                document_frequencies,
                document_count,
                weight=scoring.path_weight,
            )
            + self._field_score(
                query_term_set,
                entry.body_counter,
                document_frequencies,
                document_count,
                weight=scoring.body_weight,
            )
            + self._phrase_score(
                query_phrase,
                entry.document,
                entry.title_terms,
                entry.tag_terms,
                entry.path_terms,
                entry.body_terms,
            )
            + self._bigram_score(
                query_bigrams,
                entry.title_terms,
                weight=scoring.title_bigram_weight,
            )
            + self._bigram_score(
                query_bigrams,
                entry.tag_terms,
                weight=scoring.tag_bigram_weight,
            )
            + self._bigram_score(
                query_bigrams,
                entry.path_terms,
                weight=scoring.path_bigram_weight,
            )
            + self._bigram_score(
                query_bigrams,
                entry.body_terms,
                weight=scoring.body_bigram_weight,
            )
        )

    def _field_score(
        self,
        query_terms: set[str],
        field_counter: Mapping[str, int],
        document_frequencies: Mapping[str, int],
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

    def _bigram_score(
        self, query_bigrams: set[tuple[str, str]], field_terms: tuple[str, ...], *, weight: float
    ) -> float:
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
                candidate
                for candidate in lines[start:end]
                if not is_low_value_snippet_line(candidate)
            )
            return snippet if len(snippet) <= max_chars else snippet[:max_chars] + "..."

        for line in lines:
            if not is_low_value_snippet_line(line):
                snippet = line.strip()
                return snippet if len(snippet) <= max_chars else snippet[:max_chars] + "..."

        return ""


@dataclass(frozen=True)
class PreparedLexicalDocument:
    document: KnowledgeDocument
    title_terms: tuple[str, ...]
    tag_terms: tuple[str, ...]
    path_terms: tuple[str, ...]
    body_terms: tuple[str, ...]
    title_counter: Mapping[str, int]
    tag_counter: Mapping[str, int]
    path_counter: Mapping[str, int]
    body_counter: Mapping[str, int]

    def __post_init__(self) -> None:
        for name in (
            "title_counter",
            "tag_counter",
            "path_counter",
            "body_counter",
        ):
            object.__setattr__(
                self,
                name,
                MappingProxyType(dict(getattr(self, name))),
            )


@dataclass(frozen=True)
class LexicalSnapshot:
    entries: tuple[PreparedLexicalDocument, ...]
    document_frequencies: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "document_frequencies",
            MappingProxyType(dict(self.document_frequencies)),
        )


def prepare_lexical_snapshot(
    documents: tuple[KnowledgeDocument, ...],
) -> LexicalSnapshot:
    entries: list[PreparedLexicalDocument] = []
    frequencies: Counter[str] = Counter()
    for document in documents:
        title_terms = tokenize_terms(document.title)
        tag_terms = tokenize_terms(" ".join(document.tags))
        path_terms = tokenize_terms(document.path.replace("/", " ").replace("-", " "))
        body_terms = tokenize_terms(document.text)
        frequencies.update(set(chain(title_terms, tag_terms, path_terms, body_terms)))
        entries.append(
            PreparedLexicalDocument(
                document=document,
                title_terms=title_terms,
                tag_terms=tag_terms,
                path_terms=path_terms,
                body_terms=body_terms,
                title_counter=Counter(title_terms),
                tag_counter=Counter(tag_terms),
                path_counter=Counter(path_terms),
                body_counter=Counter(body_terms),
            )
        )
    return LexicalSnapshot(tuple(entries), frequencies)
