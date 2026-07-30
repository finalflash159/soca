from __future__ import annotations

import json
from typing import Any

from soca.knowledge import KnowledgeSource
from soca.knowledge.catalog import KnowledgeCatalog
from soca.tools.base import (
    InvalidToolInput,
    PermanentToolError,
    SideEffectLevel,
    ToolExecutionStatus,
    ToolResult,
    ToolSpec,
    object_schema,
)


def _retrieval_metadata(diagnostics: Any | None) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    metadata: dict[str, Any] = {}
    for field in (
        "sparse_state",
        "dense_state",
        "index_state",
        "sparse_top_score",
        "dense_top_score",
        "sparse_separation",
        "dense_separation",
        "query_coverage",
        "unavailable_reason",
    ):
        value = getattr(diagnostics, field, None)
        if value not in (None, ""):
            metadata[field] = value
    metadata["retrieval_state"] = str(getattr(diagnostics, "overall_state", "ready"))
    return metadata


class KnowledgeInspectTool:
    """Inspect vault navigation metadata without manufacturing evidence."""

    def __init__(
        self,
        catalog: KnowledgeCatalog,
        *,
        max_documents: int = 32,
        max_chars: int = 8_000,
    ) -> None:
        if max_documents < 1 or max_chars < 1_024:
            raise ValueError("knowledge inspect limits are invalid")
        self.catalog = catalog
        self.max_documents = max_documents
        self.max_chars = max_chars

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.inspect",
            description=(
                "Inspect bounded local vault navigation metadata when the requested "
                "answer is a list or map of documents, folders, headings, links, or "
                "relationships. This returns structure, not what the user wrote inside "
                "a subject; use knowledge.search or knowledge.read for note contents."
            ),
            input_schema=object_schema(
                properties={
                    "scope": {"type": "string", "description": "Optional folder prefix."},
                    "path": {"type": "string", "description": "Optional exact document path."},
                    "depth": {"type": "integer", "description": "Optional relation depth."},
                    "limit": {"type": "integer", "description": "Maximum documents."},
                }
            ),
            side_effect=SideEffectLevel.READ_ONLY,
            workflow_capability="knowledge_catalog",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        scope = str(arguments.get("scope") or "").strip().replace("\\", "/")
        path = str(arguments.get("path") or "").strip().replace("\\", "/")
        if (
            scope.startswith("/")
            or path.startswith("/")
            or ".." in scope.split("/")
            or ".." in path.split("/")
        ):
            raise InvalidToolInput("unsafe_scope")
        raw_depth = arguments.get("depth", 0)
        if isinstance(raw_depth, bool):
            raise InvalidToolInput("invalid_depth")
        try:
            depth = int(raw_depth or 0)
        except (TypeError, ValueError) as exc:
            raise InvalidToolInput("invalid_depth") from exc
        if depth < 0 or depth > 3:
            raise InvalidToolInput("invalid_depth")
        raw_limit = arguments.get("limit", self.max_documents)
        if isinstance(raw_limit, bool):
            raise InvalidToolInput("invalid_limit")
        try:
            limit = int(raw_limit or self.max_documents)
        except (TypeError, ValueError) as exc:
            raise InvalidToolInput("invalid_limit") from exc
        if limit < 1:
            raise InvalidToolInput("invalid_limit")
        limit = min(limit, self.max_documents)
        try:
            snapshot = self.catalog.snapshot()
        except (OSError, RuntimeError, ValueError) as exc:
            raise PermanentToolError(
                "knowledge_inspect_unavailable",
                type(exc).__name__,
            ) from exc

        matched = tuple(
            document
            for document in snapshot.documents
            if (not scope or document.folder == scope or document.folder.startswith(scope + "/"))
            and (not path or document.path == path)
        )
        selected_paths = {document.path for document in matched}
        frontier = set(selected_paths)
        for _ in range(depth):
            connected = {
                relation.target
                for relation in snapshot.relations
                if relation.source in frontier
            }
            connected.update(
                relation.source
                for relation in snapshot.relations
                if relation.target in frontier
            )
            frontier = connected - selected_paths
            selected_paths.update(connected)
            if not frontier:
                break
        expanded = tuple(
            document for document in snapshot.documents if document.path in selected_paths
        )
        documents = expanded[:limit]
        def build_payload(
            selected_documents: tuple[Any, ...],
            *,
            include_headings: bool,
            truncated: bool,
        ) -> dict[str, Any]:
            selected_paths = {document.path for document in selected_documents}
            relations = tuple(
                relation
                for relation in snapshot.relations
                if relation.source in selected_paths and relation.target in selected_paths
            )
            document_payload: list[dict[str, Any]] = []
            for document in selected_documents:
                item: dict[str, Any] = {
                    "path": document.path,
                    "title": document.title,
                    "folder": document.folder,
                    "tags": list(document.tags),
                }
                if include_headings:
                    item["headings"] = [
                        {"level": heading.level, "text": heading.text, "line": heading.line}
                        for heading in document.headings
                    ]
                else:
                    item["headings"] = []
                    item["headings_omitted"] = True
                document_payload.append(item)
            return {
                "schema_version": 1,
                "revision": snapshot.revision,
                "content_digest": snapshot.content_digest,
                "scope": scope or None,
                "path": path or None,
                "depth": depth,
                "truncated": truncated,
                "documents": document_payload,
                "relations": [
                    {
                        "source": relation.source,
                        "target": relation.target,
                        "kind": relation.kind,
                    }
                    for relation in relations
                ],
                "metadata_only": True,
            }

        prefix = "Vault navigation metadata only; it is not evidence for note contents.\n"
        payload = build_payload(
            documents,
            include_headings=True,
            truncated=len(documents) < len(expanded),
        )
        content = prefix + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(content) > self.max_chars:
            payload = build_payload(documents, include_headings=False, truncated=True)
            content = prefix + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if len(content) > self.max_chars:
            for count in range(len(documents) - 1, -1, -1):
                candidate = build_payload(
                    documents[:count],
                    include_headings=False,
                    truncated=True,
                )
                candidate_content = prefix + json.dumps(
                    candidate,
                    ensure_ascii=False,
                    sort_keys=True,
                )
                if len(candidate_content) <= self.max_chars:
                    payload = candidate
                    content = candidate_content
                    break
        if len(content) > self.max_chars:
            raise PermanentToolError("knowledge_inspect_context_limit")
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=content,
            data={
                "revision": snapshot.revision,
                "content_digest": snapshot.content_digest,
                "depth": depth,
                "documents": payload["documents"],
                "relations": payload["relations"],
                "metadata_only": True,
                "truncated": payload["truncated"],
                "evidence_status": "not_requested",
                "retrieval_state": "ready",
            },
        )


class KnowledgeSearchTool:
    def __init__(
        self,
        source: KnowledgeSource,
        max_limit: int = 5,
    ) -> None:
        self.source = source
        self.max_limit = max_limit

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.search",
            description=(
                "Search local wiki markdown note content and return evidence snippets. "
                "Do not use for file inventory or vault structure."
            ),
            input_schema=object_schema(
                properties={
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of hits to return.",
                    },
                },
                required=["query"],
            ),
            side_effect=SideEffectLevel.READ_ONLY,
            workflow_capability="knowledge_search",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments["query"]).strip()
        if not query:
            raise InvalidToolInput("empty_query")

        limit = int(arguments.get("limit") or self.max_limit)
        limit = max(1, min(limit, self.max_limit))
        retrieve = getattr(self.source, "retrieve", None)
        diagnostics: Any | None = None
        if callable(retrieve):
            batch = retrieve(query, limit=limit)
            hits = list(getattr(batch, "hits", ()))
            diagnostics = getattr(batch, "diagnostics", None)
        else:
            hits = self.source.search(query, limit=limit)

        if not hits:
            data: dict[str, Any] = {"hits": []}
            data.update(_retrieval_metadata(diagnostics))
            return ToolResult(
                name=self.spec.name,
                ok=True,
                content="Mình chưa tìm thấy thông tin đó trong knowledge vault.",
                data=data,
            )

        lines: list[str] = []
        data_hits: list[dict[str, Any]] = []
        for index, hit in enumerate(hits, start=1):
            lines.append(f"[K{index}] {hit.document.title} ({hit.document.path})\n{hit.snippet}")
            data_hit: dict[str, Any] = {
                "path": hit.document.path,
                "title": hit.document.title,
                "score": hit.score,
                "snippet": hit.snippet,
                "retrieval_backend": hit.retrieval_backend,
            }
            for field in ("sparse_score", "dense_score", "fusion_score"):
                value = getattr(hit, field, None)
                if value is not None:
                    data_hit[field] = value
            if hit.line_start is not None:
                data_hit["line_start"] = hit.line_start
                data_hit["line_end"] = hit.line_end
            data_hits.append(data_hit)

        metadata: dict[str, Any] = {"hits": data_hits}
        metadata.update(_retrieval_metadata(diagnostics))
        return ToolResult(
            name=self.spec.name,
            ok=True,
            content="\n\n".join(lines),
            data=metadata,
        )


class KnowledgeReadTool:
    def __init__(
        self,
        source: KnowledgeSource,
        max_chars: int = 4000,
        max_lines: int = 200,
    ) -> None:
        if max_chars < 256 or max_lines < 1:
            raise ValueError("knowledge read limits are invalid")
        self.source = source
        self.max_chars = max_chars
        self.max_lines = max_lines

    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="knowledge.read",
            description=(
                "Read a known local wiki Markdown path as note-body evidence. Optional "
                "1-indexed start_line/end_line select an exact bounded range. The receipt "
                "reports total lines and a continuation cursor when the requested content "
                "does not fit. Use search when the path is unknown."
            ),
            input_schema=object_schema(
                properties={
                    "path": {
                        "type": "string",
                        "description": "Vault-relative markdown path.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "Optional 1-indexed first line.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Optional inclusive 1-indexed last line.",
                    },
                },
                required=["path"],
            ),
            side_effect=SideEffectLevel.READ_ONLY,
            workflow_capability="knowledge_read",
        )

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        path = str(arguments["path"]).strip()
        if not path:
            raise InvalidToolInput("empty_path")
        start_line = self._line_argument(arguments, "start_line", default=1)
        end_line = self._optional_line_argument(arguments, "end_line")
        if end_line is not None and end_line < start_line:
            raise InvalidToolInput("invalid_line_range")
        if end_line is not None and end_line - start_line + 1 > self.max_lines:
            raise InvalidToolInput("line_range_too_large")

        try:
            doc = self.source.read(path)
        except FileNotFoundError:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                content="",
                error="not_found",
                status=ToolExecutionStatus.NOT_FOUND,
            )
        except ValueError:
            return ToolResult(
                name=self.spec.name,
                ok=False,
                content="",
                error="invalid_path",
                status=ToolExecutionStatus.INVALID,
            )
        lines = doc.text.splitlines()
        total_lines = len(lines)
        if total_lines == 0:
            lines = [""]
            total_lines = 1
        if start_line > total_lines:
            raise InvalidToolInput("start_line_out_of_range")

        explicit_end = end_line is not None
        requested_end = min(
            total_lines,
            end_line if end_line is not None else start_line + self.max_lines - 1,
        )
        header_budget = (
            f"# {doc.title}\n\nLines {start_line}-{requested_end} of {total_lines}:\n"
        )
        selected: list[str] = []
        used_chars = len(header_budget)
        actual_end = start_line - 1
        for line_number in range(start_line, requested_end + 1):
            rendered = f"{line_number}: {lines[line_number - 1]}"
            added_chars = len(rendered) + (1 if selected else 0)
            if used_chars + added_chars > self.max_chars:
                if not selected:
                    raise PermanentToolError("knowledge_read_line_too_large")
                break
            selected.append(rendered)
            used_chars += added_chars
            actual_end = line_number

        range_complete = actual_end == requested_end
        document_complete = actual_end == total_lines
        complete = range_complete and (explicit_end or document_complete)
        next_start_line = None if complete else actual_end + 1
        header = f"# {doc.title}\n\nLines {start_line}-{actual_end} of {total_lines}:\n"
        content = header + "\n".join(selected)

        return ToolResult(
            name=self.spec.name,
            ok=True,
            content=content,
            data={
                "path": doc.path,
                "title": doc.title,
                "tags": list(doc.tags),
                "line_start": start_line,
                "line_end": actual_end,
                "total_lines": total_lines,
                "requested_end_line": end_line,
                "next_start_line": next_start_line,
                "complete": complete,
                "document_complete": document_complete,
                "truncated": not complete,
            },
        )

    @staticmethod
    def _line_argument(
        arguments: dict[str, Any],
        name: str,
        *,
        default: int,
    ) -> int:
        value = arguments.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidToolInput(f"invalid_{name}")
        return value

    @staticmethod
    def _optional_line_argument(
        arguments: dict[str, Any],
        name: str,
    ) -> int | None:
        value = arguments.get(name)
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InvalidToolInput(f"invalid_{name}")
        return value
