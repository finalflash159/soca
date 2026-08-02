from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from soca.asr.context import (
    ASRContextBuilder,
    ASRContextLimits,
    ASRContextSourceRecord,
    DynamicASRContextProvider,
    approximate_context_tokens,
    normalize_context_text,
)


def test_normalization_is_unicode_and_whitespace_deterministic() -> None:
    decomposed = "Ba\u0301o   ca\u0301o\n  tua\u0302\u0300n"

    assert normalize_context_text(decomposed) == "Báo cáo tuần"


def test_builder_is_order_independent_and_preserves_explicit_provenance() -> None:
    records = [
        ASRContextSourceRecord("  Transformer\nserving ", " vault heading ", priority=2),
        ASRContextSourceRecord("ONNX Runtime", "vault title", priority=1),
        ASRContextSourceRecord("transformer serving", "session", priority=0),
    ]
    builder = ASRContextBuilder()

    first = builder.build(records)
    second = builder.build(reversed(records))

    assert first == second
    assert first.text == "Transformer serving, ONNX Runtime"
    assert first.provenances == ("vault heading", "vault title")
    assert first.term_count == 2
    assert len(first.digest) == 64


def test_digest_covers_canonical_content_provenance_and_policy() -> None:
    source = [ASRContextSourceRecord("Qwen ASR", "active model")]

    original = ASRContextBuilder(ASRContextLimits(max_chars=50)).build(source)
    same = ASRContextBuilder(ASRContextLimits(max_chars=50)).build(source)
    other_source = ASRContextBuilder(ASRContextLimits(max_chars=50)).build(
        [ASRContextSourceRecord("Qwen ASR", "vault title")]
    )
    other_policy = ASRContextBuilder(ASRContextLimits(max_chars=51)).build(source)

    assert original.digest == same.digest
    assert original.digest != other_source.digest
    assert original.digest != other_policy.digest


def test_builder_enforces_all_bounds_without_partial_terms() -> None:
    builder = ASRContextBuilder(
        ASRContextLimits(max_chars=14, max_approximate_tokens=5, max_terms=2)
    )
    snapshot = builder.build(
        [
            ASRContextSourceRecord("toolong-for-limit", "vault", priority=5),
            ASRContextSourceRecord("alpha", "vault", priority=4),
            ASRContextSourceRecord("beta", "session", priority=3),
            ASRContextSourceRecord("gamma", "session", priority=2),
        ]
    )

    assert snapshot.text == "alpha, beta"
    assert snapshot.term_count == 2
    assert len(snapshot.text) <= snapshot.limits.max_chars
    assert snapshot.approximate_tokens <= snapshot.limits.max_approximate_tokens
    assert snapshot.approximate_tokens == approximate_context_tokens(snapshot.text)


def test_empty_sources_and_empty_values_produce_valid_empty_snapshot() -> None:
    builder = ASRContextBuilder()

    empty = builder.build([])
    blank = builder.build([ASRContextSourceRecord(" \n ", "session")])

    assert empty.is_empty
    assert empty.text == ""
    assert empty.entries == ()
    assert empty.approximate_tokens == 0
    assert blank == empty
    assert len(empty.digest) == 64


def test_dynamic_provider_reads_fresh_generic_records() -> None:
    records = [ASRContextSourceRecord("first", "session")]
    provider = DynamicASRContextProvider(lambda: tuple(records), ASRContextBuilder())

    assert provider.snapshot().text == "first"
    records.append(ASRContextSourceRecord("second", "vault"))
    assert provider.snapshot().text == "first, second"


def test_context_types_validate_inputs_and_are_immutable() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        ASRContextLimits(max_terms=0)
    with pytest.raises(ValueError, match="provenance"):
        ASRContextSourceRecord("term", "  ")
    with pytest.raises(TypeError, match="ASRContextSourceRecord"):
        ASRContextBuilder().build([object()])  # type: ignore[list-item]

    record = ASRContextSourceRecord("term", "source")
    with pytest.raises(FrozenInstanceError):
        record.value = "changed"  # type: ignore[misc]
