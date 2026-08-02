from __future__ import annotations

from types import SimpleNamespace

from soca.asr.context_sources import runtime_context_records


def test_runtime_context_records_preserve_typed_catalog_provenance() -> None:
    snapshot = SimpleNamespace(
        revision=7,
        documents=(
            SimpleNamespace(
                path="wiki/ml.md",
                title="Attention dễ hiểu",
                tags=("transformer",),
                headings=(SimpleNamespace(text="Scaled dot product", line=12),),
            ),
        ),
    )
    catalog = SimpleNamespace(snapshot=lambda: snapshot)

    records = runtime_context_records(catalog, None)

    assert [(record.value, record.priority) for record in records] == [
        ("Attention dễ hiểu", 30),
        ("transformer", 20),
        ("Scaled dot product", 10),
    ]
    assert all(record.provenance.startswith("vault:7:wiki/ml.md:") for record in records)
