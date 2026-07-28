from __future__ import annotations

from scripts.provision_summary_eval_data import (
    PUBLIC_SUMMARY_DATASETS,
    _make_tree_private,
    _stable_sample,
    parse_vsolscsum_xml,
)


def test_public_summary_dataset_specs_are_revision_pinned_and_role_scoped() -> None:
    assert set(PUBLIC_SUMMARY_DATASETS) == {
        "vsolscsum_vi",
        "seahorse_vi",
        "wiki_lingua_vi",
        "xlsum_vi",
        "dialogsum_en",
    }
    assert all(len(spec.revision) == 40 for spec in PUBLIC_SUMMARY_DATASETS.values())
    assert "not Vietnamese evidence" in PUBLIC_SUMMARY_DATASETS["dialogsum_en"].role


def test_stable_sample_is_order_independent() -> None:
    rows = [{"id": value} for value in ("c", "a", "d", "b")]
    assert _stable_sample(rows, limit=2) == _stable_sample(reversed(rows), limit=2)


def test_vsolscsum_parser_keeps_human_summary_and_social_context() -> None:
    payload = b"""\
    <root><post id="p1"><title>Tieu de</title>
      <summary><sentences><sentence><content>Tom tat.</content></sentence></sentences></summary>
      <document><sentences><sentence><content>Van ban.</content></sentence></sentences></document>
      <comments><comment><sentences><sentence><content>Binh luan.</content></sentence></sentences></comment></comments>
    </post></root>"""
    assert list(parse_vsolscsum_xml(payload)) == [
        {
            "id": "vsolscsum:p1",
            "dataset": "vsolscsum_vi",
            "language": "vi",
            "kind": "social_context",
            "title": "Tieu de",
            "source": "Van ban.\nBinh luan.",
            "reference": "Tom tat.",
        }
    ]


def test_public_summary_data_tree_is_made_private(tmp_path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    artifact = nested / "data.jsonl"
    artifact.write_text("{}\n")
    nested.chmod(0o755)
    artifact.chmod(0o644)

    _make_tree_private(tmp_path)

    assert tmp_path.stat().st_mode & 0o077 == 0
    assert nested.stat().st_mode & 0o077 == 0
    assert artifact.stat().st_mode & 0o077 == 0
