from __future__ import annotations

from soca.knowledge.citation_preview import citation_fingerprint, preview_vault_citation


def _vault(tmp_path):
    vault = tmp_path / "vault"
    (vault / "wiki").mkdir(parents=True)
    return vault


def test_preview_returns_bounded_current_passage(tmp_path) -> None:
    vault = _vault(tmp_path)
    text = "# Kế hoạch\nMục tiêu\nBước một\nBước hai\n"
    (vault / "wiki" / "plan.md").write_text(text, encoding="utf-8")

    preview = preview_vault_citation(
        vault,
        path="wiki/plan.md",
        line_start=2,
        line_end=3,
        expected_fingerprint=citation_fingerprint(text),
    )

    assert preview.status == "current"
    assert preview.title == "Kế hoạch"
    assert preview.passage == "Mục tiêu\nBước một"
    assert preview.error_code is None


def test_preview_flags_changed_source_without_claiming_original_evidence(tmp_path) -> None:
    vault = _vault(tmp_path)
    original = "# Kế hoạch\nBản cũ\n"
    current = "# Kế hoạch\nBản mới\n"
    (vault / "wiki" / "plan.md").write_text(current, encoding="utf-8")

    preview = preview_vault_citation(
        vault,
        path="wiki/plan.md",
        line_start=2,
        line_end=2,
        expected_fingerprint=citation_fingerprint(original),
    )

    assert preview.status == "changed"
    assert preview.passage == "Bản mới"


def test_preview_distinguishes_legacy_and_missing_evidence(tmp_path) -> None:
    vault = _vault(tmp_path)
    (vault / "wiki" / "plan.md").write_text("# Kế hoạch\nNội dung\n", encoding="utf-8")

    legacy = preview_vault_citation(
        vault,
        path="wiki/plan.md",
        line_start=2,
        line_end=2,
        expected_fingerprint=None,
    )
    missing = preview_vault_citation(
        vault,
        path="wiki/missing.md",
        line_start=1,
        line_end=1,
        expected_fingerprint=None,
    )

    assert legacy.status == "unverified"
    assert legacy.passage == "Nội dung"
    assert missing.status == "missing"
    assert missing.error_code == "source_missing"
    assert missing.passage is None


def test_preview_rejects_unsafe_and_unbounded_locations(tmp_path) -> None:
    vault = _vault(tmp_path)

    outside = preview_vault_citation(
        vault,
        path="../outside.md",
        line_start=1,
        line_end=1,
        expected_fingerprint=None,
    )
    oversized_range = preview_vault_citation(
        vault,
        path="wiki/plan.md",
        line_start=1,
        line_end=121,
        expected_fingerprint=None,
    )

    assert outside.status == "unavailable"
    assert outside.error_code == "source_unavailable"
    assert oversized_range.status == "unavailable"
    assert oversized_range.error_code == "citation_range_too_large"


def test_preview_reports_an_unavailable_vault_without_claiming_the_source_is_missing(tmp_path) -> None:
    preview = preview_vault_citation(
        tmp_path / "missing-vault",
        path="wiki/plan.md",
        line_start=1,
        line_end=1,
        expected_fingerprint=None,
    )

    assert preview.status == "unavailable"
    assert preview.error_code == "knowledge_vault_unavailable"
