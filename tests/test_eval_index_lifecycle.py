from __future__ import annotations

from eval.eval_index_lifecycle import run


def test_lifecycle_probe_measures_incremental_reuse() -> None:
    report = run(4)

    assert report["kind"] == "index_lifecycle_probe"
    assert report["chunks"] == 4
    assert report["edit_embedded_rows"] == 1
    assert report["rename_reused_rows"] == 4
    assert report["warm_dense_state"] == "ready"
