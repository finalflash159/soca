from __future__ import annotations

from eval.system_metrics import get_current_memory_mb


def test_get_current_memory_mb_returns_positive_value() -> None:
    memory_mb = get_current_memory_mb()

    assert memory_mb is None or memory_mb > 0
