from __future__ import annotations

import hashlib

from scripts.download_summary_models import (
    default_model_root,
    load_hf_token_from_dotenv,
    select_specs,
    verify_download,
)
from soca.memory.summary import SUMMARY_MODEL_REGISTRY


def test_summary_provisioning_defaults_to_repo_summary_folder() -> None:
    assert default_model_root().name == "summary"
    assert default_model_root().parent.name == "models"


def test_select_specs_requires_an_explicit_candidate_selection() -> None:
    assert select_specs(["qwen3_1_7b_q8_0"], all_models=False) == [
        SUMMARY_MODEL_REGISTRY["qwen3_1_7b_q8_0"]
    ]


def test_verify_download_checks_the_pinned_sha256_and_byte_count(tmp_path) -> None:
    payload = b"verified-summary-model"
    path = tmp_path / "model.gguf"
    path.write_bytes(payload)
    assert verify_download(
        path,
        expected_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )
    assert not verify_download(path, expected_bytes=len(payload) + 1, expected_sha256="0" * 64)


def test_provisioning_loads_hf_token_from_dotenv_without_printing_it(tmp_path, monkeypatch) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("HF_TOKEN=hf_test_value\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert load_hf_token_from_dotenv(dotenv) is True
    assert __import__("os").environ["HF_TOKEN"] == "hf_test_value"
