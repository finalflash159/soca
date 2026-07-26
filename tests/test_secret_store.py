"""Tests for SecretStore (keyring-first read/write, no real keyring)."""

from __future__ import annotations

import stat

import pytest

from soca.config.secret_store import READ_DOTENV_ENV, SecretStore


class FakeKeyring:
    """In-memory stand-in for the ``keyring`` module."""

    def __init__(self) -> None:
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, value: str) -> None:
        self.store[(service, username)] = value


def _store(**kwargs) -> SecretStore:
    # No env, no .env, no json unless a test asks for them.
    kwargs.setdefault("env", {})
    kwargs.setdefault("dotenv_path", None)
    return SecretStore(**kwargs)


# -- mask -------------------------------------------------------------------


def test_mask_shows_only_a_prefix_and_suffix() -> None:
    assert SecretStore.mask("sk-proj-abcd1234") == "sk-...1234"


def test_mask_hides_short_secrets_entirely() -> None:
    assert "1234" not in SecretStore.mask("short")


def test_mask_handles_empty() -> None:
    assert SecretStore.mask("") == ""


# -- set / get via keyring --------------------------------------------------


def test_set_key_writes_to_keyring_and_get_reads_it_back() -> None:
    kr = FakeKeyring()
    store = _store(keyring_module=kr)

    store.set_key("groq", "sk-secret-123")

    assert kr.store[("soca-llm", "groq")] == "sk-secret-123"
    assert store.get_key("groq") == "sk-secret-123"
    assert store.has_key("groq") is True


def test_get_key_missing_returns_none() -> None:
    store = _store(keyring_module=FakeKeyring())
    assert store.get_key("openai") is None
    assert store.has_key("openai") is False


def test_set_key_rejects_unknown_provider() -> None:
    store = _store(keyring_module=FakeKeyring())
    with pytest.raises(ValueError, match="provider"):
        store.set_key("nope", "x")


def test_set_key_rejects_empty_value() -> None:
    store = _store(keyring_module=FakeKeyring())
    with pytest.raises(ValueError):
        store.set_key("groq", "  ")


# -- read precedence: keyring > env > .env > json ---------------------------


def test_keyring_wins_over_env() -> None:
    kr = FakeKeyring()
    kr.set_password("soca-llm", "openai", "from-keyring")
    store = _store(keyring_module=kr, env={"OPENAI_API_KEY": "from-env"})

    assert store.get_key("openai") == "from-keyring"


def test_env_used_when_keyring_empty() -> None:
    store = _store(keyring_module=FakeKeyring(), env={"OPENAI_API_KEY": "from-env"})
    assert store.get_key("openai") == "from-env"


def test_dotenv_used_when_keyring_and_env_empty(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text('# comment\nOPENAI_API_KEY="from-dotenv"\nOTHER=ignored\n', encoding="utf-8")
    store = SecretStore(keyring_module=FakeKeyring(), env={}, dotenv_path=dotenv)

    assert store.get_key("openai") == "from-dotenv"


def test_default_store_does_not_read_dotenv_from_the_working_directory(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-cwd-dotenv\n", encoding="utf-8")
    store = SecretStore(
        keyring_module=FakeKeyring(),
        env={},
        json_path=tmp_path / "keys.json",
    )

    assert store.get_key("openai") is None


def test_dotenv_read_requires_explicit_environment_opt_in(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=from-cwd-dotenv\n", encoding="utf-8")
    store = SecretStore(
        keyring_module=FakeKeyring(),
        env={READ_DOTENV_ENV: "1"},
        json_path=tmp_path / "keys.json",
    )

    assert store.get_key("openai") == "from-cwd-dotenv"


def test_json_fallback_used_last(tmp_path) -> None:
    keys_json = tmp_path / "keys.json"
    keys_json.write_text('{"groq": "from-json"}', encoding="utf-8")
    store = SecretStore(keyring_module=FakeKeyring(), env={}, dotenv_path=None, json_path=keys_json)

    assert store.get_key("groq") == "from-json"


# -- no-keyring machine: fall back to writing keys.json (0600), never .env --


def test_set_key_without_keyring_writes_json_0600(tmp_path) -> None:
    keys_json = tmp_path / "keys.json"
    store = SecretStore(keyring_module=None, env={}, dotenv_path=None, json_path=keys_json)

    store.set_key("groq", "sk-json-fallback")

    assert store.get_key("groq") == "sk-json-fallback"
    mode = stat.S_IMODE(keys_json.stat().st_mode)
    assert mode == 0o600


def test_set_key_never_writes_dotenv(tmp_path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text("EXISTING=1\n", encoding="utf-8")
    keys_json = tmp_path / "keys.json"
    store = SecretStore(keyring_module=None, env={}, dotenv_path=dotenv, json_path=keys_json)

    store.set_key("openai", "sk-should-not-touch-dotenv")

    assert "sk-should-not-touch-dotenv" not in dotenv.read_text(encoding="utf-8")
