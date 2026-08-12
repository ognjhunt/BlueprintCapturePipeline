"""Provider secrets must resolve for a hardened service, not just a user shell.

`gpu_render_providers._read_secret` resolved only `~/.blueprint-secrets`, fixed
at import time. Every control-plane unit runs as `blueprint` with
`ProtectHome=true` and home `/nonexistent`, so that path can never exist there.

The units already publish the right locations -- `VAST_API_KEY_FILE` and
`BLUEPRINT_GPU_PROVIDER_SECRETS_DIR` -- and the runbook is explicit that the
allocator "binds its canonical allocator files explicitly so it never falls
back to a user home". The reader ignored both.

The observed cost: a terminal release reached the provider adapter, reported
`vast_api_key_missing`, and left a stopped instance stranded, which in turn
kept global provider-zero unverified and blocked every paid launch.
"""

from pathlib import Path

import pytest

from blueprint_pipeline import gpu_render_providers


@pytest.fixture
def secrets_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "provider-secrets"
    directory.mkdir()
    (directory / "vast_api_key").write_text("configured-directory-key\n")
    return directory


def test_reads_from_the_configured_provider_secrets_directory(
    monkeypatch: pytest.MonkeyPatch, secrets_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR", str(secrets_dir))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nonexistent"))
    assert gpu_render_providers._read_secret("vast_api_key") == "configured-directory-key"


def test_an_explicit_file_override_wins_over_the_directory(
    monkeypatch: pytest.MonkeyPatch, secrets_dir: Path, tmp_path: Path
) -> None:
    """The units name each file explicitly; that is the most specific instruction."""
    explicit = tmp_path / "explicit_key"
    explicit.write_text("explicit-file-key\n")
    monkeypatch.setenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR", str(secrets_dir))
    monkeypatch.setenv("VAST_API_KEY_FILE", str(explicit))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nonexistent"))
    assert gpu_render_providers._read_secret("vast_api_key") == "explicit-file-key"


def test_still_reads_the_developer_home_when_nothing_is_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local development must keep working exactly as before."""
    home = tmp_path / "home"
    (home / ".blueprint-secrets").mkdir(parents=True)
    (home / ".blueprint-secrets" / "vast_api_key").write_text("home-key\n")
    monkeypatch.delenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR", raising=False)
    monkeypatch.delenv("VAST_API_KEY_FILE", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(lambda: home))
    assert gpu_render_providers._read_secret("vast_api_key") == "home-key"


def test_resolution_is_not_frozen_at_import_time(
    monkeypatch: pytest.MonkeyPatch, secrets_dir: Path, tmp_path: Path
) -> None:
    """A module-level constant cannot see a systemd environment set later."""
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nonexistent"))
    monkeypatch.delenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR", raising=False)
    assert gpu_render_providers._read_secret("vast_api_key") is None

    monkeypatch.setenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR", str(secrets_dir))
    assert gpu_render_providers._read_secret("vast_api_key") == "configured-directory-key"


def test_missing_secret_returns_none_rather_than_raising(
    monkeypatch: pytest.MonkeyPatch, secrets_dir: Path, tmp_path: Path
) -> None:
    monkeypatch.setenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR", str(secrets_dir))
    monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path / "nonexistent"))
    assert gpu_render_providers._read_secret("runpod_api_key") is None


def test_unreadable_home_does_not_crash_the_reader(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ProtectHome makes home resolution itself fail on some systems."""

    def explode() -> Path:
        raise RuntimeError("home directory cannot be determined")

    monkeypatch.delenv("BLUEPRINT_GPU_PROVIDER_SECRETS_DIR", raising=False)
    monkeypatch.delenv("VAST_API_KEY_FILE", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(explode))
    assert gpu_render_providers._read_secret("vast_api_key") is None
