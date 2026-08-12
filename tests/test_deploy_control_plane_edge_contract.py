"""Contract for the parts of the control-plane host that are not systemd units.

The 2026-08-12 rebuild found that a destroyed Pipeline host could not be
recreated from the repository: the TLS/reverse-proxy layer existed only as prose
in an archived runbook, and the installer never wrote the deployed source
commit, so ``/api/live-pipeline/version`` answered 503 with
``commit_proven=false`` on an otherwise healthy host.  Both are pinned here so a
future rebuild is a script run rather than a rediscovery.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CADDYFILE = REPO_ROOT / "deploy" / "caddy" / "Caddyfile"
INSTALL_SCRIPT = REPO_ROOT / "scripts" / "install_live_pipeline_control_plane.sh"
INTAKE_SERVICE = REPO_ROOT / "deploy" / "systemd" / "blueprint-pipeline-intake.service"

# The intake service binds loopback only; the edge is the sole public surface.
INTAKE_UPSTREAM = "127.0.0.1:8765"
PUBLIC_HOSTNAME_ENV = "BLUEPRINT_PIPELINE_PUBLIC_HOSTNAME"
SOURCE_COMMIT_ENV = "BLUEPRINT_SOURCE_COMMIT"


def _caddyfile() -> str:
    return CADDYFILE.read_text(encoding="utf-8")


def _install_script() -> str:
    return INSTALL_SCRIPT.read_text(encoding="utf-8")


def test_caddy_edge_config_is_checked_in() -> None:
    assert CADDYFILE.is_file(), (
        "The control-plane TLS/reverse-proxy config must live in the repository. "
        "Without it a destroyed host cannot be rebuilt from protected main."
    )


def test_caddy_edge_forwards_live_pipeline_api_to_the_loopback_intake() -> None:
    text = _caddyfile()
    assert "/api/live-pipeline/*" in text
    assert f"reverse_proxy {INTAKE_UPSTREAM}" in text


def test_caddy_edge_hostname_is_configurable_not_hardcoded() -> None:
    """A rebuilt host gets a new address; the edge must not pin a dead one."""
    text = _caddyfile()
    assert f"{{${PUBLIC_HOSTNAME_ENV}}}" in text, (
        "The public hostname must come from "
        f"{PUBLIC_HOSTNAME_ENV} so a rebuild does not require editing the repo."
    )
    # The destroyed host's literal address must never reappear as config.
    assert "206.81.11.69" not in text


def test_caddy_edge_does_not_expose_surfaces_beyond_the_live_pipeline_api() -> None:
    text = _caddyfile()
    proxied = [line for line in text.splitlines() if "reverse_proxy" in line]
    assert proxied, "expected at least one reverse_proxy directive"
    assert all(INTAKE_UPSTREAM in line for line in proxied), (
        "the edge must not proxy anything except the loopback intake service"
    )


def test_installer_records_the_deployed_source_commit() -> None:
    """``/api/live-pipeline/version`` returns 503 until this env var is set."""
    text = _install_script()
    assert SOURCE_COMMIT_ENV in text, (
        f"{SOURCE_COMMIT_ENV} must be written by the installer; the intake "
        "service reports commit_proven=false and answers 503 without it."
    )


def test_installer_installs_the_caddy_edge_config() -> None:
    text = _install_script()
    assert "caddy" in text.lower(), (
        "the installer must render the checked-in Caddy edge config"
    )


def test_intake_service_stays_loopback_only() -> None:
    """The edge is the only public listener; pin that the unit has not drifted."""
    text = INTAKE_SERVICE.read_text(encoding="utf-8")
    assert "live_pipeline_intake_service" in text
