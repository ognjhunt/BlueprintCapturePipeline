"""The dependencies a control-plane host needs must be installed by a script.

The 2026-08-12 rebuild installed them by hand, twice. A base `pip install -e .`
omits the `runtime` extra, so the canonical allocator could not be imported;
installing that extra then pulled a non-headless OpenCV whose `cv2` needs
`libGL.so.1`, absent on a server image. Both were discovered only when a
stranded provider record could not be released.

The runtime guard now detects an unimportable entrypoint, but detection is not
installation: a rebuilt host would still fail until someone remembered these
two steps. Pin them here so the rebuild is a script run.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP = REPO_ROOT / "scripts" / "bootstrap_control_plane_host.sh"


def _bootstrap() -> str:
    return BOOTSTRAP.read_text(encoding="utf-8")


def test_bootstrap_script_is_checked_in() -> None:
    assert BOOTSTRAP.is_file(), (
        "a control-plane host rebuild must not depend on remembered shell history"
    )


def test_installs_the_runtime_extra_not_a_bare_package() -> None:
    """A bare install leaves the canonical allocator unimportable."""
    text = _bootstrap()
    assert re.search(r'pip["\']?\s+install[^\n]*\[runtime\]', text), (
        'the host must install the "runtime" extra; a bare install omits opencv '
        "and the canonical allocator cannot be imported"
    )


def test_installs_the_system_libraries_headless_opencv_needs() -> None:
    text = _bootstrap()
    for package in ("libgl1", "libglib2.0-0"):
        assert package in text, (
            f"{package} is required for cv2 on a server image; without it the "
            "allocator raises ImportError: libGL.so.1"
        )


def test_installs_the_edge_and_reverse_proxy() -> None:
    assert "caddy" in _bootstrap().lower(), (
        "the intake service binds loopback only, so the host needs its edge"
    )


def test_verifies_entrypoints_before_declaring_success() -> None:
    """Installing is not proof; the script must prove the result."""
    text = _bootstrap()
    assert "production_runtime_env_guard" in text, (
        "the bootstrap must run the runtime guard so an unimportable entrypoint "
        "fails the rebuild rather than surfacing at the first paid operation"
    )


def test_is_executable() -> None:
    assert BOOTSTRAP.stat().st_mode & 0o111, "bootstrap script must be executable"
