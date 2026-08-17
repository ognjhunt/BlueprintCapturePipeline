"""Stop new credentials leaking into VM UserData while the old ones are fixed.

`cloud_vm_render_providers._worker_cloud_init` base64s the whole of
``spec.env`` into GCP instance metadata and EC2 UserData. Both are readable
from the instance over IMDS and from the account via
``DescribeInstanceAttribute``, so anything secret-shaped in that mapping is
published to every principal who can describe the VM.

Two lanes already do this. Fixing them means moving those values to a signed
fetch-then-delete, the way the Postshot licence works — a real change to live
lanes, not a guard that can simply be switched on. Until that lands, this
ratchet holds the line: the known debt is enumerated, and a *new* secret-shaped
key reaching this path fails here rather than in production.

The Windows bootstrap already refuses outright; it has no legacy callers.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "blueprint_pipeline"

_SECRET_FRAGMENTS = ("password", "secret", "token", "private_key", "credential")

#: Known debt, with the lane that owns it. Shrink this list; never grow it.
#: Each entry is (module, env key).
KNOWN_USER_DATA_SECRET_DEBT = {
    ("vast_provider_adapter.py", "HF_TOKEN"),
    ("vast_provider_adapter.py", "HUGGING_FACE_HUB_TOKEN"),
    ("isaac_g1_kitchen_parity_job.py", "BLUEPRINT_WARM_RENDER_BROKER_TOKEN"),
}

_ENV_ASSIGN = re.compile(r"""env\[\s*["']([A-Z0-9_]+)["']\s*\]\s*=""")


def _observed_secret_env_assignments() -> set[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    for path in sorted(SRC.glob("*.py")):
        try:
            source = path.read_text(encoding="utf-8")
        except OSError:
            continue
        for key in _ENV_ASSIGN.findall(source):
            if any(fragment in key.lower() for fragment in _SECRET_FRAGMENTS):
                found.add((path.name, key))
    return found


def test_no_new_secret_shaped_key_is_assigned_into_worker_env() -> None:
    observed = _observed_secret_env_assignments()
    new = observed - KNOWN_USER_DATA_SECRET_DEBT
    assert not new, (
        "New secret-shaped key(s) assigned into a worker env mapping: "
        f"{sorted(new)}. VM UserData is readable via IMDS and "
        "DescribeInstanceAttribute. Use a signed fetch-then-delete like the "
        "Postshot licence instead of putting the value in env."
    )


def test_the_debt_list_does_not_outlive_the_debt() -> None:
    """Once a lane is fixed, its entry must be removed, not left to rot."""
    observed = _observed_secret_env_assignments()
    stale = KNOWN_USER_DATA_SECRET_DEBT - observed
    assert not stale, (
        f"These entries no longer exist and should be deleted: {sorted(stale)}"
    )


def test_the_windows_bootstrap_has_no_such_debt() -> None:
    """It refuses secret-shaped keys outright, having no legacy callers."""
    source = (SRC / "cloud_vm_render_providers.py").read_text(encoding="utf-8")
    assert "windows_worker_bootstrap_refuses_credential_in_user_data" in source
