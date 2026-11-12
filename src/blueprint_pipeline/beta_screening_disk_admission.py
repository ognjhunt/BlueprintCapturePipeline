"""Room on the host to land a test's result: asked before the test is paid for, and held until it lands.

Beta journey step: a site's paid AI robot test (beta screening). A finished test's output is landed on the
host under the attempt's directory: the provider zip is downloaded (``provider_output_disk_capacity``),
then extracted under the shared disk ledger with an exact reservation of its expanded size (role
``policy_canary_output``, ``adp_isaac_lab_arena_vast._extract_provider_output``). Below the work disk's
floor that landing was refused after the GPU had been rented and paid for.

Two parts:

- ``work_disk_admission``, the coordinator's pre-launch check (``beta_policy_screening_launch``): would
  the landing hold fit on the attempt's filesystem now? It reserves and at once releases. A launch that
  would not fit waits in line (the coordinator's capacity wait): nothing is rented or spent, the row is
  tried again when room returns, and staff get one alert while any row waits (``alert_disk_waits``).
- ``reserve_landing_hold``, taken by the test's own session process before its paid authority is
  consumed (``native_task_arena_vast``) and released on every exit path. While it is held its token is
  exported as an enclosing reservation (``enclosing``), so the extraction's own exact reservation inside
  it does not count it a second time. Every other writer that uses the ledger counts it, so the room it
  holds stays free for the landing from admission until the output has landed or the run has ended.

The room is the room the row's own landing path needs (measured read-only on the host, 2026-11-12, the 47
website policy-canary runs among the last 91 screening attempts):

- streamed (every one of the 47: ``auto_needed_set_within_budget``): only the needed members land. Largest
  0.275 GiB, 95th percentile 0.274 GiB (27 measured); the stream refuses more than its own budget
  (1,016,004,608 bytes). The session already takes the 1 GiB forecast hold for it
  (``arena_provider_output_streaming.reserve_forecast_hold``), so the check asks for exactly that.
- non-streamed (only when the delivery resolves to download: an operator setting, no artifact store, or no
  needed-set record; none of the 47): the provider archive is downloaded and expanded. Its largest recorded
  size is 5,088,693,294 bytes (4.74 GiB; 95th percentile 3.57 GiB); 1.25 x that, never above the lane's own
  upload cap (8 GB, ``expected_provider_upload_bytes``, which the download enforces), twice for zip and
  expansion: 11.85 GiB. Held by the session (``reserve_landing_hold``) only on that path.

The G1 lane and the task-family lane keep their own collection holds; this module adds none for them.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

ROLE = "policy_canary_output"
WORK_DISK_WAIT_BLOCKER = "beta_screening_work_disk_wait"
ALERT_CODE = "beta_screening_work_disk_wait"
ALERT_STATE_FILENAME = "work_disk_wait_alert.json"
#: The attempt field the coordinator sets while a row waits for disk (when the wait began).
WAIT_SINCE_FIELD = "work_disk_wait_since"
#: How long a row waits for room: the screening pass's own bound before a stalled source is held (6 h). Past it
#: the row is held by name (``CANNOT_LAND_BLOCKER``), its reservation released, and staff alerted once for it.
WAIT_LIMIT_SECONDS = 6 * 3600
CANNOT_LAND_BLOCKER = "beta_screening_output_cannot_land_on_work_disk"
CANNOT_LAND_ALERT = "beta_screening_output_cannot_land_on_work_disk"
RESERVATION_ROOT_ENV = "BLUEPRINT_CONTROL_PLANE_DISK_RESERVATION_ROOT"
LANDING_HOLD_ENV = "BLUEPRINT_SCREENING_LANDING_HOLD_BYTES"
#: The largest provider archive among the website's policy-canary runs on the host (see the module docstring).
LARGEST_RECORDED_ARCHIVE_BYTES = 5_088_693_294
LANDING_MARGIN = 1.25
#: The lane's own upload cap: no archive larger than this is downloaded.
UPLOAD_CAP_BYTES = 8_000_000_000
DEFAULT_LANDING_HOLD_BYTES = 2 * min(math.ceil(LANDING_MARGIN * LARGEST_RECORDED_ARCHIVE_BYTES), UPLOAD_CAP_BYTES)
WORKLOAD = "screening_output_landing"
#: How the filesystem is measured; a test replaces it.
measure_disk = shutil.disk_usage


def streamed_hold_bytes() -> int:
    """The room a streamed row's landing holds: the session's own forecast hold."""
    from .policy_canary_output_members import forecast_hold_cap_bytes

    return forecast_hold_cap_bytes()


def landing_hold_bytes(environ: Mapping[str, str] | None = None) -> int:
    """The room a non-streamed landing may need: the zip and its expansion (see the module docstring)."""
    raw = str((os.environ if environ is None else environ).get(LANDING_HOLD_ENV) or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else DEFAULT_LANDING_HOLD_BYTES


def _root(reservation_root: str | Path | None) -> str | Path:
    from .control_plane_disk_budget import DEFAULT_RESERVATION_ROOT

    return reservation_root or os.environ.get(RESERVATION_ROOT_ENV) or DEFAULT_RESERVATION_ROOT


def reserve_landing_hold(job_dir: str | Path, *, reservation_root: str | Path | None = None,
                         disk_usage: Callable[[Any], Any] | None = None, expected_bytes: int | None = None):
    """The landing hold, held by the caller's process until it releases it. Raises ``ControlPlaneDiskBudgetError``."""
    from .control_plane_disk_budget import reserve_control_plane_disk

    return reserve_control_plane_disk(ROLE, target_root=Path(job_dir),
                                      expected_bytes=expected_bytes or landing_hold_bytes(),
                                      reservation_root=_root(reservation_root), workload=WORKLOAD,
                                      disk_usage=disk_usage or measure_disk)


@contextlib.contextmanager
def enclosing(token: str) -> Iterator[None]:
    """Export ``token`` as an enclosing reservation for the landing inside it; restored on exit."""
    from .control_plane_disk_budget import ENCLOSING_RESERVATIONS_ENV

    previous = os.environ.get(ENCLOSING_RESERVATIONS_ENV)
    os.environ[ENCLOSING_RESERVATIONS_ENV] = ",".join(filter(None, (previous, token)))
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(ENCLOSING_RESERVATIONS_ENV, None)
        else:
            os.environ[ENCLOSING_RESERVATIONS_ENV] = previous


def work_disk_admission(attempt_dir: str | Path, *, streamed: bool = True, reservation_root: str | Path | None = None,
                        disk_usage: Callable[[Any], Any] | None = None) -> dict[str, Any]:
    """``{"admitted": bool, ...}``: whether the row's landing hold (by its path) would be admitted now."""
    from .control_plane_disk_ledger import ControlPlaneDiskBudgetError

    size = streamed_hold_bytes() if streamed else landing_hold_bytes()
    try:
        with reserve_landing_hold(attempt_dir, reservation_root=reservation_root, disk_usage=disk_usage,
                                  expected_bytes=size) as hold:
            receipt = hold.receipt()
    except ControlPlaneDiskBudgetError as exc:
        return {"admitted": False, "role": ROLE, "streamed": streamed, "expected_bytes": size, "reason": str(exc)[:200]}
    return {"admitted": True, "role": ROLE, "streamed": streamed,
            **{key: receipt.get(key) for key in ("expected_bytes", "footprint_basis",
                                                   "available_bytes_before_admission")}}


def alert_disk_waits(queue_root: str | Path, *, webhook_url: str,
                     poster: Callable[[str, Mapping[str, Any]], None] | None = None) -> dict[str, Any]:
    """One alert while any row waits for disk: how many, and since when. Cleared once none waits.

    Reads the screening ledger only. A new wait after every row stopped waiting is a new alert; a failed
    post is tried again on the next pass. Never raises for the post.
    """
    root = Path(queue_root)
    try:
        ledger = json.loads((root / "screening_ledger.v1.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"status": "ledger_unreadable"}
    since = sorted(str(row["attempt"][WAIT_SINCE_FIELD]) for state in (ledger.get("campaigns") or {}).values()
                   for row in (state.get("candidates") or {}).values()
                   if row.get("status") == "preparing" and isinstance(row.get("attempt"), Mapping)
                   and row["attempt"].get(WAIT_SINCE_FIELD))
    state_path = root / ALERT_STATE_FILENAME
    try:
        previous = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        previous = {}
    cannot_land = _alert_cannot_land(root, ledger, webhook_url=webhook_url, poster=poster)
    if not since:
        with contextlib.suppress(OSError):
            state_path.unlink()
        return {"status": "none_waiting", "waiting": 0, "cannot_land": cannot_land}
    record = {"waiting": len(since), "since": since[0], "posted": bool(previous.get("posted")),
              "episode_since": previous.get("episode_since") or since[0], "last_error": None}
    if not record["posted"]:
        if not webhook_url:
            record["last_error"] = "operator_alert_route_unconfigured"
        else:
            from . import control_plane_capacity_controller as capacity

            alert = {"severity": "page", "code": ALERT_CODE, "mount": "beta screening work disk",
                     "rows_waiting": len(since), "waiting_since": since[0]}
            try:
                (poster or capacity.post_alert)(webhook_url, {"level": "critical", "mounts": [], "alerts": [alert]})
                record["posted"] = True
            except Exception as error:  # noqa: BLE001 - the message may carry the URL; tried again next pass
                record["last_error"] = type(error).__name__
    with contextlib.suppress(OSError):
        state_path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")
    return {"status": "waiting", **record, "cannot_land": cannot_land}


def _alert_cannot_land(root: Path, ledger: Mapping[str, Any], *, webhook_url: str,
                       poster: Callable[[str, Mapping[str, Any]], None] | None) -> dict[str, Any]:
    """One alert per row whose output cannot land on this disk (``CANNOT_LAND_BLOCKER``); never raises."""
    rows = sorted(f"{campaign}:{key}" for campaign, state in (ledger.get("campaigns") or {}).items()
                  for key, row in (state.get("candidates") or {}).items()
                  if row.get("status") == "blocked" and row.get("blockers") == [CANNOT_LAND_BLOCKER])
    path = root / "work_disk_cannot_land_alerts.json"
    try:
        posted = set(json.loads(path.read_text(encoding="utf-8")).get("posted") or [])
    except (OSError, ValueError):
        posted = set()
    new = [row for row in rows if row not in posted]
    if not new or not webhook_url:
        return {"rows": rows, "posted_now": [], "last_error": None if not new else "operator_alert_route_unconfigured"}
    from . import control_plane_capacity_controller as capacity

    alerts = [{"severity": "page", "code": CANNOT_LAND_ALERT, "mount": "beta screening work disk", "row": row,
               "message": "this test's output cannot land on the work disk: held, its reservation released"}
              for row in new]
    try:
        (poster or capacity.post_alert)(webhook_url, {"level": "critical", "mounts": [], "alerts": alerts})
    except Exception as error:  # noqa: BLE001 - tried again next pass
        return {"rows": rows, "posted_now": [], "last_error": type(error).__name__}
    with contextlib.suppress(OSError):
        path.write_text(json.dumps({"posted": sorted(posted | set(new))}) + "\n", encoding="utf-8")
    return {"rows": rows, "posted_now": new, "last_error": None}


__all__ = ["ALERT_CODE", "CANNOT_LAND_BLOCKER", "WAIT_LIMIT_SECONDS", "DEFAULT_LANDING_HOLD_BYTES", "LANDING_HOLD_ENV", "ROLE", "WAIT_SINCE_FIELD",
           "WORK_DISK_WAIT_BLOCKER", "alert_disk_waits", "enclosing", "landing_hold_bytes", "reserve_landing_hold",
           "work_disk_admission"]
