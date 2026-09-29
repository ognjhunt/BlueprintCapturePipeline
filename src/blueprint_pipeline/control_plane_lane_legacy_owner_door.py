"""Bounded, report-only operator-door publication of current legacy owner labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import signal
import time
from pathlib import Path

from . import control_plane_lane_legacy_owner as legacy
from . import control_plane_lane_owner_consents as owners

_ID = re.compile(r"[0-9]{8}T[0-9]{6}Z-legacy-owner-census-[0-9a-f]{8}\Z")
_MAX_RESULT = 512 * 1024
_MAX_ROWS = 1024
_WALL_SECONDS = 270


def _require(value: bool) -> None:
    if not value:
        raise legacy.LegacyOwnerError("legacy_owner_census_incomplete")


def _public_report(observed: dict) -> dict:
    """Copy only current owner labels, never full process, env, or census evidence."""
    _require(isinstance(observed, dict))
    if observed.get("status") != "complete":
        return dict(schema_version="control_plane_lane_legacy_owner_public.v1",
                    status="incomplete", rows=[], observed_owner_count=0,
                    gc_eligible=False, references_clear=False,
                    candidate_bytes=None, eta_seconds=None, mutations=0)
    source = observed.get("rows")
    _require(isinstance(source, list) and observed.get("scan_errors") == [])
    selected = [row for row in source if isinstance(row, dict)
                and row.get("classification") == "legacy_owner_review"]
    if len(source) > 200_000 or len(selected) > _MAX_ROWS:
        return dict(schema_version="control_plane_lane_legacy_owner_public.v1",
                    status="incomplete", rows=[], observed_owner_count=0,
                    gc_eligible=False, references_clear=False,
                    candidate_bytes=None, eta_seconds=None, mutations=0)
    rows = []
    for row in selected:
        _require(isinstance(row.get("path"), str) and isinstance(row.get("owner"), str)
                 and type(row.get("approved_expiry")) in (int, float)
                 and row.get("gc_eligible") is False and row.get("references_clear") is False
                 and row.get("references") == [] and row.get("unreadable") == 0)
        rows.append(dict(path=row["path"], owner=row["owner"],
                         approved_expiry=row["approved_expiry"],
                         classification="legacy_owner_review", gc_eligible=False,
                         references_clear=False, candidate_bytes=None,
                         eta_seconds=None))
    _require(len(rows) == observed.get("observed_owner_count"))
    return dict(schema_version="control_plane_lane_legacy_owner_public.v1",
                status="complete", rows=rows, observed_owner_count=len(rows),
                gc_eligible=False, references_clear=False,
                candidate_bytes=None, eta_seconds=None, mutations=0)


def publish_current(*, installed_config_path: str, results_dir: str,
                    request_id: str, now: float | None = None) -> dict:
    _require(isinstance(request_id, str) and _ID.fullmatch(request_id) is not None)
    observed = legacy.observe_owner_review(installed_config_path=installed_config_path,
                                           now=time.time() if now is None else now,
                                           max_seconds=240)
    result = _public_report(observed)
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if len(payload) > _MAX_RESULT:
        result = _public_report({"status": "incomplete"})
        payload = json.dumps(result, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with legacy._installed_session(installed_config_path, time.monotonic) as (files, _, config, _):
        _require(results_dir == str(Path(config.spool_root) / "results"))
        result_path = Path(results_dir) / (request_id + ".legacy-owner-census.json")
        parent, name = owners._public_parent(files, config, result_path)
        owners._publish(files, parent, name, payload, mode=0o644, immutable=False)
        summary = dict(schema="blueprint_operator_door_outcome.v1",
                       status="legacy_owner_census_observed" if result["status"] == "complete" else "incomplete",
                       code=None if result["status"] == "complete" else "legacy_owner_census_incomplete",
                       exit_code=0, result=str(result_path),
                       result_sha256="sha256:" + hashlib.sha256(payload).hexdigest(),
                       result_size_bytes=len(payload),
                       observed_owner_count=result["observed_owner_count"],
                       gc_eligible=False, references_clear=False,
                       candidate_bytes=None, eta_seconds=None, mutations=0)
        outcome = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode("utf-8")
        _require(len(outcome) <= 8192)
        parent, name = owners._public_parent(files, config, Path(results_dir) / (request_id + ".outcome.json"))
        owners._publish(files, parent, name, outcome, mode=0o644, immutable=False)
    return summary


def main(argv: list[str] | None = None) -> int:
    class Parser(argparse.ArgumentParser):
        def error(self, _message):
            raise legacy.LegacyOwnerError("legacy_owner_options_invalid")

    parser = Parser(allow_abbrev=False)
    parser.add_argument("mode", choices=("report",))
    parser.add_argument("--door-config", required=True)
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--request-id", required=True)
    def expired(_signum, _frame):
        raise legacy.LegacyOwnerError("legacy_owner_budget_exhausted")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, _WALL_SECONDS)
    try:
        args = parser.parse_args(argv)
        publish_current(installed_config_path=args.door_config,
                        results_dir=args.results_dir, request_id=args.request_id)
        return 0
    except (legacy.LegacyOwnerError, owners.OwnerCensusConsentError,
            OSError, TypeError, ValueError, UnicodeError):
        # Door logs are owner-readable. Never echo protected paths, values, or env.
        print('{"status":"refused","code":"legacy_owner_census_incomplete"}')
        return 1
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
