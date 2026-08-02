#!/usr/bin/env python3
"""Run one measurement-research monitoring pass and write the report.

Diffs release observations against the research intake catalog and the R1
engine capability profiles, emitting version-change alerts, R0 intake drafts,
regression-check flags, and requalification-trigger *proposals*. Nothing is
approved, admitted, or suspended by this script: every proposal requires a
human to act through the R0-R8 admission machinery.

Observations come from a JSON file (``--observations``) and/or live GitHub
lookups (``--fetch-github method_id=owner/name``, repeatable). Admitted
R7/R8 admission records may be supplied (``--admissions``) so version changes
propose the matching requalification triggers.

Intended cadence: monthly (for example from a scheduler invoking
``python scripts/measurement_research_monitor.py --fetch-github
mujoco-3=google-deepmind/mujoco ... --output output/monitoring/<date>.json``).
This repository change deliberately installs no scheduler entry.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from blueprint_pipeline.measurement_research_monitoring import (  # noqa: E402
    build_release_observation,
    compile_research_monitoring_report,
    github_latest_release_observation,
)


def _load_rows(path: str | None) -> list[dict]:
    if not path:
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"expected a JSON list in {path}")
    return [dict(row) for row in payload]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", help="JSON list of release observations")
    parser.add_argument(
        "--fetch-github",
        action="append",
        default=[],
        metavar="METHOD_ID=OWNER/NAME",
        help="fetch the latest GitHub release for a method (repeatable)",
    )
    parser.add_argument("--admissions", help="JSON list of admission records")
    parser.add_argument("--output", help="path for the monitoring report JSON")
    parser.add_argument(
        "--observed-on",
        default=_datetime.date.today().isoformat(),
        help="observation date recorded in the report (default: today)",
    )
    arguments = parser.parse_args()

    observations = [
        build_release_observation(
            method_id=str(row.get("method_id", "")),
            observed_version=str(row.get("observed_version", "")),
            observed_release_date=str(row.get("observed_release_date", "")),
            source_reference=str(row.get("source_reference", "")),
            observed_on=str(row.get("observed_on") or arguments.observed_on),
            notes=str(row.get("notes", "")),
        )
        for row in _load_rows(arguments.observations)
    ]
    for pair in arguments.fetch_github:
        method_id, _, repository = pair.partition("=")
        if not method_id or "/" not in repository:
            raise SystemExit(f"invalid --fetch-github value: {pair}")
        observations.append(
            github_latest_release_observation(
                method_id=method_id,
                repository=repository,
                observed_on=arguments.observed_on,
            )
        )
    if not observations:
        raise SystemExit("no observations supplied; use --observations or --fetch-github")

    report = compile_research_monitoring_report(
        observations,
        observed_on=arguments.observed_on,
        admission_records=_load_rows(arguments.admissions),
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if arguments.output:
        output_path = Path(arguments.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    changed = [row for row in report["alerts"] if row["kind"] == "version_changed"]
    fresh = [row for row in report["alerts"] if row["kind"] == "new_method_discovered"]
    print(
        f"monitoring: {len(report['alerts'])} alert(s), "
        f"{len(changed)} version change(s), {len(fresh)} new method(s), "
        f"{len(report['requalification_trigger_proposals'])} trigger proposal(s); "
        "human action required for every proposal",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
