"""ADP-009D/day-21 local-artifact rehearsal, stopping before paid placement.

Run with the controls worker's environment and service identity. Reads retained
scene/robot inputs (including artifact-store GETs), writes only a fresh temporary
directory, and never installs a registry, reserves model spend or launches a GPU.
This is a software check, not an evaluation receipt or a replacement controller.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

from blueprint_pipeline.task_evaluation_configured_controls_autostart import materialize_configured_controls_autostart


class PaidBoundaryReached(RuntimeError):
    pass


def _stop_before_paid_gate(**_kwargs):
    raise PaidBoundaryReached('cpu_placement_complete_before_openai_reservation')


def rehearse(*, source_launch_id: str, launch_state_root: str, intent: str) -> dict:
    root = Path(tempfile.mkdtemp(prefix='blueprint-evaluation-cpu-rehearsal-'))
    try:
        materialize_configured_controls_autostart(source_launch_id=source_launch_id,
            launch_state_root=launch_state_root, intent_path_override=intent,
            progression_root=root/'progression', plan_root=root/'plans',
            openai_gate_builder=_stop_before_paid_gate,
            openai_scope_lock=lambda **_: nullcontext())
    except PaidBoundaryReached:
        return {'status':'cpu_paid_boundary_reached', 'rehearsal_root':str(root),
                'model_called':False, 'provider_allocated':False, 'production_evidence':False}
    raise RuntimeError('cpu_rehearsal_paid_boundary_not_reached')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-launch-id', required=True)
    parser.add_argument('--launch-state-root', required=True)
    parser.add_argument('--intent', required=True)
    args = parser.parse_args()
    print(json.dumps(rehearse(**vars(args)), sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
