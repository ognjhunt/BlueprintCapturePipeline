"""Isaac bindings for the separate new-asset placement feasibility probe."""
from __future__ import annotations

from pathlib import Path
from typing import Any


class NativeAssetPhysicsMonitor:
    """Subscribe before asset import; retain every native PhysX error.

    NVIDIA documents this error stream and its subscription lifecycle at
    https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/extensions/runtime/source/omni.physx/docs/api/python.html
    and the Carbonite IEventStream API. No global logging settings are changed.
    """

    def __init__(self, physx_interface: Any = None):
        if physx_interface is None:
            import omni.physx
            physx_interface = omni.physx.get_physx_interface()
        self.errors: list[dict[str, Any]] = []
        self._stream = physx_interface.get_error_event_stream()
        self._subscription = self._stream.create_subscription_to_pop(
            self._on_error, name='blueprint_astra_native_asset_feasibility')
        if self._subscription is None:
            raise RuntimeError('native_asset_physx_error_monitor_unavailable')

    def _on_error(self, event):
        payload = dict(event.payload)
        self.errors.append({'event_type': int(event.type),
                            'message': str(payload.get('errorString', payload))[:4096]})

    def read(self) -> dict[str, Any]:
        self._stream.pump()
        return {'monitoring_active': self._subscription is not None,
                'source': 'native_physx_error_event_stream_before_asset_import',
                'errors': list(self.errors)}


def run_isaac_asset_feasibility(*, built, plan, monitor, output_root: Path):
    from .native_task_arena_construction_worker import _camera_snapshot
    from .task_object_native_settle_gate import make_native_settle_adapter, run_native_settle_gate
    adoption = plan['task_spec']['astra_asset_adoption']

    def frames(label, root):
        return _camera_snapshot(env=built.env, camera_scene_names=built.camera_scene_names,
            output_root=root, snapshot_id=label,
            framing_expectations=(plan.get('task_object_observability') or {}).get('cameras'))

    adapter = make_native_settle_adapter(built=built, capture_frames=frames,
        read_cooking_errors=monitor.read)
    return run_native_settle_gate(spec=adoption['native_settle_spec'], adapter=adapter,
        output_root=output_root, seed=int(plan['scenario']['seed']),
        scene_plan_digest=plan['plan_digest'], adoption_digest=adoption['adoption_digest'], reset=True)
