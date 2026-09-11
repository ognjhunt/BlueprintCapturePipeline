"""Strict clock double using method bodies extracted from the pinned source."""
import json
from pathlib import Path
from types import SimpleNamespace

API = json.loads((Path(__file__).parent / 'fixtures/native_task_combined_bootstrap_v28d/pinned_simulation_context_api.json').read_text())


class PinnedSimulationClock:
    __slots__ = ('_physics_step_count', 'physics_manager')

    def __init__(self, dt=1. / 120.):
        self._physics_step_count = 0
        self.physics_manager = SimpleNamespace(get_physics_dt=lambda: dt)

    def render(self):
        pass


for name, source in API['public_clock_methods'].items():
    namespace = {}
    exec(compile(source, API['source_relative_path'], 'exec'), namespace)
    setattr(PinnedSimulationClock, name, namespace[name])

assert 'current_time' not in API['class_attribute_names']
assert 'current_time_step_index' not in API['class_attribute_names']
