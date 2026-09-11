"""Native error monitoring must start before scene import and retain failures."""
from types import SimpleNamespace

import pytest

from blueprint_pipeline.task_object_native_settle_runtime import NativeAssetPhysicsMonitor


def test_native_monitor_pumps_and_retains_physx_errors():
    class Stream:
        def create_subscription_to_pop(self, callback, *, name):
            self.callback = callback
            assert name == 'blueprint_astra_native_asset_feasibility'
            return object()
        def pump(self):
            if pending:
                self.callback(pending.pop())
    pending = []
    stream = Stream()
    monitor = NativeAssetPhysicsMonitor(SimpleNamespace(get_error_event_stream=lambda: stream))
    assert monitor.read()['monitoring_active'] is True
    assert monitor.read()['errors'] == []
    pending.append(SimpleNamespace(type=2, payload={'errorString': 'fixture convex cooking failed'}))
    assert monitor.read()['errors'] == [{'event_type': 2, 'message': 'fixture convex cooking failed'}]
    assert monitor.read()['errors'] == [{'event_type': 2, 'message': 'fixture convex cooking failed'}]


def test_absent_native_subscription_refuses():
    stream = SimpleNamespace(create_subscription_to_pop=lambda *args, **kwargs: None)
    with pytest.raises(RuntimeError, match='error_monitor_unavailable'):
        NativeAssetPhysicsMonitor(SimpleNamespace(get_error_event_stream=lambda: stream))
