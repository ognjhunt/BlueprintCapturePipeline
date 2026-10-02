"""The legacy AWS CLI remains an inert, locally refusing entrypoint."""
import importlib.util
from pathlib import Path


def test_legacy_worker_has_no_sdk_or_credential_setup(capsys):
    path = Path(__file__).parents[1] / "scripts/postshot_windows_worker/launch_postshot_worker.py"
    spec = importlib.util.spec_from_file_location("retired_postshot", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    assert worker.main() == 2
    assert "aws_provider_integration_removed" in capsys.readouterr().out
    assert "boto3" not in path.read_text()
