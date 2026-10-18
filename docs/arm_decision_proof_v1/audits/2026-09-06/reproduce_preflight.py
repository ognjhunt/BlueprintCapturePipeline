from pathlib import Path
from tempfile import TemporaryDirectory
from tests.test_owner_scope_preflight import _controls_config, _wired_units, IDS
from blueprint_pipeline.task_evaluation_production_chain_preflight import owner_scope_checks
from blueprint_pipeline.task_evaluation_controls_autoprovision import process_config

with TemporaryDirectory() as temporary:
    root = Path(temporary).resolve()
    config = _controls_config(root)
    findings = owner_scope_checks(_wired_units(root, config), IDS)
    print("preflight_blockers", [f for f in findings if f.get("severity") == "blocker"])
    try:
        process_config(config, expected_production_commit="a" * 40)
    except ValueError as error:
        print("real_consumer_refusal", str(error))
