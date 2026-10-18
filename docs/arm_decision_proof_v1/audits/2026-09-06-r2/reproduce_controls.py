import json, os
from pathlib import Path
from tempfile import TemporaryDirectory
from pytest import MonkeyPatch
from tests.test_task_evaluation_controls_autoprovision import _configured_scene
from tests.test_task_evaluation_controls_autoprovision_installation import _install
from tests.test_owner_scope_preflight import _wired_units, IDS
from blueprint_pipeline import task_evaluation_controls_autoprovision as w
from blueprint_pipeline import task_evaluation_production_chain_preflight as p
from tests.test_task_evaluation_configured_controls_continuation_provisioning import COMMIT
from tests.test_task_evaluation_scene_intake import request
from blueprint_pipeline import task_evaluation_scene_intake as intake
from tests.test_scene_preparation_installation import _installed
from blueprint_pipeline import task_evaluation_scene_preparation_installation as si
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
report={}
with TemporaryDirectory(prefix='controls-audit-') as d, MonkeyPatch.context() as mp:
 root=Path(d).resolve(); mp.setenv(intake.CLIENTS_ENV,'webapp')
 args=_configured_scene(root/'controls'); etc=root/'etc'; etc.mkdir()
 installed=_install(root,args,etc)
 config=Path(installed['config']['path']); catalog=json.loads(Path(installed['catalog']['path']).read_text())
 report['actual_installer_catalog_schema']=catalog['schema_version']
 report['actual_consumer_resolves']=w.resolve_robot_catalog(catalog,source_commit=COMMIT)['schema_version']
 report['preflight_for_real_installer']=[x['code'] for x in p.owner_scope_checks(_wired_units(root,config),IDS) if x['severity']=='blocker']
with TemporaryDirectory(prefix='activation-audit-') as d, MonkeyPatch.context() as mp:
 root=Path(d).resolve(); bp,_,_,_=_installed(root,mp)
 bootstrap=json.loads(bp.read_text()); bootstrap['activation_authorized']=True
 bootstrap['bootstrap_digest']=canonical_digest(bootstrap,digest_field='bootstrap_digest'); bp.write_text(json.dumps(bootstrap))
 installed=si.install_scene_preparation(bootstrap_path=bp); config=json.loads(Path(installed['config']['path']).read_text())
 env=Path(installed['environment']['path']).read_text()
 unit=Path('deploy/systemd/blueprint-task-evaluation-configured-controls-progression.service').read_text()
 report['activation_producer_root']=config['activation_intent_root']
 report['activation_env_override_exported']='BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_ACTIVATION_INTENT_ROOT=' in env
 report['activation_consumer_unit_root']=next(line for line in unit.splitlines() if line.startswith('Environment=BLUEPRINT_TASK_EVALUATION_SCENE_CONFIGURATION_ACTIVATION_INTENT_ROOT='))
value=request(); report['intake_accepts_arbitrary_checkpoint_digests']=intake.validate_request(value,now=100)['execution']['policy_candidates']
print(json.dumps(report,indent=2))
