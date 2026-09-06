import json
from pathlib import Path
from tempfile import TemporaryDirectory
from pytest import MonkeyPatch
from tests.test_scene_preparation_installation import _installed
from blueprint_pipeline import task_evaluation_scene_preparation_installation as si
from blueprint_pipeline import task_evaluation_scene_progression as e
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
with TemporaryDirectory(prefix='upload-activation-audit-') as d, MonkeyPatch.context() as mp:
 root=Path(d).resolve(); bp,_,_,_=_installed(root,mp)
 b=json.loads(bp.read_text()); b['activation_authorized']=True; b['bootstrap_digest']=canonical_digest(b,digest_field='bootstrap_digest'); bp.write_text(json.dumps(b))
 installed=si.install_scene_preparation(bootstrap_path=bp)
 print(json.dumps(e.process_scene_intents(config_path=installed['config']['path']),indent=2))
