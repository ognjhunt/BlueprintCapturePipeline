"""Start the actual intake app against a dedicated local synthetic work root."""
import json
import os
import sys
from pathlib import Path

import uvicorn

config = json.loads(Path(sys.argv[1]).read_text())
root = Path(config["proof_root"]).resolve()
os.environ["BLUEPRINT_LIVE_PIPELINE_CLIENT_SECRETS_JSON"] = json.dumps({"blueprint-webapp": config["pipeline_token"]})
os.environ["BLUEPRINT_COMPANY_POLICY_ALLOWED_REGISTRIES"] = config["registry_host"]
os.environ["BLUEPRINT_COMPANY_POLICY_CONTAINER_ADMISSION_ROOT"] = str(root / "admissions")
os.environ["BLUEPRINT_LIVE_PIPELINE_NONCE_STORE_DIR"] = str(root / "nonces")
os.environ["BLUEPRINT_LIVE_PIPELINE_INTAKE_WORK_DIR"] = str(root / "intake")
from blueprint_pipeline.live_pipeline_intake_service import app
uvicorn.run(app, host="127.0.0.1", port=8801, access_log=False)
