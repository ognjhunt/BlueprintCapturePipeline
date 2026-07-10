from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNBOOK = REPO_ROOT / "docs" / "runbooks" / "beta-ops-incident-response.md"
DEPLOY_SCRIPT = REPO_ROOT / "deploy" / "scripts" / "deploy.sh"


def test_beta_ops_incident_runbook_covers_cross_repo_response() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")

    assert "Primary owner: `blueprint-cto`" in text
    assert "Pipeline owner: `pipeline-oncall`" in text
    assert "Escalation owner: Founder/CEO" in text
    assert "../Blueprint-WebApp/docs/beta-ops-incident-runbook-2026-07-08.md" in text
    assert "npm run deploy:rollback" in text
    assert "deploy/scripts/deploy.sh --rollback --rollback-image-tag" in text
    assert "Takedown and access freeze" in text
    assert "Customer communications" in text
    assert "Rollback evidence required before closeout" in text


def test_pipeline_deploy_script_has_health_checked_rollback_mode() -> None:
    text = DEPLOY_SCRIPT.read_text(encoding="utf-8")

    assert "--rollback-image-tag" in text
    assert "ROLLBACK_VERIFY_COMMAND" in text
    assert "ROLLBACK_HEALTH_CHECK" in text
    assert "rollback_deployment()" in text
    assert "gcloud run jobs update blueprint-pipeline" in text
    assert "gcloud run jobs describe blueprint-pipeline" in text
    assert "jq -r '.template.template.containers[0].image // \"\"'" in text
    assert 'IMAGE_TAG="${IMAGE_TAG:-}"' in text
    assert "validate_release_image_tag()" in text
    assert "latest|dev|test|local" in text
    assert "pin_pushed_image_digests()" in text
    assert "gcloud container images describe" in text
    assert "image_summary.fully_qualified_digest" in text
    assert "DEPLOYMENT_MANIFEST_PATH" in text
    assert "write_deployment_manifest()" in text
    assert "blueprint.pipeline_deployment_manifest.v1" in text
    assert "verify_deploy_release_provenance.py" in text
    assert "run_deployment_service_canaries.py" in text
    assert "Authenticated deployment service canaries did not pass." in text
    assert "Provider-refreshed Terraform topology plus authenticated no-op service canaries" in text
    assert "reset --hard" not in text

    subprocess.run(["bash", "-n", str(DEPLOY_SCRIPT)], check=True)
