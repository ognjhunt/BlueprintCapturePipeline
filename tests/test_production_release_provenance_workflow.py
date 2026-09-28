from pathlib import Path
import re

import yaml


def test_provenance_is_verified_in_ci_with_read_only_ephemeral_authority() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / ".github/workflows/production-release-provenance.yml").read_text()
    workflow = yaml.load(source, Loader=yaml.BaseLoader)
    assert workflow["permissions"] == {"contents": "read", "actions": "read"}
    assert workflow["on"]["workflow_run"]["workflows"] == ["Full Test Lane"]
    assert workflow["on"]["workflow_run"]["branches"] == ["main"]
    job = workflow["jobs"]["verify"]
    assert job["env"]["GH_TOKEN"] == "${{ github.token }}"
    assert "secrets." not in source
    steps = job["steps"]
    guard, checkout = steps[:2]
    assert guard["id"] == "promotion"
    assert '.head_branch == "main"' in guard["run"]
    assert '.status == "completed" and .conclusion == "success"' in guard["run"]
    assert 'test "${promoted_sha}" = "${current_main}"' in guard["run"]
    assert '.display_title == "Full Test Lane / production_deployment_promotion"' in guard["run"]
    assert checkout["with"]["persist-credentials"] == "false"
    assert checkout["with"]["ref"] == "${{ steps.promotion.outputs.sha }}"
    for step in steps[1:]:
        assert step["if"] == "steps.promotion.outputs.eligible == 'true'"
        if "uses" in step:
            assert re.search(r"@[0-9a-f]{40}$", step["uses"])
    verification = steps[3]["run"]
    assert "scripts/verify_deploy_release_provenance.py" in verification
    assert "--expected-sha" in verification and "--run-url" in verification
    assert steps[-1]["with"]["if-no-files-found"] == "error"
    assert "paid_resource_allocator" not in source
