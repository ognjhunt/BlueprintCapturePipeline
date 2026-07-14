from pathlib import Path


WORKFLOW = Path(".github/workflows/groot-oscar-release-image.yml")


def test_release_workflow_is_serialized_and_excludes_docs_only_changes():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "cancel-in-progress: false" in text
    assert "build-scan-once" in text
    assert "docs/**" not in text
    assert "src/blueprint_pipeline/gpu_campaign_state_machine.py" not in text
    assert "src/blueprint_pipeline/groot_oscar_release_hardening.py" in text
    assert "src/blueprint_pipeline/gear_sonic_official_zmq_executor.py" in text
    assert "scripts/build_push_groot_oscar_closed_loop_image.sh" in text
    assert 'BLUEPRINT_ALLOW_GROOT_OSCAR_CLOSED_LOOP_IMAGE_PUSH: "true"' in text


def test_release_workflow_uses_ephemeral_file_credentials_and_always_cleans_them():
    text = WORKFLOW.read_text(encoding="utf-8")
    assert "--password-stdin" in text
    assert "BLUEPRINT_GROOT_OSCAR_CLOSED_LOOP_HF_TOKEN_FILE" in text
    assert "if: always()" in text
    assert 'rm -rf "$RUNNER_TEMP/blueprint-secrets" "$RUNNER_TEMP/blueprint-docker"' in text
