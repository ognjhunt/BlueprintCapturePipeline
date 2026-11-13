from __future__ import annotations

import os
import base64
import gzip
import re
import shlex
import subprocess

import blueprint_pipeline.vast_provider_adapter as vpa
from blueprint_pipeline import vast_args_payload_transport as transport


MAX_SAFE_VAST_ARGS_STR_BYTES = 16_000


def _decoded_compressed_script(value: str) -> str:
    match = re.search(
        re.escape(transport.VAST_ARGS_GZIP_BASE64_MARKER) + r"([A-Za-z0-9+/=]+)",
        value,
    )
    assert match is not None
    return gzip.decompress(base64.b64decode(match.group(1))).decode("utf-8")


def test_scene_configuration_args_payload_compresses_below_vast_safe_limit() -> None:
    probe = vpa._probe_shell_script(
        "https://example.invalid/heartbeat",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )

    payload = vpa._create_payload(
        image=vpa.DEFAULT_ISAAC_IMAGE,
        label="blueprint-scene-configuration-size-regression",
        launch_mode="args",
        probe_script=probe,
        disk_gb=vpa.DEFAULT_ISAAC_DISK_GB,
    )

    args_str = payload["args_str"]
    assert len(args_str.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    assert transport.VAST_ARGS_GZIP_BASE64_MARKER in args_str
    command = shlex.split(args_str)
    syntax = subprocess.run(
        ["bash", "-n", "-c", command[2]],
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr.decode("utf-8", errors="replace")


def test_compressed_args_transport_preserves_probe_output_and_exit_code() -> None:
    probe = (
        "echo BLUEPRINT_COMPRESSED_ARGS_STARTED\n"
        + ("# deterministic-compressible-padding\n" * 1_000)
        + "exit 7\n"
    )
    payload = vpa._create_payload(
        image="image",
        label="compressed-args-probe",
        launch_mode="args",
        probe_script=probe,
        disk_gb=20,
    )
    args_str = payload["args_str"]
    command = shlex.split(args_str)

    assert transport.VAST_ARGS_GZIP_BASE64_MARKER in args_str
    assert len(args_str.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    executed = subprocess.run(
        command,
        env={**os.environ, "BLUEPRINT_VAST_ARGS_LOG_HOLD_SECONDS": "0"},
        capture_output=True,
        check=False,
        text=True,
    )
    assert executed.returncode == 7
    assert "BLUEPRINT_COMPRESSED_ARGS_STARTED" in executed.stdout
    assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_STARTED" in executed.stdout
    assert "BLUEPRINT_VAST_ARGS_LOG_HOLD_DONE" in executed.stdout


def test_scene_configuration_onstart_payload_compresses_below_vast_safe_limit() -> None:
    """ssh_direct launches hit the same 16384-byte ceiling as args mode.

    Run ``...-2ad2f8f6-...-154844Z`` (offer 39678103) was refused with
    ``error 400/3471: Invalid args: len(image) > 1024, or len(args) > 16384,
    or len(label) > 256``: the args-mode fix left the onstart branch --
    which ssh_direct uses -- uncompressed.
    """

    probe = vpa._probe_shell_script(
        "https://example.invalid/heartbeat",
        enable_isaac_smoke=True,
        enable_blueprint_bundle=True,
        provider_bundle_kind="task_evaluation_scene_configuration",
    )
    assert len(probe.encode("utf-8")) > MAX_SAFE_VAST_ARGS_STR_BYTES

    payload = vpa._create_payload(
        image=vpa.DEFAULT_ISAAC_IMAGE,
        label="blueprint-scene-configuration-onstart-size",
        launch_mode="ssh_direct",
        probe_script=probe,
        disk_gb=vpa.DEFAULT_ISAAC_DISK_GB,
    )

    onstart = payload["onstart"]
    assert len(onstart.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    assert transport.VAST_ARGS_GZIP_BASE64_MARKER in onstart
    syntax = subprocess.run(
        ["bash", "-n", "-c", onstart],
        capture_output=True,
        check=False,
    )
    assert syntax.returncode == 0, syntax.stderr.decode("utf-8", errors="replace")


def test_scene_configuration_private_startup_values_never_enter_container_env() -> None:
    fake_secret = "private-bootstrap-value-for-test"
    full_env = {
        "ACCEPT_EULA": "Y",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "OPENAI_PROJECT_ID": "proj_public_identity",
        "BLUEPRINT_EVAL_MANIFEST_URI": (
            "https://objects.example/bundle?X-Amz-Signature=fake-signature"
        ),
        "BLUEPRINT_WORKER_RUNTIME_MANIFEST_SIGNED_PUT_URL": (
            "https://objects.example/output?X-Amz-Signature=fake-put-signature"
        ),
        "HF_TOKEN": fake_secret,
        vpa.VAST_RUNTIME_SECRET_BOOTSTRAP_PREFIX + "OPENAI_API_KEY_FILE": (
            "cHJpdmF0ZS10ZXN0LWtleQ=="
        ),
    }
    container_env, private_startup_env = (
        vpa._scene_configuration_startup_environments(full_env)
    )
    assert container_env == {
        "ACCEPT_EULA": "Y",
        "NVIDIA_DRIVER_CAPABILITIES": "all",
        "OPENAI_PROJECT_ID": "proj_public_identity",
    }
    assert set(private_startup_env) == set(full_env) - set(container_env)

    payload = vpa._create_payload(
        image=vpa.DEFAULT_ISAAC_IMAGE,
        label="blueprint-scene-private-startup-env",
        launch_mode="ssh_direct",
        probe_script="echo probe-started",
        disk_gb=vpa.DEFAULT_ISAAC_DISK_GB,
        env=container_env,
        private_startup_env=private_startup_env,
    )
    assert payload["env"] == container_env
    assert fake_secret not in payload["onstart"]
    assert transport.VAST_ARGS_GZIP_BASE64_MARKER in payload["onstart"]
    decoded = _decoded_compressed_script(payload["onstart"])
    assert "rm -f /root/onstart.sh" in decoded
    assert "BLUEPRINT_VAST_PRIVATE_STARTUP_ENV_READY" in decoded
    for name, value in private_startup_env.items():
        assert f"export {name}=" in decoded
        assert value in decoded
    summary = vpa._create_request_summary(payload, secret_values=[fake_secret])
    assert summary["raw_payload_redacted"]["onstart"] == vpa.REDACTED_SECRET_FIELD
    assert transport.VAST_ARGS_GZIP_BASE64_MARKER not in str(summary)


def _scene_configuration_launch_environment(tmp_path) -> dict[str, str]:
    """The container-plus-private environment one scene-configuration launch builds.

    Mirrors ``run_task_evaluation_scene_configuration_vast_job``: the OpenAI
    identities and caps of ``_provider_runtime_inputs``, the upload ceiling,
    the parent deadline and closure reserve, the ArtiFixer pretraining capsule
    and the control-plane stage prefix, passed through ``_probe_env`` with the
    signed bundle and output links and a runtime secret file.
    """

    from blueprint_pipeline.agent_operator_runtime import LIVE_AGENTS_SDK_ENV
    from blueprint_pipeline.task_evaluation_artifixer_pretraining import (
        CAPSULE_BYTES_ENV,
        CAPSULE_SHA_ENV,
        CAPSULE_URL_ENV,
        GPU_PREPARATION_REQUIRED_ENV,
    )
    from blueprint_pipeline.task_evaluation_scene_configuration_cpu_prestage import (
        PREFIX_BYTES_ENV,
        PREFIX_SHA_ENV,
        PREFIX_URL_ENV,
    )
    from blueprint_pipeline.task_evaluation_scene_configuration_runtime_budget import (
        OUTPUT_CLOSURE_RESERVE_SECONDS_ENV,
        PARENT_DEADLINE_EPOCH_ENV,
    )
    from blueprint_pipeline.vast_provider_transfer_upload import EXPECTED_PROVIDER_UPLOAD_BYTES_ENV

    signed = "?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=" + "f" * 64
    runtime_environment = {
        "OPENAI_PROJECT_ID": "proj_public_identity",
        "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID": "key_teacher",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID": "key_review",
        "OPENAI_CONTENT_AGENTS_API_KEY_ID": "key_agents",
        LIVE_AGENTS_SDK_ENV: "true",
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST": "sha256:" + "a" * 64,
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD": "4.0",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS": "40",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD": "1.0",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD": "1.0",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD": "2.0",
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORING_RUNTIME": "openai_agents_api",
        EXPECTED_PROVIDER_UPLOAD_BYTES_ENV: "8589934592",
        PARENT_DEADLINE_EPOCH_ENV: "1794577868.0",
        OUTPUT_CLOSURE_RESERVE_SECONDS_ENV: "1800",
        CAPSULE_URL_ENV: "https://objects.example/semantic.zip" + signed,
        CAPSULE_SHA_ENV: "sha256:" + "b" * 64,
        CAPSULE_BYTES_ENV: "1024",
        GPU_PREPARATION_REQUIRED_ENV: "1",
        PREFIX_URL_ENV: "https://objects.example/prefix.zip" + signed,
        PREFIX_SHA_ENV: "sha256:" + "c" * 64,
        PREFIX_BYTES_ENV: "274390428",
    }
    return vpa._probe_env(
        job_dir=tmp_path / "vast_provider_run",
        enable_isaac_smoke=True,
        provider_bundle_url="https://objects.example/bundle.zip" + signed,
        provider_output_put_url="https://objects.example/output.zip" + signed,
        forward_hf_token=False,
        provider_bundle_kind="task_evaluation_scene_configuration",
        runtime_secret_file_values={"OPENAI_CONTENT_AGENTS_API_KEY_FILE": "sk-not-a-real-key"},
        provider_runtime_environment=runtime_environment,
    )


def test_scene_configuration_stage_prefix_link_is_private_and_its_binding_is_not(tmp_path) -> None:
    from blueprint_pipeline.task_evaluation_scene_configuration_cpu_prestage import (
        PREFIX_BYTES_ENV,
        PREFIX_SHA_ENV,
        PREFIX_URL_ENV,
    )

    full_env = _scene_configuration_launch_environment(tmp_path)
    container_env, private_startup_env = vpa._scene_configuration_startup_environments(full_env)

    assert PREFIX_URL_ENV in private_startup_env
    assert PREFIX_URL_ENV not in container_env
    assert full_env[PREFIX_URL_ENV] not in "\n".join(container_env.values())
    # Not secrets: the on-start checks the download against them.
    assert container_env[PREFIX_SHA_ENV] == "sha256:" + "c" * 64
    assert container_env[PREFIX_BYTES_ENV] == "274390428"


def test_no_link_of_a_scene_configuration_launch_reaches_the_container_env(tmp_path) -> None:
    """Every signed link travels only in the one-shot private start-up preamble.

    The container environment stays readable for the life of the instance. A
    name added later that ends in ``_URL``/``_URI``, or any value that is a
    link, must not land there by default.
    """

    full_env = _scene_configuration_launch_environment(tmp_path)
    full_env["BLUEPRINT_SCENE_CONFIGURATION_FUTURE_CAPSULE_URL"] = "s3-presigned-token"
    full_env["BLUEPRINT_SCENE_CONFIGURATION_FUTURE_REFERENCE"] = "https://objects.example/x?sig=1"
    full_env["BLUEPRINT_SCENE_CONFIGURATION_FUTURE_URI"] = "gs://bucket/object"
    container_env, private_startup_env = vpa._scene_configuration_startup_environments(full_env)

    links = sorted(
        name for name, value in container_env.items()
        if name.upper().endswith(("_URL", "_URI"))
        or value.strip().lower().startswith(("http://", "https://"))
    )
    assert links == []
    assert set(container_env) | set(private_startup_env) == set(full_env)
    assert not set(container_env) & set(private_startup_env)
    # What does stay in the container env: licence terms, identities that are
    # not credentials, digests, sizes, caps and deadlines.
    assert set(container_env) == {
        "ACCEPT_EULA", "PRIVACY_CONSENT", "NVIDIA_DRIVER_CAPABILITIES",
        "BLUEPRINT_VAST_PROBE", "BLUEPRINT_VAST_PROBE_JOB_DIR_BASENAME",
        "OPENAI_PROJECT_ID", "OPENAI_ARTIFIXER_SEMANTIC_TEACHER_API_KEY_ID",
        "OPENAI_ARTIFIXER_VISUAL_REVIEW_API_KEY_ID", "OPENAI_CONTENT_AGENTS_API_KEY_ID",
        "BLUEPRINT_ALLOW_LIVE_AGENTS_SDK_OPERATORS",
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORITY_DIGEST",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_COST_USD",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_MAX_REQUESTS",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_SEMANTIC_TEACHER_MAX_COST_USD",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_ARTIFIXER_VISUAL_REVIEW_MAX_COST_USD",
        "BLUEPRINT_SCENE_CONFIGURATION_OPENAI_CONTENT_AGENTS_MAX_COST_USD",
        "BLUEPRINT_SCENE_CONFIGURATION_AUTHORING_RUNTIME",
        "BLUEPRINT_VAST_EXPECTED_PROVIDER_UPLOAD_BYTES",
        "BLUEPRINT_SCENE_CONFIGURATION_PARENT_DEADLINE_EPOCH",
        "BLUEPRINT_SCENE_CONFIGURATION_OUTPUT_CLOSURE_RESERVE_SECONDS",
        "BLUEPRINT_ARTIFIXER_PRETRAINING_CAPSULE_SHA256",
        "BLUEPRINT_ARTIFIXER_PRETRAINING_CAPSULE_BYTES",
        "BLUEPRINT_ARTIFIXER_PRETRAINING_REQUIRED",
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_CAPSULE_SHA256",
        "BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_CAPSULE_BYTES",
    }

    payload = vpa._create_payload(
        image=vpa.DEFAULT_ISAAC_IMAGE,
        label="blueprint-scene-no-container-links",
        launch_mode="ssh_direct",
        probe_script="echo probe-started",
        disk_gb=vpa.DEFAULT_ISAAC_DISK_GB,
        env=container_env,
        private_startup_env=private_startup_env,
    )
    assert payload["env"] == container_env
    for value in private_startup_env.values():
        assert value not in str(payload["env"])


def test_private_stage_prefix_link_reaches_the_process_the_onstart_starts(tmp_path) -> None:
    """The private preamble exports into the on-start shell, and every process
    it starts (the bundle's entrypoint, then the runner) inherits it."""

    from blueprint_pipeline.task_evaluation_scene_configuration_cpu_prestage import PREFIX_URL_ENV

    full_env = _scene_configuration_launch_environment(tmp_path)
    container_env, private_startup_env = vpa._scene_configuration_startup_environments(full_env)
    entrypoint = tmp_path / "entrypoint.sh"
    entrypoint.write_text(
        "#!/usr/bin/env bash\n"
        f"exec python3 -c 'import os; print(\"RUNNER_SAW:\" + os.environ[\"{PREFIX_URL_ENV}\"])'\n"
    )
    payload = vpa._create_payload(
        image=vpa.DEFAULT_ISAAC_IMAGE,
        label="blueprint-scene-private-link-reaches-runner",
        launch_mode="ssh_direct",
        probe_script=f"bash {shlex.quote(str(entrypoint))}",
        disk_gb=vpa.DEFAULT_ISAAC_DISK_GB,
        env=container_env,
        private_startup_env=private_startup_env,
    )
    # Run the real decoded on-start, with its one-shot ``/root/onstart.sh``
    # removal pointed at a scratch path instead of this machine's.
    decoded = _decoded_compressed_script(payload["onstart"])
    assert decoded.count("/root/onstart.sh") > 0
    decoded = decoded.replace("/root/onstart.sh", str(tmp_path / "onstart.sh"))
    completed = subprocess.run(
        ["bash", "-c", decoded],
        env={"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path)},
        capture_output=True, text=True, check=False, timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "BLUEPRINT_VAST_PRIVATE_STARTUP_ENV_READY" in completed.stdout
    assert f"RUNNER_SAW:{full_env[PREFIX_URL_ENV]}" in completed.stdout


def test_compressed_onstart_preserves_probe_output_and_exit_code() -> None:
    probe = (
        "echo BLUEPRINT_COMPRESSED_ONSTART_STARTED\n"
        + ("# deterministic-compressible-padding\n" * 1_000)
        + "exit 7\n"
    )
    onstart = transport.onstart_mode_script(probe)
    assert len(onstart.encode("utf-8")) <= MAX_SAFE_VAST_ARGS_STR_BYTES
    completed = subprocess.run(
        ["bash", "-c", onstart],
        capture_output=True,
        check=False,
        env={"PATH": os.environ["PATH"]},
    )
    assert completed.returncode == 7
    assert b"BLUEPRINT_COMPRESSED_ONSTART_STARTED" in completed.stdout


def test_small_onstart_script_is_passed_through_unchanged() -> None:
    probe = "echo hello\n"
    assert transport.onstart_mode_script(probe) == probe
