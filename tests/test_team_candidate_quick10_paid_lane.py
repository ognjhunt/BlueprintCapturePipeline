"""The paid lane for ONE admitted team candidate's Quick-10 session, proven without spend.

Beta journey step: L5 team policy evaluation. A real team bundle is sealed from
the scene's frozen Quick-10 inputs and checked by the same static preflight and
Vast bundle preflight a live launch runs; the canonical allocator lane routes a
team authority to the team transport only with a private credential file; and
the transport hands the bearer token to the provider only through its private
runtime-secret channel. The Vast call itself is a recording fake: no provider,
GPU, network or spend.
"""

from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Any

import pytest

from blueprint_pipeline import paid_resource_allocator as allocator
from blueprint_pipeline import policy_canary_allocator_lane as lane
from blueprint_pipeline import team_candidate_quick10_vast as team_vast
from blueprint_pipeline import vast_provider_adapter
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.native_task_arena_execution_contract import native_task_arena_execution_transport_completed
from blueprint_pipeline.native_task_arena_policy_canary_bundle import preflight_sealed_policy_canary_bundle
from blueprint_pipeline.policy_canary_session_bundle_identity import (
    TEAM_ENTRYPOINT_MARKER,
    run_kind_entrypoint_blockers,
    validate_policy_canary_provider_manifest,
)
from blueprint_pipeline.team_candidate_quick10_bundle import (
    TeamCandidateBundleError,
    assert_secret_absent,
    build_team_candidate_session_bundle,
)
from blueprint_pipeline.team_candidate_quick10_staging import scene_inputs_from_frozen_pair, stage_team_runtime
from tests.test_provider_runtime_import_closure import _real_canary_pair_inputs
from tests.test_team_candidate_quick10_rehearsal import TEAM, TOKEN_ENV, _admission

TOKEN = "team-secret-token-0123456789"


def stage_team_bundle(tmp_path: Path, *, admission: dict[str, Any] | None = None) -> dict[str, Any]:
    """A real sealed team bundle over the frozen pair's real Quick-10 inputs (shared with the launch tests)."""

    packet, runtime_receipt, pair_inputs, pair_activation = _real_canary_pair_inputs(tmp_path)
    frozen = json.loads(pair_inputs.read_text())
    paths = stage_team_runtime(
        root=tmp_path / "team", run_id="team-quick10-run-1",
        scene_inputs=scene_inputs_from_frozen_pair(pair_inputs, pair_activation),
        admission=admission or _admission(frozen["task_success_contract"]),
        prompt="Push the mug onto the green target marker.", max_policy_queries=10,
        hard_cap_usd=20.0, hard_ttl_seconds=9_000, cells=frozen["cells"])
    receipt = build_team_candidate_session_bundle(
        job_dir=tmp_path / "team-job", packet_dir=packet, runtime_source_packet_receipt=runtime_receipt,
        runtime_input_manifest_path=paths["runtime_inputs"], session_authority_path=paths["session_authority"],
        execution_spec_path=paths["execution_spec"], implementation_commit="a" * 40, generated_at="fixed")
    return {"receipt": receipt, "paths": paths,
            "authority": json.loads(paths["session_authority"].read_text())}


def private_token_file(tmp_path: Path, value: str = TOKEN) -> Path:
    directory = tmp_path / "token"
    directory.mkdir(mode=0o700)
    path = directory / "bearer"
    path.write_text(value + "\n", encoding="utf-8")
    path.chmod(0o600)
    return path


def test_staged_team_authorities_pass_the_real_watchdog_name_gate(tmp_path: Path) -> None:
    from blueprint_pipeline.vast_independent_watchdog_control import validate_independent_vast_watchdog_names

    _packet, _runtime, pair_inputs, pair_activation = _real_canary_pair_inputs(tmp_path)
    frozen = json.loads(pair_inputs.read_text())
    names = []
    for index in range(2):
        paths = stage_team_runtime(
            root=tmp_path / f"shard-{index}", run_id=f"team-evaluation-s{index:02d}",
            scene_inputs=scene_inputs_from_frozen_pair(pair_inputs, pair_activation),
            admission=_admission(frozen["task_success_contract"]),
            prompt="Push the mug onto the green target marker.", max_policy_queries=10,
            hard_cap_usd=2.5, hard_ttl_seconds=9_300, cells=frozen["cells"])
        name = json.loads(paths["session_authority"].read_text())["resource_name"]
        validate_independent_vast_watchdog_names(pod_name_prefix="blueprint-native-task-policy-canary-",
                                                resource_name_exact=name)
        names.append(name)
    assert len(set(names)) == 2  # Each shard has its own teardown/watchdog identity.


def test_team_bundle_seals_one_candidate_without_weights_servers_or_secret(tmp_path: Path) -> None:
    staged = stage_team_bundle(tmp_path)
    receipt = staged["receipt"]
    assert receipt["status"] == "ready" and receipt["run_kind"] == "team_candidate_evaluation"
    assert receipt["execution_mode"] == "team_candidate_evaluation_session"
    assert receipt["candidate_ids"] == [TEAM] and receipt["learned_policy_rollout_count"] == 10
    assert receipt["team_credential_env"] == TOKEN_ENV and receipt["secret_material_bundled"] is False
    bundle_path = Path(receipt["bundle_path"])
    with zipfile.ZipFile(bundle_path) as archive:
        names = set(archive.namelist())
        entrypoint = archive.read("provider_runtime/run_adp_arena_provider_runtime.sh").decode()
    assert f"provider_runtime/runtime_inputs/policy_execution_spec.{TEAM}.json" in names
    assert not any("pi05_droid" in name or "groot_n17_droid.json" in name or "provisioning" in name
                   for name in names if name.startswith("provider_runtime/runtime_inputs/")
                   or name.startswith("provider_runtime/adp009d_policy_provisioning"))
    assert "provider_runtime/adp009d_policy_server_worker.py" not in names
    assert TEAM_ENTRYPOINT_MARKER in entrypoint and "adp009d_policy_provisioning" not in entrypoint
    assert f'"${{{TOKEN_ENV}:-}}"' in entrypoint
    # The worker's own static preflight and the Vast adapter's bundle preflight both pass.
    static = preflight_sealed_policy_canary_bundle(receipt)
    assert static["status"] == "passed", static["blockers"]
    preflight = vast_provider_adapter._blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight", generated_at="fixed", enable_blueprint_bundle=True,
        enable_isaac_smoke=True, provider_bundle_kind="native_task_arena_policy_canary_session",
        bundle_path=bundle_path, provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert preflight["status"] == "passed", preflight.get("blockers")
    assert_secret_absent(bundle_path, TOKEN)
    # Without the candidate's own spec the adapter refuses the bundle before any mutation.
    stripped = tmp_path / "stripped.zip"
    with zipfile.ZipFile(bundle_path) as source, zipfile.ZipFile(stripped, "w") as target:
        for member in source.infolist():
            if not member.filename.endswith(f"policy_execution_spec.{TEAM}.json"):
                target.writestr(member, source.read(member))
    refused = vast_provider_adapter._blueprint_bundle_preflight(
        job_dir=tmp_path / "preflight-stripped", generated_at="fixed", enable_blueprint_bundle=True,
        enable_isaac_smoke=True, provider_bundle_kind="native_task_arena_policy_canary_session",
        bundle_path=stripped, provider_bundle_url="https://storage.example.com/bundle.zip",
        provider_output_put_url="https://storage.example.com/output.zip")
    assert "provider_runtime_bundle_required_entries_missing" in refused["blockers"]


def test_a_secret_anywhere_in_the_bundle_refuses_it(tmp_path: Path) -> None:
    path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("provider_runtime/runtime_inputs/spec.json", json.dumps({"url": "https://x/act?key=" + TOKEN}))
    with pytest.raises(TeamCandidateBundleError, match="team_candidate_secret_in_bundle"):
        assert_secret_absent(path, TOKEN)
    with pytest.raises(TeamCandidateBundleError, match="too_short"):
        assert_secret_absent(path, "abc")


def test_the_adapter_binds_teardown_and_members_to_the_run_kind(tmp_path: Path) -> None:
    receipt = stage_team_bundle(tmp_path)["receipt"]
    with zipfile.ZipFile(receipt["bundle_path"]) as archive:
        manifest = json.loads(archive.read("provider_runtime/adp_arena_provider_manifest.json"))
    assert validate_policy_canary_provider_manifest(manifest)["local_bundle_ready_for_remote_staging"] is True
    assert run_kind_entrypoint_blockers(manifest, TEAM_ENTRYPOINT_MARKER) == []
    # A team manifest that would start a local server, or a pair manifest without its teardown, is refused.
    assert run_kind_entrypoint_blockers(manifest, TEAM_ENTRYPOINT_MARKER + " adp009d_policy_server_worker.py")
    assert run_kind_entrypoint_blockers({**manifest, "run_kind": "internal_policy_canary",
                                         "execution_mode": "internal_policy_canary_paired_session"},
                                        TEAM_ENTRYPOINT_MARKER)
    for change in ({"candidate_ids": ["pi05_droid"]}, {"secret_material_bundled": True},
                   {"team_credential_env": "BLUEPRINT_TEAM_POLICY_BEARER"}, {"learned_policy_rollout_count": 20}):
        forged = {**manifest, **change}
        forged["input_digest"] = canonical_digest(forged, digest_field="input_digest")
        with pytest.raises(ValueError, match="manifest_invalid"):
            validate_policy_canary_provider_manifest(forged)


@pytest.mark.parametrize("episode_count", [1, 10, 20, 24, 25])
def test_team_transport_accepts_supported_session_sizes(episode_count: int) -> None:
    result = {"status": "runtime_completed_unqualified_pending_closeout",
              "schema_version": "native_task_arena_policy_canary_session_result.v1",
              "run_kind": "team_candidate_evaluation", "claim_ceiling": "diagnostic_policy_execution",
              "learned_policy_rollout_count": episode_count, "candidate_policy_queried": True,
              "scene_promotion_performed": False, "official_ranking_performed": False}
    name = "native_task_arena_policy_canary_session_result.v1.json"
    assert native_task_arena_execution_transport_completed(result, expected_output_filename=name)
    for malformed in (True, False, 25.0, "25", None, 0, -1, 26, 50):
        assert not native_task_arena_execution_transport_completed(
            {**result, "learned_policy_rollout_count": malformed}, expected_output_filename=name)


def test_the_token_reaches_the_provider_only_through_the_runtime_secret_channel(tmp_path, monkeypatch) -> None:
    staged = stage_team_bundle(tmp_path)
    token_file = private_token_file(tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(team_vast, "run_arena_native_control_vast",
                        lambda **kwargs: calls.append(kwargs) or {"status": "completed"})
    kwargs = {"job_dir": tmp_path / "allocator", "prepared_bundle": staged["receipt"],
              "session_authority": staged["authority"], "paid_resource_admission_grant": None, "execute": True,
              "credential_file": token_file, "hard_cap_usd": 20.0, "hard_ttl_seconds": 9_000}
    assert team_vast.run_team_candidate_session_vast(**kwargs) == {"status": "completed"}
    sent = calls[0]
    assert sent["runtime_secret_file_paths"] == {TOKEN_ENV: token_file}
    # No streaming: the session passes no delivery mode, so the launcher's download default applies.
    assert sent["forward_hf_token"] is False and not any(key.startswith("provider_output_") for key in sent)
    assert sent["provider_bundle_kind"] == "native_task_arena_policy_canary_session"
    assert "paired_witness_binding" not in sent and sent["candidate_policy_query_expected"] is True
    assert TOKEN not in json.dumps({key: str(value) for key, value in sent.items()})
    # The one-shot authority is consumed: a second launch allocates nothing.
    again = team_vast.run_team_candidate_session_vast(**kwargs)
    assert again["status"] == "blocked" and again["provider_mutations_performed"] == 0 and len(calls) == 1
    assert again["blockers"] == ["policy_canary_session_authority_already_consumed"]


def test_a_missing_or_shared_token_file_allocates_nothing(tmp_path, monkeypatch) -> None:
    staged = stage_team_bundle(tmp_path)
    monkeypatch.setattr(team_vast, "run_arena_native_control_vast", lambda **kwargs: pytest.fail("launched"))
    token_file = private_token_file(tmp_path)
    token_file.chmod(0o644)
    link = tmp_path / "token-link"
    link.symlink_to(token_file)
    for credential, reason in ((None, "team_candidate_credential_file_missing"),
                               (token_file, "team_candidate_credential_file_not_private"),
                               (link, "team_candidate_credential_file_not_private")):
        result = team_vast.run_team_candidate_session_vast(
            job_dir=tmp_path / "allocator", prepared_bundle=staged["receipt"], session_authority=staged["authority"],
            paid_resource_admission_grant=None, execute=True, credential_file=credential,
            hard_cap_usd=20.0, hard_ttl_seconds=9_000)
        assert result["blockers"] == [reason] and result["provider_mutations_performed"] == 0
    assert not (tmp_path / "allocator" / "policy_canary_session_consumption.json").exists()


def _allocator_argv(tmp_path: Path, staged: dict[str, Any], *extra: str) -> list[str]:
    receipt_path = Path(staged["receipt"]["bundle_path"]).with_name(
        "team_candidate_quick10_session_bundle_receipt.v1.json")
    return ["gpu-canary", "--provider", "vast", "--probe-kind", lane.PROBE_KIND,
            "--native-task-arena-policy-canary-session-authority", str(staged["paths"]["session_authority"]),
            "--native-task-arena-policy-canary-session-bundle-receipt", str(receipt_path),
            "--adp-job-dir", str(tmp_path / "allocator"), "--adp-max-hourly-rate-usd", "0.8",
            "--adp-max-spend-usd", "20.0", "--adp-hard-ttl-seconds", "9000",
            "--admission-out", str(tmp_path / "admission.json"), "--adapter-output", str(tmp_path / "adapter.json"),
            *extra]


def test_the_canonical_allocator_routes_a_team_authority_to_the_team_transport(tmp_path, monkeypatch) -> None:
    staged = stage_team_bundle(tmp_path)
    token_file = private_token_file(tmp_path)
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": "d" * 40}))
    monkeypatch.setattr(lane, "_launch_environment_blockers", lambda *args: [])
    monkeypatch.setattr(lane, "run_native_task_arena_policy_canary_session_vast",
                        lambda **kwargs: pytest.fail("the pair transport ran a team session"))
    monkeypatch.setattr(lane, "run_team_candidate_session_vast",
                        lambda **kwargs: observed.append(kwargs) or {"status": "dry_run_ready",
                                                                    "provider_mutations_performed": 0})
    monkeypatch.delenv(lane.REQUIRE_INTEGRATION_CANARY_ENV, raising=False)

    assert allocator.main(_allocator_argv(tmp_path, staged, "--team-candidate-credential-file", str(token_file))) == 0
    assert observed[0]["credential_file"] == str(token_file) and observed[0]["execute"] is False
    admission = json.loads((tmp_path / "admission.json").read_text())
    binding = admission["allocation_binding"]
    assert admission["status"] == "admitted" and admission["program_id"] == "blueprint-free-beta-evaluated-pilot"
    assert binding["run_kind"] == "team_candidate_evaluation" and binding["candidate_ids"] == [TEAM]
    assert binding["learned_policy_rollout_count"] == 10 and binding["hard_cap_usd"] == 20.0
    assert TOKEN not in (tmp_path / "admission.json").read_text()

    # Without the private credential file the lane refuses before any transport, even with --execute.
    (tmp_path / "admission.json").unlink()
    (tmp_path / "adapter.json").unlink()
    assert allocator.main(_allocator_argv(tmp_path, staged, "--execute")) == 2
    adapter = json.loads((tmp_path / "adapter.json").read_text())
    assert "team_candidate_credential_file_missing" in adapter["blockers"]
    assert adapter["provider_mutations_performed"] == 0 and len(observed) == 1


def test_the_pair_lane_never_takes_a_credential_file(tmp_path, monkeypatch) -> None:
    staged = stage_team_bundle(tmp_path)
    pair_authority = {**staged["authority"], "run_kind": "internal_policy_canary"}
    monkeypatch.setattr(allocator, "_control_plane_checkout_blockers",
                        lambda: ([], {"orchestrator_source_commit": "d" * 40}))
    monkeypatch.setattr(lane, "validate_session_authority", lambda _value: pair_authority)
    monkeypatch.setattr(lane, "validate_provider_bundle", lambda _value, **_kwargs: staged["receipt"])
    monkeypatch.setattr(lane, "_launch_environment_blockers", lambda *args: [])
    monkeypatch.setattr(lane, "run_native_task_arena_policy_canary_session_vast", lambda **kwargs: pytest.fail("ran"))
    token_file = private_token_file(tmp_path)
    assert allocator.main(_allocator_argv(tmp_path, staged, "--team-candidate-credential-file", str(token_file))) == 2
    assert "policy_canary_credential_file_unexpected" in json.loads(
        (tmp_path / "adapter.json").read_text())["blockers"]
    assert os.stat(token_file).st_mode & 0o777 == 0o600
