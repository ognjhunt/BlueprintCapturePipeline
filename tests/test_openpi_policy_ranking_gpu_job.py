from blueprint_pipeline.openpi_droid_policy_runtime import OpenPIDroidPolicySpec
from blueprint_pipeline import openpi_policy_ranking_gpu_job as job_module
from blueprint_pipeline.openpi_policy_ranking_gpu_job import LocalOpenPIDroidPolicyClient


def _spec() -> OpenPIDroidPolicySpec:
    return OpenPIDroidPolicySpec(
        policy_id="pi0_fast_droid_jointpos_polaris",
        config_name="pi0_fast_droid_jointpos_polaris",
        checkpoint_uri="gs://openpi-assets/checkpoints/polaris/pi0_fast_droid_jointpos_polaris",
        checkpoint_object_manifest_sha256="a" * 64,
        checkpoint_generation_manifest_sha256="b" * 64,
        checkpoint_inventory_sha256="c" * 64,
        checkpoint_object_count=2,
        checkpoint_size_bytes=10,
        action_space="joint_position",
        action_chunk_rows=10,
    )


def test_local_client_requires_verified_checkpoint_and_preserves_identity() -> None:
    class Policy:
        def infer(self, observation):
            return {"actions": observation["actions"]}

    client = LocalOpenPIDroidPolicyClient(
        spec=_spec(),
        policy=Policy(),
        local_verification={
            "local_checkpoint_verified": True,
            "local_checkpoint_verification_sha256": "d" * 64,
        },
    )
    assert client.infer({"actions": [[1.0]]}) == {"actions": [[1.0]]}
    evidence = client.evidence_summary()
    assert evidence["identity_verified"] is True
    assert evidence["policy_identity"]["policy_id"] == client.policy_id


def test_campaign_keeps_variant_metadata_outside_hashed_episode(
    tmp_path, monkeypatch
) -> None:
    cohort = tmp_path / "cohort.json"
    inventory = tmp_path / "inventory.json"
    background = tmp_path / "background.png"
    warehouse_background = tmp_path / "warehouse.png"
    menagerie = tmp_path / "menagerie"
    checkpoint = tmp_path / "checkpoint"
    for path in (cohort, inventory, background, warehouse_background):
        path.write_bytes(b"x")
    menagerie.mkdir()
    checkpoint.mkdir()
    monkeypatch.setattr(
        job_module, "_gpu_runtime_evidence", lambda: {"gpu_device_present": True}
    )
    monkeypatch.setattr(job_module, "load_policy_spec", lambda *_args, **_kwargs: _spec())
    monkeypatch.setattr(
        job_module,
        "verify_local_checkpoint",
        lambda **_kwargs: {
            "local_checkpoint_verified": True,
            "local_checkpoint_verification_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(job_module, "prepare_franka_droid_runtime", lambda **_kwargs: object())
    observed = []
    observed_calls = []

    def run_episode(**kwargs):
        episode = {"manifest_sha256": "e" * 64, "status": "completed"}
        observed.append(episode)
        observed_calls.append(kwargs)
        return episode

    monkeypatch.setattr(job_module, "run_franka_droid_closed_loop", run_episode)
    monkeypatch.setattr(
        job_module,
        "aggregate_policy_rankings",
        lambda episodes: {
            "status": "completed",
            "policy_count": len(episodes),
            "total_ranking_emitted": True,
        },
    )
    result = job_module.run_openpi_policy_ranking_gpu_campaign(
        cohort_path=cohort,
        checkpoint_inventory_path=inventory,
        captured_site_background_path=background,
        menagerie_root=menagerie,
        output_dir=tmp_path / "output",
        policy_ids=("p1", "p2", "p3", "p4"),
        scene_backgrounds=(
            {
                "scene_id": "captured",
                "scene_kind": "captured_3dgs",
                "background_path": background,
            },
            {
                "scene_id": "warehouse",
                "scene_kind": "controlled_nvidia_usd",
                "background_path": warehouse_background,
            },
        ),
        checkpoint_downloader=lambda _uri: checkpoint,
        policy_loader=lambda _spec, _checkpoint: object(),
    )
    assert result["status"] == "completed"
    assert all("variant_id" not in episode for episode in observed)
    assert [
        row["variant_id"] for row in result["policy_runs"][0]["episode_records"]
    ] == [
        "center",
        "left_2cm",
        "right_2cm",
        "center",
        "left_2cm",
        "right_2cm",
    ]
    assert set(result["rankings"]) == {"captured", "warehouse"}
    assert result["claim_boundary"]["prospective_controlled_warehouse_ranking"] is True
    assert {row["external_background_kind"] for row in observed_calls} == {
        "captured_3dgs",
        "controlled_nvidia_usd",
    }

    def blocked_episode(**_kwargs):
        return {"manifest_sha256": "f" * 64, "status": "blocked"}

    monkeypatch.setattr(job_module, "run_franka_droid_closed_loop", blocked_episode)
    blocked = job_module.run_openpi_policy_ranking_gpu_campaign(
        cohort_path=cohort,
        checkpoint_inventory_path=inventory,
        captured_site_background_path=background,
        menagerie_root=menagerie,
        output_dir=tmp_path / "blocked-output",
        policy_ids=("p1", "p2", "p3", "p4"),
        scene_backgrounds=(
            {
                "scene_id": "captured",
                "scene_kind": "captured_3dgs",
                "background_path": background,
            },
            {
                "scene_id": "warehouse",
                "scene_kind": "controlled_nvidia_usd",
                "background_path": warehouse_background,
            },
        ),
        checkpoint_downloader=lambda _uri: checkpoint,
        policy_loader=lambda _spec, _checkpoint: object(),
    )
    assert blocked["status"] == "blocked"
    assert blocked["rankings"] == {}
    assert all(row["status"] == "blocked" for row in blocked["policy_runs"])
    assert blocked["claim_boundary"]["learned_policy_simulator_execution"] is False
    assert blocked["claim_boundary"]["prospective_captured_site_ranking"] is False
    assert (
        blocked["claim_boundary"]["prospective_controlled_warehouse_ranking"]
        is False
    )
