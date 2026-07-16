from __future__ import annotations

import json

from blueprint_pipeline import groot_oscar_digitalocean_prebaked_host as P


IMAGE = "docker.io/nijelhunt/blueprint-groot-oscar-eval@sha256:" + "a" * 64
MANIFEST = "sha256:" + "b" * 64


def _release() -> dict:
    return {
        "status": "completed",
        "release_image_ref": IMAGE,
        "thin_release_contract": {
            "status": "passed",
            "models_externalized": True,
        },
    }


def _cache() -> dict:
    return {
        "status": "passed",
        "provider_volume_id": P.RUNPOD_S3_BUCKET,
        "remote_prefix": P.RUNPOD_CACHE_PREFIX,
        "runtime_path_mapping_verified": True,
        "model_manifest_digest": MANIFEST,
        "verified_file_count": 30,
        "verified_size_bytes": 16_791_338_353,
    }


def test_prebake_admission_preserves_probe_and_full_campaign() -> None:
    result = P.build_prebake_admission(
        release=_release(),
        model_cache=_cache(),
        preflight={"status": "ready"},
        volume_size_gib=50,
        reservation_seconds=1_396,
        future_gpu_seconds=3_980,
        initial_spent_usd=14.557003,
        initial_gpu_seconds=15_624,
        total_spend_cap_usd=20.0,
        gpu_wall_cap_seconds=21_000,
        max_hourly_rate_usd=3.50,
    )
    assert result["status"] == "admitted"
    assert result["maximum_gpu_spend_usd"] == 1.357222
    assert result["maximum_retained_storage_spend_usd"] == 0.198082
    assert result["retention_ttl_seconds"] == 10_800


def test_prebake_admission_rejects_old_wall_cap_and_lost_future_allowance() -> None:
    result = P.build_prebake_admission(
        release=_release(),
        model_cache=_cache(),
        preflight={"status": "ready"},
        volume_size_gib=50,
        reservation_seconds=1_396,
        future_gpu_seconds=3_500,
        initial_spent_usd=14.557003,
        initial_gpu_seconds=15_624,
        total_spend_cap_usd=20.0,
        gpu_wall_cap_seconds=19_154,
        max_hourly_rate_usd=3.39,
    )
    assert result["status"] == "blocked"
    assert "prebake_future_campaign_reservation_below_3980_seconds" in result["blockers"]
    assert "prebake_combined_plan_exceeds_gpu_wall_cap" in result["blockers"]


def test_prebake_admission_keeps_models_external_and_digest_pinned() -> None:
    release = _release()
    release["release_image_ref"] = "docker.io/example:latest"
    release["thin_release_contract"]["models_externalized"] = False
    result = P.build_prebake_admission(
        release=release,
        model_cache=_cache(),
        preflight={"status": "ready"},
        volume_size_gib=50,
        reservation_seconds=1_000,
        future_gpu_seconds=3_980,
        initial_spent_usd=14.557003,
        initial_gpu_seconds=15_624,
        total_spend_cap_usd=20.0,
        gpu_wall_cap_seconds=21_000,
        max_hourly_rate_usd=3.39,
    )
    assert "prebake_release_image_not_digest_pinned" in result["blockers"]
    assert "prebake_release_models_not_externalized" in result["blockers"]


def test_prebake_admission_rejects_cache_without_verified_counts() -> None:
    cache = _cache()
    cache.pop("verified_file_count")
    cache["verified_size_bytes"] = 0
    result = P.build_prebake_admission(
        release=_release(),
        model_cache=cache,
        preflight={"status": "ready"},
        volume_size_gib=50,
        reservation_seconds=1_000,
        future_gpu_seconds=3_980,
        initial_spent_usd=14.557003,
        initial_gpu_seconds=15_624,
        total_spend_cap_usd=20.0,
        gpu_wall_cap_seconds=21_000,
        max_hourly_rate_usd=3.39,
    )
    assert "prebake_model_cache_verified_file_count_invalid" in result["blockers"]
    assert "prebake_model_cache_verified_size_invalid" in result["blockers"]


def test_remote_script_verifies_local_image_and_external_cache() -> None:
    script = P._remote_prebake_script(
        image_ref=IMAGE,
        volume_name="blueprint-models",
        expected={
            "model_manifest_digest": MANIFEST,
            "file_count": 30,
            "total_size_bytes": 16_791_338_353,
        },
    )
    assert f"docker pull {IMAGE}" in script
    assert f"docker image inspect {IMAGE}" in script
    assert "/models/blueprint-groot-oscar-v1" in script
    assert "download_integrity_mismatch" in script
    assert "docker_pat" in script
    assert "raw_secret_values_recorded" in script


def test_read_only_preflight_selects_h100_and_requires_zero_campaign_resources(
    monkeypatch,
) -> None:
    def fake_request(*, method, path, **_kwargs):
        assert method == "GET"
        if path == "/account":
            return 200, {"account": {"status": "active"}}
        if path.startswith("/sizes"):
            return 200, {
                "sizes": [
                    {
                        "slug": P.DEFAULT_SIZE,
                        "available": True,
                        "regions": [P.DEFAULT_REGION],
                        "price_hourly": 3.39,
                        "memory": 245760,
                    }
                ]
            }
        if path == f"/images/{P.DEFAULT_SOURCE_IMAGE}":
            return 200, {
                "image": {
                    "id": 236110000,
                    "slug": P.DEFAULT_SOURCE_IMAGE,
                    "status": "available",
                    "regions": [P.DEFAULT_REGION],
                }
            }
        if path.startswith("/droplets"):
            return 200, {"droplets": []}
        if path.startswith("/volumes"):
            return 200, {"volumes": []}
        if path.startswith("/images?private"):
            return 200, {"images": []}
        raise AssertionError(path)

    monkeypatch.setattr(P, "_request", fake_request)
    result = P.read_only_preflight(
        token="secret",
        region=P.DEFAULT_REGION,
        size=P.DEFAULT_SIZE,
        source_image=P.DEFAULT_SOURCE_IMAGE,
        name="blueprint-groot-oscar-prebake-test",
    )
    assert result["status"] == "ready"
    assert result["price_hourly_usd"] == 3.39
    assert result["campaign_resource_inventory"] == {
        "droplet_ids": [],
        "volume_ids": [],
        "image_ids": [],
    }


def test_delete_exact_named_requires_final_confirmed_absence(monkeypatch) -> None:
    inventories = iter([(True, ["123"]), (True, [])])
    monkeypatch.setattr(P, "_exact_resource_ids", lambda *_args: next(inventories))
    monkeypatch.setattr(
        P,
        "_delete_and_verify",
        lambda *_args: {"provider_absence_confirmed": True},
    )
    result = P._delete_exact_named("token", "droplets", "exact-name")
    assert result["matching_resource_ids"] == ["123"]
    assert result["provider_absence_confirmed"] is True


def test_watchdog_deletes_retained_storage_at_bounded_deadline_by_exact_name(
    tmp_path, monkeypatch
) -> None:
    state = tmp_path / "watchdog_state.json"
    state.write_text(
        json.dumps(
            {
                "deadline_epoch": 0,
                "watchdog_nonce": "nonce",
                "droplet_name": "lost-droplet",
                "volume_name": "lost-volume",
                "snapshot_name": "lost-snapshot",
                "replacement_cache_verified": True,
                "retention_mode": "bounded_prebaked_host_and_model_cache",
            }
        )
    )
    monkeypatch.setattr(P, "_read_secret", lambda _path: "token")
    observed: list[tuple[str, str]] = []

    def fake_delete(_token, kind, name):
        observed.append((kind, name))
        return {"provider_absence_confirmed": True}

    monkeypatch.setattr(P, "_delete_exact_named", fake_delete)
    assert P.watchdog(state_path=state, token_file=tmp_path / "token") == 0
    assert observed == [
        ("droplets", "lost-droplet"),
        ("images", "lost-snapshot"),
        ("volumes", "lost-volume"),
    ]
    result = json.loads((tmp_path / "watchdog_result.json").read_text())
    assert result["status"] == "deadline_cleanup_complete"
