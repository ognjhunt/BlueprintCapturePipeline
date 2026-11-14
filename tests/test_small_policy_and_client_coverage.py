from __future__ import annotations

from pathlib import Path

import pytest

from blueprint_pipeline import launch_proof_policy, safe_env
from blueprint_pipeline import world_model_policy as wmp


class _SkillRunner:
    def __init__(self, result: object) -> None:
        self.result = result

    def __call__(self, skill_name: str, payload: object) -> object:
        return {"skill_name": skill_name, "payload": dict(payload)} if self.result == "echo" else self.result

    def runtime_metadata(self) -> dict[str, object]:
        return {"preferred_tool": "sdk", "runner": "custom"}

    def skill_metadata(self, skill_name: str) -> dict[str, object]:
        return {"skill_name": skill_name, "skill_runner": "custom"}


def test_launch_proof_policy_and_world_model_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prod_env = {"BLUEPRINT_LAUNCH_PROOF_MODE": "production"}
    local_env = {
        "BLUEPRINT_LAUNCH_PROOF_MODE": "local",
        "SITE_WORLD_RUNTIME_REQUIRED": "yes",
        "SITE_WORLD_RUNTIME_SERVICE_URL": " https://runtime.test ",
        "PIPELINE_SYNC_BUYER_ACCESS_REQUIRED": "true",
        "ALLOW_ME": "true",
        "DENY_ME": "true",
    }

    assert launch_proof_policy.launch_proof_mode(prod_env) == "production"
    assert launch_proof_policy.production_forces_true("ALLOW_ME", env=prod_env) is True
    assert launch_proof_policy.production_forces_false("DENY_ME", env=prod_env) is False
    assert launch_proof_policy.production_forces_true("ALLOW_ME", env=local_env) is True
    assert launch_proof_policy.production_forces_false("DENY_ME", env=local_env) is True
    assert launch_proof_policy.runtime_required(local_env) is True
    assert launch_proof_policy.runtime_url_present(local_env) is True
    assert launch_proof_policy.fallback_geometry_launchable_allowed(prod_env) is False
    assert launch_proof_policy.fallback_geometry_launchable_allowed({}) is True
    assert launch_proof_policy.buyer_access_required(local_env) is True

    for name in [
        "WORLD_MODEL_OUTPUT_POLICY",
        "WORLD_MODEL_EMIT_PRESENTATION",
        "WORLD_MODEL_PROVENANCE_REQUIRED",
    ]:
        monkeypatch.delenv(name, raising=False)
    default_policy = wmp.WorldModelPolicy.from_env()
    assert default_policy.output_policy == "grounding_first"
    monkeypatch.setenv("WORLD_MODEL_OUTPUT_POLICY", "  ")
    monkeypatch.setenv("WORLD_MODEL_EMIT_PRESENTATION", "false")
    monkeypatch.setenv("WORLD_MODEL_PROVENANCE_REQUIRED", "false")
    env_policy = wmp.WorldModelPolicy.from_env()
    assert env_policy.output_policy == "grounding_first"
    assert env_policy.emit_presentation is False
    assert env_policy.provenance_required is False
    assert wmp._string_list({"bad": "shape"}) == []
    derivation = wmp.build_presentation_derivation_policy(
        policy=env_policy,
        variance_policy={
            "allowed_editable_region_classes": ["door", "door", "sign"],
            "forbidden_changes": "safety_zone",
        },
        canonical_authority="capture",
    )
    assert derivation["allowed_editable_region_classes"] == ["door", "sign"]
    assert derivation["forbidden_changes"] == ["safety_zone"]
    provenance = wmp.build_provenance_record(
        grounding_level="capture_backed",
        evidence_sources=["a", "a", "", None],
        confidence="bad",
        extra={"review": "local"},
    )
    assert provenance["evidence_sources"] == ["a"]
    assert provenance["confidence"] is None
    assert provenance["review"] == "local"
    assert wmp.build_provenance_record(
        grounding_level="capture_backed",
        evidence_sources=[],
        confidence=2.5,
    )["confidence"] == 1.0
    assert wmp.build_output_linkage(
        policy=env_policy,
        canonical_artifact_uri="gs://bucket/canonical.json",
        presentation_artifact_uri=None,
        authoritative_record=True,
    )["derivation_mode"] == env_policy.output_policy


def test_safe_env_parser_edge_cases(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("EMPTY_VALUE", raising=False)
    monkeypatch.delenv("EXPORTED_VALUE", raising=False)
    monkeypatch.delenv("VALID_VALUE", raising=False)
    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "# local-only test env",
                "EMPTY_VALUE=# comment only",
                "export EXPORTED_VALUE=present",
                "NO_EQUALS_LINE",
                "BAD-KEY=value",
                "VALID_VALUE='kept value'",
            ]
        ),
        encoding="utf-8",
    )

    summary = safe_env.load_env_files([tmp_path], filenames=[".env"])

    assert summary["loaded_keys"] == ["EMPTY_VALUE", "EXPORTED_VALUE", "VALID_VALUE"]
    assert safe_env._parse_env_file(tmp_path / ".env")["EMPTY_VALUE"] == ""
    assert "BAD-KEY" not in safe_env._parse_env_file(tmp_path / ".env")
    assert "NO_EQUALS_LINE" not in safe_env._parse_env_file(tmp_path / ".env")
    assert safe_env.contract_test_env().get("VALID_VALUE") == "kept value"


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self.body = body
        self.status = status

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.body


