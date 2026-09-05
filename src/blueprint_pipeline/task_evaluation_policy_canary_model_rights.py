"""Bind preserved policy rights terms to exact current adapter source bytes.

A historical rights document is input evidence, never proof that current adapter
code or a new scene has executed. This producer preserves that document and its
historical smoke while deriving a separate current execution binding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .decision_evidence_contracts import canonical_digest, canonical_json

SCHEMA_VERSION = "task_evaluation_policy_canary_model_rights.v1"
CANDIDATES = ("pi05_droid", "groot_n17_droid")


class PolicyCanaryModelRightsError(ValueError):
    """The preserved terms cannot be bound to these exact source bytes."""


def _require(value: bool, code: str) -> None:
    if not value:
        raise PolicyCanaryModelRightsError("policy_canary_model_rights_" + code)


def _bytes(path: Path) -> bytes:
    _require(path.is_file() and not any(p.is_symlink() for p in (path, *path.parents)), "file_invalid")
    return path.read_bytes()


def _sha(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True)
    _require(result.returncode == 0, "source_commit_unavailable")
    return result.stdout


def materialize_policy_canary_model_rights(
    *, template_path: str | Path, repo_root: str | Path, source_commit: str,
    scene_id: str, task_id: str, output_path: str | Path,
) -> dict[str, Any]:
    """Derive current source evidence without changing model terms or granting rights."""
    _require(re.fullmatch(r"[0-9a-f]{40}", source_commit) is not None, "commit_invalid")
    _require(all(isinstance(value, str) and value.strip() for value in (scene_id, task_id)), "identity_invalid")
    repo = Path(repo_root)
    _require(repo.is_absolute() and repo.is_dir(), "repo_invalid")
    _require(_git(repo, "rev-parse", "HEAD").decode().strip() == source_commit, "checkout_commit_mismatch")
    template_bytes = _bytes(Path(template_path))
    template = json.loads(template_bytes)
    _require(isinstance(template, dict)
             and template.get("rights_digest") == canonical_digest(template, digest_field="rights_digest"),
             "template_digest_invalid")
    _require([row.get("candidate_id") for row in template.get("candidates", [])] == list(CANDIDATES)
             and all(template.get(field) is False for field in (
                 "policy_self_grading_permitted", "official_ranking_permitted", "scene_promotion_permitted",
                 "secret_material_recorded")), "template_claim_boundary_invalid")
    code = template.get("blueprint_adapter_code")
    _require(isinstance(code, Mapping) and isinstance(code.get("modules"), list) and bool(code["modules"]),
             "template_modules_invalid")
    modules = []
    seen = set()
    for row in code["modules"]:
        name = row.get("path") if isinstance(row, Mapping) else None
        _require(isinstance(name, str) and re.fullmatch(r"src/blueprint_pipeline/[A-Za-z0-9_]+\.py", name) is not None
                 and name not in seen, "template_module_path_invalid")
        seen.add(name)
        payload = _bytes(repo / name)
        committed = _git(repo, "show", f"{source_commit}:{name}")
        _require(payload == committed, "source_bytes_differ_from_commit")
        modules.append({"path": name, "sha256": _sha(payload), "size_bytes": len(payload),
                        "template_recorded_sha256": row.get("sha256")})
    historical = template.get("historical_runtime_smoke")
    _require(isinstance(historical, Mapping) and historical.get("input_evidence_only") is True
             and historical.get("current_scene_runtime_proof") is False, "historical_claim_boundary_invalid")
    value = {
        "schema_version": SCHEMA_VERSION, "status": "exact_source_bound_rights_reference",
        "source_commit": source_commit, "scene_id": scene_id, "task_id": task_id,
        "source_template": {"schema_version": template.get("schema_version"),
                            "scene_id": template.get("scene_id"), "task_id": template.get("task_id"),
                            "rights_digest": template["rights_digest"], "sha256": _sha(template_bytes),
                            "size_bytes": len(template_bytes)},
        "provider_disclosure_scope": template["provider_disclosure_scope"],
        "candidates": template["candidates"],
        "blueprint_adapter_code": {**code, "modules": modules},
        "historical_runtime_smoke": dict(historical),
        "rights_reauthorization_performed": False, "current_scene_runtime_proof": False,
        "policy_self_grading_permitted": False, "official_ranking_permitted": False,
        "scene_promotion_permitted": False, "secret_material_recorded": False,
    }
    value["rights_digest"] = canonical_digest(value, digest_field="rights_digest")
    output = Path(output_path)
    _require(output.is_absolute() and not any(p.is_symlink() for p in (output, *output.parents)), "output_invalid")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (canonical_json(value) + "\n").encode()
    if output.exists():
        _require(_bytes(output) == payload, "output_conflict")
    else:
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
                os.fchmod(stream.fileno(), 0o440)
                os.link(temporary, output)
            finally:
                temporary.unlink(missing_ok=True)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("template", "repo-root", "source-commit", "scene-id", "task-id", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    value = materialize_policy_canary_model_rights(template_path=args.template, repo_root=args.repo_root,
        source_commit=args.source_commit, scene_id=args.scene_id, task_id=args.task_id, output_path=args.output)
    payload = _bytes(Path(args.output))
    result = {"rights_digest": value["rights_digest"], "artifact": {"path": args.output,
              "sha256": _sha(payload), "size_bytes": len(payload)}}
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
