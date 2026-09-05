"""Fail-closed input and staging primitives for production scene submissions."""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from .decision_evidence_contracts import canonical_digest


class SceneConfigurationSubmissionError(ValueError):
    """Inputs cannot form one coherent production submission."""


def require(condition: bool, suffix: str) -> None:
    if not condition:
        raise SceneConfigurationSubmissionError("scene_configuration_submission_" + suffix)


def _no_symlinks(path: Path) -> None:
    require(not any(item.is_symlink() for item in (path, *path.parents)),
            "input_symlink_forbidden")


def sha(path: Path) -> str:
    _no_symlinks(path)
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return "sha256:" + value.hexdigest()


def read(path: str | Path, *, digest_field: str | None = None) -> dict[str, Any]:
    path = Path(path)
    _no_symlinks(path)
    require(path.is_file() and not path.is_symlink(), "input_file_invalid")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SceneConfigurationSubmissionError(
            "scene_configuration_submission_input_json_invalid"
        ) from exc
    require(isinstance(value, dict), "input_json_invalid")
    if digest_field:
        require(value.get(digest_field) == canonical_digest(
            value, digest_field=digest_field
        ), "input_digest_mismatch")
    return value


def checked_file(path: str | Path, record: dict[str, Any]) -> Path:
    path = Path(path)
    _no_symlinks(path)
    size = record.get("size_bytes")
    require(isinstance(size, int) and not isinstance(size, bool) and size > 0,
            "input_bytes_mismatch")
    require(path.is_file() and not path.is_symlink(), "input_file_invalid")
    require(path.stat().st_size == record.get("size_bytes") and
            sha(path) == record.get("sha256"), "input_bytes_mismatch")
    return path


def beneath(root: Path, relative: str) -> Path:
    require(isinstance(relative, str) and "\\" not in relative,
            "relative_path_invalid")
    item = Path(relative)
    require(not item.is_absolute() and ".." not in item.parts and
            item.as_posix() not in ("", "."), "relative_path_invalid")
    candidate = root / item
    _no_symlinks(candidate)
    require(candidate.resolve().is_relative_to(root.resolve()), "relative_path_invalid")
    return candidate


def slug(value: str) -> str:
    require(isinstance(value, str) and
            re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,180}", value) is not None,
            "identity_invalid")
    return value


def source_inputs(*, installation_path: Path, publisher_path: Path,
                  preparation_path: Path, task: dict[str, Any],
                  commit: str) -> dict[str, Any]:
    installation = read(installation_path, digest_field="receipt_digest")
    preparation = read(preparation_path, digest_field="receipt_digest")
    publisher = read(publisher_path)
    require(preparation.get("status") == "source_context_prepared_pending_calibrated_views"
            and not preparation.get("blockers"), "source_preparation_blocked")
    require(installation.get("schema_version") ==
            "public_scene_host_input_installation_receipt.v1" and
            installation.get("status") == "installed" and
            installation.get("service_readable") is True and
            preparation.get("schema_version") == "public_scene_source_preparation.v1",
            "source_preparation_invalid")
    require(installation.get("source_commit_sha") == commit and
            preparation.get("source_commit") == commit, "source_preparation_commit_mismatch")
    require(preparation.get("source_installation_digest") == installation["receipt_digest"],
            "source_preparation_installation_mismatch")
    root = Path(installation["destination_root"])
    _no_symlinks(root)
    require(root.resolve() == installation_path.parent.resolve(),
            "source_installation_root_mismatch")
    scene_id = str(installation["scene_id"])
    require(str(task.get("publisher_scene_id")) == scene_id and
            str(preparation.get("scene_id")) == scene_id and
            str(publisher.get("scene_id")) == scene_id, "scene_identity_mismatch")
    raw: dict[str, Any] = {}
    roles = {"appearance_3dgs", "semantic_metadata", "scene_structure",
             "collision_usd", "publisher_scene_usdz"}
    inventory_paths: set[str] = set()
    for row in installation["files"]:
        relative = row["relative_path"]
        require(relative not in inventory_paths, "source_inventory_duplicate")
        inventory_paths.add(relative)
        path = checked_file(beneath(root, relative), row)
        if "receipt_id" in row or row.get("kind") == "rights_receipt":
            # The real installer inventories rights records alongside source
            # files. They must still be rehashed, but are not publisher assets.
            require(not row.get("role") and bool(row.get("receipt_id"))
                    and "rights_receipt_ids" not in row
                    and row.get("kind") in {None, "rights_receipt"},
                    "source_role_invalid")
            continue
        role = row.get("role")
        require(role in roles and role not in raw, "source_role_invalid")
        matches = [p for p in publisher["artifacts"]
                   if p.get("sha256") == row["sha256"] and
                   p.get("size_bytes") == row["size_bytes"]]
        require(len(matches) == 1, "publisher_binding_mismatch")
        published = matches[0]
        uri = urlparse(published["publisher_url"])
        revision = published["publisher_revision"]
        require(uri.scheme == "https" and uri.hostname == "huggingface.co" and
                not uri.username and not uri.password and not uri.query and
                not uri.fragment and uri.netloc == "huggingface.co" and
                re.fullmatch(r"[0-9a-f]{40}", revision) is not None and
                f"/resolve/{revision}/" in uri.path, "publisher_revision_invalid")
        raw[role] = {**published, "path": path}
    require(set(raw) == roles, "source_roles_missing")
    artifacts = []
    for row in preparation["artifacts"]:
        path = checked_file(beneath(preparation_path.parent, row["relative_path"]), row)
        artifacts.append((path, read(path)))
    identities = {}
    for role, key in (("movable_subject", "subject"), ("source_support", "support")):
        instance = str(task[key]["source_instance_id"])
        rows = [r for r in preparation["source_identities"] if r.get("role") == role and
                str(r.get("source_instance_id")) == instance]
        require(len(rows) == 1, "source_identity_missing")
        matches = [(p, v) for p, v in artifacts if
                   v.get("receipt_digest") == rows[0]["identity_receipt_digest"]]
        require(len(matches) == 1, "source_identity_missing")
        path, value = matches[0]
        require(value.get("receipt_digest") == canonical_digest(value, digest_field="receipt_digest")
                and value.get("schema_version") == "interiorgs_sage_collision_identity.v1"
                and str(value.get("target", {}).get("interiorgs_instance_id")) == instance,
                "source_identity_invalid")
        for field, raw_role in (("interiorgs_labels", "semantic_metadata"),
                                ("sage_collision_usd", "collision_usd")):
            source = value["source_files"][field]
            require(all(source.get(k) == raw[raw_role][k] for k in ("sha256", "size_bytes")),
                    "source_identity_bytes_mismatch")
        require(value.get("coordinate_frame") == {
            "up_axis": "Z", "meters_per_unit": 1.0, "transform_applied": "identity"
        }, "source_identity_frame_invalid")
        whole = value.get("whole_object_matches", [])
        require(value.get("whole_object_collision_identity_passed") is True and
                len(whole) == 1 and whole[0].get("collision_api_applied") is True,
                "whole_object_identity_not_qualified")
        match = whole[0]
        for metric, minimum in (("aabb_iou", 0.85),
                                ("target_coverage_fraction", 0.9),
                                ("mesh_coverage_fraction", 0.9)):
            number = match.get(metric)
            require(isinstance(number, (int, float)) and not isinstance(number, bool)
                    and math.isfinite(number) and minimum <= number <= 1.0,
                    "whole_object_identity_not_qualified")
        for bounds in (value["target"], match):
            minimum, maximum = bounds.get("world_aabb_min_m"), bounds.get("world_aabb_max_m")
            require(isinstance(minimum, list) and isinstance(maximum, list)
                    and len(minimum) == len(maximum) == 3
                    and all(isinstance(x, (int, float)) and not isinstance(x, bool)
                            and math.isfinite(x) for x in minimum + maximum)
                    and all(lo < hi for lo, hi in zip(minimum, maximum, strict=True)),
                    "source_identity_bounds_invalid")
        require(all(isinstance(match.get(k), int) and not isinstance(match[k], bool)
                    and match[k] > 0 for k in ("point_count", "face_count")),
                "source_identity_mesh_invalid")
        require(sum(row == match for row in value.get("overlapping_meshes", [])) == 1,
                "source_identity_mesh_invalid")
        identities[key] = {"path": path, "receipt": value, "match": match}
    require(identities["subject"]["match"]["prim_path"] !=
            identities["support"]["match"]["prim_path"], "subject_support_collider_shared")
    return {"scene_id": scene_id, "raw": raw, "identities": identities,
            "preparation": preparation, "artifacts": artifacts}


def release_inputs(*, deploy_path: Path, provenance_path: Path,
                   publication_root: Path, commit: str,
                   release_admission_mode: str = "promoted") -> tuple[dict, dict, dict]:
    deploy = read(deploy_path)
    provenance = read(provenance_path)
    binding = deploy.get("release_provenance", {})
    require(deploy.get("source_commit") == commit and
            provenance.get("git_sha") == commit and binding.get("git_sha") == commit,
            "release_commit_mismatch")
    require(deploy.get("schema_version") == "control_plane_commit_deploy_receipt.v1"
            and deploy.get("intake_runtime", {}).get("commit_proven") is True
            and deploy["intake_runtime"].get("source_commit") == commit,
            "release_deployment_unproven")
    require(release_admission_mode in {"promoted", "development_iteration"},
            "release_admission_mode_invalid")
    if release_admission_mode == "development_iteration":
        claim = provenance.get("claim_boundary")
        require(binding.get("sha256") == sha(provenance_path)
                and binding.get("provenance_status") == "iteration"
                and binding.get("promotion_eligible") is False
                and binding.get("canonical_full_lane_verified") is False
                and binding.get("run_id") is None and binding.get("run_url") is None
                and set(provenance) == {"schema_version", "status", "git_sha", "promotion_eligible", "claim_boundary"}
                and provenance.get("schema_version") == "blueprint.deploy_release_provenance.v1"
                and provenance.get("status") == "iteration"
                and provenance.get("promotion_eligible") is False
                and isinstance(claim, dict)
                and set(claim) == {"canonical_full_lane_verified", "promotion_eligible", "evidence_grade"}
                and claim["canonical_full_lane_verified"] is False and claim["promotion_eligible"] is False
                and claim["evidence_grade"] == "development_only",
                "release_iteration_provenance_unproven")
    else:
        require(binding.get("sha256") == sha(provenance_path)
                and binding.get("provenance_status") == "verified"
                and binding.get("canonical_full_lane_verified") is True
                and binding.get("promotion_eligible") is True
                and provenance.get("status") == "verified"
                and provenance.get("schema_version") == "blueprint.deploy_release_provenance.v1"
                and provenance.get("workflow_name") == "Full Test Lane"
                and provenance.get("workflow_path") == ".github/workflows/full-test-lane.yml"
                and provenance.get("job_name") == "Full pytest lane on CPU runner"
                and provenance.get("claim_boundary", {}).get("canonical_full_lane_verified") is True
                and provenance.get("run_id") == binding.get("run_id")
                and isinstance(provenance.get("run_id"), int)
                and not isinstance(provenance["run_id"], bool) and provenance["run_id"] > 0,
                "release_provenance_unproven")
        collection = provenance.get("collection", {})
        test_count = collection.get("test_count")
        require(isinstance(test_count, int) and not isinstance(test_count, bool)
                and test_count > 0 and collection.get("skipped_count") == 0,
                "release_provenance_unproven")
    publications = []
    for kind, schema, digest_key in (
        ("scene-configuration", "task_evaluation_scene_configuration_toolchain_publication.v1",
         "toolchain_digest"),
        ("splat-render", "task_evaluation_splat_render_runtime_publication.v1", "runtime_digest"),
    ):
        value = read(publication_root / kind / f"{commit}.publication.v1.json",
                     digest_field="receipt_digest")
        count = value.get("file_count")
        digest = value.get(digest_key)
        require(value.get("source_commit") == commit and
                value.get("schema_version") == schema and
                value.get("status") == "published_and_read_back" and
                value.get("full_byte_service_account_readback_passed") is True and
                value.get("readback_actor") == "service-account:blueprint" and
                isinstance(count, int) and not isinstance(count, bool) and count > 0 and
                isinstance(digest, str) and
                re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is not None,
                "runtime_publication_unproven")
        publications.append(value)
    return deploy, *publications


class Staging:
    """Exact byte inventory; raw sources retain publisher URLs and cannot be uploaded."""

    def __init__(self, root: Path, namespace: str):
        _no_symlinks(root)
        require(not root.exists(), "staging_root_exists")
        root.mkdir(parents=True)
        self.root = root
        self.prefix = f"s3://blueprint/task-evaluation/production-inputs/{namespace}/"
        self.files: dict[str, dict[str, Any]] = {}

    def _record(self, relative: str, *, publisher_uri: str | None = None) -> dict[str, Any]:
        path = beneath(self.root, relative)
        ref = {"uri": publisher_uri or self.prefix + relative,
               "digest": sha(path), "size_bytes": path.stat().st_size}
        require(ref["uri"] not in self.files, "duplicate_reference")
        self.files[ref["uri"]] = {"relative_path": relative, **ref,
                                 "publication_allowed": publisher_uri is None}
        return ref

    def copy(self, source: Path, relative: str, *,
             publisher_uri: str | None = None) -> dict[str, Any]:
        _no_symlinks(source)
        require(source.is_file(), "input_file_invalid")
        target = beneath(self.root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
        require(sha(source) == sha(target), "staged_bytes_mismatch")
        return self._record(relative, publisher_uri=publisher_uri)

    def json(self, relative: str, value: dict[str, Any]) -> dict[str, Any]:
        target = beneath(self.root, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.write("\n")
        return self._record(relative)

    def reference_rows(self, request: Any) -> list[dict[str, Any]]:
        rows = []
        def walk(value: Any, path: str) -> None:
            if isinstance(value, dict):
                if set(value) == {"uri", "digest", "size_bytes"}:
                    item = self.files[value["uri"]]
                    require(all(item[k] == v for k, v in value.items()),
                            "staged_reference_mismatch")
                    local = beneath(self.root, item["relative_path"])
                    require(sha(local) == value["digest"] and
                            local.stat().st_size == value["size_bytes"], "staged_bytes_mismatch")
                    rows.append({"contract_path": path, **value, "materialized_path": str(local),
                                 "full_byte_service_account_readback_passed": True})
                else:
                    for key, child in value.items():
                        walk(child, f"{path}.{key}" if path else key)
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    walk(child, f"{path}.{index}")
        walk(request, "")
        return rows
