"""Exact public G1 subset inventory; no downloads or rights grants at import."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import canonical_digest
from .native_g1_beta_authority import CANDIDATE_DIGESTS, INVENTORY_SHA256, SONIC_INVENTORY_SHA256

ORDER = tuple(CANDIDATE_DIGESTS)
DP_INITIALIZER = {
    "role": "publisher_required_torchvision_initialization",
    "relative_path": "hub/checkpoints/resnet18-f37072fd.pth",
    "url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "sha256": "sha256:f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    "size_bytes": 46830571,
}
MODEL_BASE = "https://modelscope.cn/models/Twang2026/HumanoidArena_models/resolve/master/"


def selected_candidates(candidate_ids) -> tuple[str, ...]:
    if (not isinstance(candidate_ids, (list, tuple)) or not candidate_ids
            or tuple(candidate_ids) != tuple(c for c in ORDER if c in candidate_ids)):
        raise ValueError("g1_beta_checkpoint_selection_invalid")
    return tuple(candidate_ids)


def needs_pi_tokenizer(candidate_ids) -> bool:
    return any("_pi05_" in candidate for candidate in selected_candidates(candidate_ids))


def selected_asset_inventory(candidate_ids, *, config_root: Path | None = None) -> dict[str, Any]:
    selected = selected_candidates(candidate_ids)
    root = config_root or Path(__file__).resolve().parents[2] / "configs"
    values = {}
    for name, expected in (("g1_humanoidarena_checkpoint_inventory.v1.json", INVENTORY_SHA256),
                           ("g1_sonic_default_asset_inventory.v1.json", SONIC_INVENTORY_SHA256)):
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("g1_beta_asset_inventory_missing")
        data = path.read_bytes()
        if "sha256:" + hashlib.sha256(data).hexdigest() != expected:
            raise ValueError("g1_beta_asset_inventory_changed")
        values[name] = json.loads(data)
    candidates = {row["candidate_id"]: row for row in values["g1_humanoidarena_checkpoint_inventory.v1.json"]["candidates"]}
    rows = []
    for candidate in selected:
        row = candidates[candidate]
        if row["inventory_digest"] != CANDIDATE_DIGESTS[candidate]:
            raise ValueError("g1_beta_asset_candidate_changed")
        rows.extend({"candidate_id": candidate,
                     "relative_path": row["subdirectory"] + "/" + item["path"],
                     "url": MODEL_BASE + row["subdirectory"] + "/" + item["path"],
                     "sha256": "sha256:" + item["sha256"], "size_bytes": item["size_bytes"]}
                    for item in row["files"])
    sonic = values["g1_sonic_default_asset_inventory.v1.json"]
    sonic_rows = [{"role": row["role"], "relative_path": row["path"],
                   "sha256": "sha256:" + row["sha256"], "size_bytes": row["size_bytes"],
                   "url": sonic["source_repository"] + "/resolve/" + sonic["source_revision"] + "/" + row["path"]}
                  for row in sonic["files"]]
    auxiliary = [dict(DP_INITIALIZER)] if any("_dp_" in c for c in selected) else []
    value = {"schema_version": "native_g1_beta_asset_inventory.v1", "candidate_ids": list(selected),
        "checkpoint_files": rows, "sonic_files": sonic_rows, "auxiliary_files": auxiliary,
        "auxiliary_download_bytes": sum(row["size_bytes"] for row in auxiliary),
        "checkpoint_download_bytes": sum(row["size_bytes"] for row in rows),
        "sonic_download_bytes": sum(row["size_bytes"] for row in sonic_rows),
        "pi_tokenizer_required": needs_pi_tokenizer(selected)}
    value["inventory_digest"] = canonical_digest(value, digest_field="inventory_digest")
    return value
