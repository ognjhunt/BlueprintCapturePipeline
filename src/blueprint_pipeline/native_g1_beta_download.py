"""Stream only the exact selected G1 assets under one transfer/deadline bound."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request

from .decision_evidence_contracts import canonical_digest
from .local_policy_provisioning import remaining_seconds
from .native_g1_beta_assets import selected_asset_inventory
from .native_g1_provider_runtime import _load_script


class _HTTPSOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        if urllib.parse.urlsplit(url).scheme != "https":
            raise ValueError("g1_beta_asset_insecure_redirect")
        return super().redirect_request(request, response, code, message, headers, url)


def _fetch(row, destination: Path, *, deadline: float, opener=None):
    """One transfer, no retries; full bytes are checked before atomic publication."""
    if (destination.exists() or destination.is_symlink()
            or any(p.is_symlink() for p in destination.parents)):
        raise ValueError("g1_beta_asset_destination_exists_or_symlink")
    url = row["url"]
    if urllib.parse.urlsplit(url).scheme != "https":
        raise ValueError("g1_beta_asset_source_invalid")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("g1_beta_asset_partial_exists")
    digest, size = hashlib.sha256(), 0
    open_url = opener or urllib.request.build_opener(_HTTPSOnly()).open
    try:
        with partial.open("xb") as output:
            with open_url(urllib.request.Request(url, headers={"Accept-Encoding": "identity"}),
                          timeout=min(45, remaining_seconds(deadline))) as response:
                if urllib.parse.urlsplit(response.geturl()).scheme != "https":
                    raise ValueError("g1_beta_asset_insecure_response")
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) != row["size_bytes"]:
                    raise ValueError("g1_beta_asset_size_header_mismatch")
                while True:
                    remaining_seconds(deadline)
                    chunk = response.read(min(1024 * 1024, row["size_bytes"] - size + 1))
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > row["size_bytes"]:
                        raise ValueError("g1_beta_asset_size_bound_exceeded")
                    output.write(chunk)
                    digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        remaining_seconds(deadline)
        if size != row["size_bytes"] or "sha256:" + digest.hexdigest() != row["sha256"]:
            raise ValueError("g1_beta_asset_identity_mismatch")
        os.link(partial, destination)
    finally:
        partial.unlink(missing_ok=True)
    return {"relative_path": row["relative_path"], "sha256": row["sha256"], "size_bytes": size}


def stage_selected_assets(*, root: Path, output: Path, candidate_ids, deadline: float, fetch=None):
    """All URLs come from the exact packaged inventories, never a registry URL."""
    from .native_g1_provider_runtime import _json
    inventory = selected_asset_inventory(candidate_ids, config_root=root.parent / "configs")
    models = output / "models"
    if models.exists():
        raise ValueError("g1_beta_model_staging_already_started")
    models.mkdir()
    transferred = 0
    maximum = sum(inventory[key] for key in ("checkpoint_download_bytes", "sonic_download_bytes", "auxiliary_download_bytes"))
    for field, folder in (("sonic_files", "sonic"), ("auxiliary_files", "torch"), ("checkpoint_files", "checkpoints")):
        for row in inventory[field]:
            transferred += row["size_bytes"]
            if transferred > maximum:
                raise ValueError("g1_beta_total_transfer_bound_exceeded")
            (fetch or _fetch)(row, models / folder / row["relative_path"], deadline=deadline)
    # Reuse the native byte verifiers, without invoking their download branches.
    sonic_fetcher = _load_script(root / "scripts/fetch_g1_sonic_assets.py", "g1_beta_sonic_fetcher")
    checkpoint_fetcher = _load_script(root / "scripts/fetch_g1_humanoidarena_checkpoint.py", "g1_beta_checkpoint_fetcher")
    sonic = sonic_fetcher.stage_sonic_assets(inventory_path=root.parent / "configs/g1_sonic_default_asset_inventory.v1.json",
        output_dir=models / "sonic", verify_only=True)
    checkpoints = [checkpoint_fetcher.materialize_candidate(
        inventory_path=root.parent / "configs/g1_humanoidarena_checkpoint_inventory.v1.json",
        candidate_id=c, output_dir=models / "checkpoints", verify_only=True) for c in candidate_ids]
    remaining_seconds(deadline)
    result = {"schema_version": "native_g1_beta_asset_stage.v1", "status": "selected_asset_bytes_verified",
        "candidate_ids": list(candidate_ids), "inventory_digest": inventory["inventory_digest"],
        "checkpoints": checkpoints, "sonic": sonic, "auxiliary_files": inventory["auxiliary_files"],
        "maximum_download_bytes": maximum, "attempted_download_bytes": transferred,
        "completed_at_epoch": time.time(), "checkpoint_inference_executed": False}
    result["receipt_digest"] = canonical_digest(result, digest_field="receipt_digest")
    import json
    (models / "selected_assets.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if _json(models / "selected_assets.json") != result:
        raise ValueError("g1_beta_asset_receipt_write_failed")
    return result
