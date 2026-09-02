from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.gaussian_splat_decode import SplatData
from blueprint_pipeline.particlefield_runtime_asset_cache import (
    materialize_cached_particlefield,
    publish_particlefield_runtime_asset,
)
from blueprint_pipeline.particlefield_usd import write_particlefield_usd


def _upstream_asset(root: Path, source_digest: str) -> tuple[Path, Path]:
    splat = SplatData(
        count=2,
        xyz=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
        opacity=np.asarray([0.0, 1.0], dtype=np.float32),
        f_dc=np.zeros((2, 3), dtype=np.float32),
        scales=np.full((2, 3), -2.0, dtype=np.float32),
        quats=np.asarray([[1.0, 0.0, 0.0, 0.0]] * 2, dtype=np.float32),
        properties=(),
    )
    asset = root / "scene.usdc"
    receipt_path = root / "receipt.json"
    receipt = write_particlefield_usd(splat, asset)
    receipt.update(
        source_sha256=source_digest,
        source_kind="nurec_usdz",
        exact_learned_arrays_preserved=True,
        representation_conversion_only=True,
        particlefield_authoring_implementation="nvidia_usd_convert_gsplat",
        particlefield_emissive_material_binding_authored=False,
        particlefield_custom_render_hints_authored=False,
    )
    receipt["receipt_digest"] = canonical_digest(
        receipt, digest_field="receipt_digest"
    )
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return asset, receipt_path


def test_publish_and_materialize_cached_upstream_particlefield(tmp_path: Path) -> None:
    source_digest = "sha256:" + "a" * 64
    asset, receipt = _upstream_asset(tmp_path, source_digest)
    cache_root = tmp_path / "cache"

    published = publish_particlefield_runtime_asset(
        source_digest=source_digest,
        particlefield_path=asset,
        authoring_receipt_path=receipt,
        cache_root=cache_root,
    )
    materialized = materialize_cached_particlefield(
        source_digest=source_digest,
        output_root=tmp_path / "episode" / "native-appearance",
        cache_root=cache_root,
    )

    assert published["upstream_converter"]["version"] == "0.1.15"
    assert materialized is not None
    output = Path(materialized["asset_path"])
    assert output.read_bytes() == asset.read_bytes()
    derived = materialized["authoring_receipt"]
    assert derived["cache_reused"] is True
    assert derived["cache_source_digest"] == source_digest
    assert derived["receipt_digest"] == canonical_digest(
        derived, digest_field="receipt_digest"
    )


def test_cache_refuses_tampered_asset_and_never_overwrites(tmp_path: Path) -> None:
    source_digest = "sha256:" + "b" * 64
    asset, receipt = _upstream_asset(tmp_path, source_digest)
    cache_root = tmp_path / "cache"
    published = publish_particlefield_runtime_asset(
        source_digest=source_digest,
        particlefield_path=asset,
        authoring_receipt_path=receipt,
        cache_root=cache_root,
    )
    with pytest.raises(ValueError, match="particlefield_runtime_cache_entry_exists"):
        publish_particlefield_runtime_asset(
            source_digest=source_digest,
            particlefield_path=asset,
            authoring_receipt_path=receipt,
            cache_root=cache_root,
        )
    cached_asset = Path(published["root"]) / "scene_appearance.usdc"
    cached_asset.chmod(0o640)
    cached_asset.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="particlefield_runtime_cache_asset_invalid"):
        materialize_cached_particlefield(
            source_digest=source_digest,
            output_root=tmp_path / "episode",
            cache_root=cache_root,
        )
