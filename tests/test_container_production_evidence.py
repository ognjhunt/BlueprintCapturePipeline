from __future__ import annotations

from pathlib import Path

from scripts.build_container_production_evidence import (
    MAX_SOURCE_SIZES,
    SOURCE_NAMES,
    build_container_production_evidence,
)


SHA = "a" * 40


def _seed(source_dir: Path) -> None:
    source_dir.mkdir()
    values = {
        "production_image_id.txt": f"sha256:{'b' * 64}\n",
        "development_image_id.txt": f"sha256:{'c' * 64}\n",
        "production_user.txt": "blueprint:blueprint\n",
        "development_user.txt": "blueprint:blueprint\n",
        "compose-config.yml": "services:\n  pipeline:\n    image: test\n",
        "compose-config.ok": "compose-config-ok\n",
        "systemd-security.log": "blueprint-pipeline.service 3.2 OK\n",
        "systemd-security.ok": "systemd-security-ok\n",
        "production-runtime-smoke.log": "container-smoke-ok\n",
        "development-runtime-smoke.log": "container-smoke-ok\n",
    }
    for name, value in values.items():
        (source_dir / name).write_text(value, encoding="utf-8")


def test_container_evidence_binds_actual_ci_sources_and_both_images(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sources"
    _seed(source_dir)

    evidence = build_container_production_evidence(source_dir=source_dir, repository_sha=SHA)

    assert evidence["status"] == "passed"
    assert evidence["production_runtime_smoke_passed"] is True
    assert evidence["development_runtime_smoke_passed"] is True
    assert evidence["production_nonroot_user"] is True
    assert evidence["development_nonroot_user"] is True
    assert set(evidence["artifact_digests"]) == set(SOURCE_NAMES)


def test_container_evidence_rejects_forged_sentinel_and_extra_source(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sources"
    _seed(source_dir)
    (source_dir / "systemd-security.ok").write_text("looks-good", encoding="utf-8")
    (source_dir / "untrusted.txt").write_text("extra", encoding="utf-8")

    evidence = build_container_production_evidence(source_dir=source_dir, repository_sha=SHA)

    assert evidence["status"] == "blocked"
    assert "container_source_file_set_invalid" in evidence["blockers"]
    assert "container_systemd_security_not_passed" in evidence["blockers"]


def test_container_evidence_rejects_oversize_source_before_reading(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "sources"
    _seed(source_dir)
    (source_dir / "compose-config.ok").write_bytes(
        b"x" * (MAX_SOURCE_SIZES["compose_sentinel"] + 1)
    )

    evidence = build_container_production_evidence(source_dir=source_dir, repository_sha=SHA)

    assert evidence["status"] == "blocked"
    assert "container_source_oversize:compose_sentinel" in evidence["blockers"]
