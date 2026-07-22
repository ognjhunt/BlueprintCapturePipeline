#!/usr/bin/env python3
"""Validate and normalize the exact legacy embedded carrier for release use."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Mapping


COSMOS_REPO = "nvidia/Cosmos-Reason2-2B"
COSMOS_REVISION = "9ce19a195e423419c349abfc86fd07178b230561"
SOURCE_REVISION_MARKER = ".blueprint-source-revision"
OSCAR_SOURCE_SEAL_SCHEMA_VERSION = "blueprint.oscar_runtime_source_seal.v1"


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()


def _sealed_revision(root: Path) -> str:
    marker = root / SOURCE_REVISION_MARKER
    if marker.is_symlink() or not marker.is_file():
        raise RuntimeError(f"embedded carrier source revision evidence missing: {root}")
    revision = marker.read_text(encoding="utf-8").strip().lower()
    if not revision:
        raise RuntimeError(f"embedded carrier source revision marker empty: {root}")
    return revision


def _git_or_sealed_revision(root: Path) -> str:
    if (root / ".git").exists():
        return _git_head(root)
    return _sealed_revision(root)


def _oscar_source_revision(root: Path, provenance_path: Path) -> str:
    if (root / ".git").exists():
        return _git_head(root)
    if provenance_path.is_symlink() or not provenance_path.is_file():
        raise RuntimeError("embedded OSCAR source provenance missing or unsafe")
    try:
        loaded = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("embedded OSCAR source provenance unreadable") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError("embedded OSCAR source provenance invalid")
    if (
        loaded.get("schema_version") != OSCAR_SOURCE_SEAL_SCHEMA_VERSION
        or loaded.get("status") != "sealed"
        or loaded.get("git_metadata_required_at_runtime") is not False
    ):
        raise RuntimeError("embedded OSCAR source provenance invalid")
    revision = str(loaded.get("source_commit") or "").strip().lower()
    if not revision:
        raise RuntimeError("embedded OSCAR source provenance lacks source commit")
    return revision


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def repair(
    *,
    wbc_revision: str,
    groot_revision: str,
    oscar_revision: str,
    roots: Mapping[str, Path] | None = None,
    sonic: Path = Path("/opt/blueprint/ckpts/sonic"),
    models_root: Path = Path("/opt/blueprint/models"),
    oscar_provenance: Path = Path("/opt/blueprint/oscar_source_provenance.json"),
) -> dict:
    roots = dict(
        roots
        or {
            "wbc": Path("/opt/wbc"),
            "groot": Path("/opt/gr00t"),
            "oscar": Path("/opt/oscar-public"),
        }
    )
    if set(roots) != {"wbc", "groot", "oscar"}:
        raise ValueError("embedded carrier repair requires WBC, GR00T, and OSCAR roots")
    expected = {
        "wbc": wbc_revision.lower(),
        "groot": groot_revision.lower(),
        "oscar": oscar_revision.lower(),
    }
    observed = {
        "wbc": _git_or_sealed_revision(roots["wbc"]),
        "groot": _git_or_sealed_revision(roots["groot"]),
        "oscar": _oscar_source_revision(roots["oscar"], oscar_provenance),
    }
    if observed != expected:
        raise RuntimeError(f"embedded carrier source revision mismatch: {observed!r}")

    (roots["wbc"] / ".blueprint-source-revision").write_text(
        expected["wbc"] + "\n", encoding="utf-8"
    )
    (roots["groot"] / ".blueprint-source-revision").write_text(
        expected["groot"] + "\n", encoding="utf-8"
    )

    sonic_config_path = sonic / "config.json"
    processor_config_path = sonic / "processor/processor_config.json"
    sonic_config = json.loads(sonic_config_path.read_text(encoding="utf-8"))
    processor_config = json.loads(processor_config_path.read_text(encoding="utf-8"))
    original_model_name = sonic_config.get("blueprint_original_model_name")
    model_revision = sonic_config.get("blueprint_model_revision")
    if original_model_name != COSMOS_REPO or model_revision != COSMOS_REVISION:
        raise RuntimeError("embedded SONIC checkpoint lacks the exact Cosmos provenance")

    old_alias = models_root / "cosmos-reason2-2b"
    if not old_alias.is_symlink() or not old_alias.resolve().is_dir():
        raise RuntimeError("embedded Cosmos alias is absent or not a directory symlink")
    cosmos_snapshot = old_alias.resolve()
    configured_model = Path(str(sonic_config.get("model_name") or ""))
    if not configured_model.is_absolute() or configured_model.resolve() != cosmos_snapshot:
        raise RuntimeError("embedded SONIC model selector does not resolve to the sealed snapshot")
    processor_kwargs = processor_config.get("processor_kwargs")
    if not isinstance(processor_kwargs, dict):
        raise RuntimeError("SONIC processor_kwargs missing")
    if processor_kwargs.get("model_name") != COSMOS_REPO:
        raise RuntimeError("embedded SONIC processor selector lacks the pinned repository identity")

    selector_root = models_root / "cosmos-selector"
    selector_root.mkdir(parents=True, exist_ok=True)
    for source in cosmos_snapshot.iterdir():
        destination = selector_root / source.name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"Cosmos selector entry mismatch: {destination}")
        else:
            destination.symlink_to(source, target_is_directory=source.is_dir())
    selector_anchor = selector_root / COSMOS_REPO
    selector_anchor.mkdir(parents=True, exist_ok=True)
    selector = selector_anchor / "../.."
    if selector.resolve() != selector_root:
        raise RuntimeError("Cosmos selector escaped its sealed model root")

    sonic_config["model_name"] = str(selector)
    processor_kwargs["model_name"] = str(selector)
    processor_config["processor_kwargs"] = processor_kwargs
    _write_json(sonic_config_path, sonic_config)
    _write_json(processor_config_path, processor_config)

    return {
        "schema_version": "blueprint.embedded_carrier_repair.v1",
        "status": "repaired",
        "source_revisions": observed,
        "cosmos_repo": COSMOS_REPO,
        "cosmos_revision": COSMOS_REVISION,
        "cosmos_selector": str(selector),
        "raw_secret_values_recorded": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wbc-revision", required=True)
    parser.add_argument("--groot-revision", required=True)
    parser.add_argument("--oscar-revision", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = repair(
        wbc_revision=args.wbc_revision,
        groot_revision=args.groot_revision,
        oscar_revision=args.oscar_revision,
    )
    _write_json(Path(args.output), result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
