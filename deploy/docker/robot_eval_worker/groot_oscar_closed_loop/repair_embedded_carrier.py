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


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip().lower()


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
    observed = {name: _git_head(root) for name, root in roots.items()}
    if observed != expected:
        raise RuntimeError(f"embedded carrier source revision mismatch: {observed!r}")

    (roots["wbc"] / ".blueprint-source-revision").write_text(
        expected["wbc"] + "\n", encoding="utf-8"
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
