"""Create a packet-checked, owner-scoped G1 campaign registry without spend.

The operator supplies authenticated owner IDs and retained packet paths. This
module never invents an owner, mutates a provider, or weakens the runtime intake
checks that reverify the registry after it is installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .decision_evidence_contracts import cross_runtime_canonical_digest as digest
from .native_g1_team_campaign_intake import (
    BINDING_FIELDS,
    REGISTRY_SCHEMA,
    _selected_binding,
    list_g1_team_campaign_setups,
)
from .task_evaluation_packet_planning_setup import make_packet_planning_setup


def build_g1_team_campaign_registry(bindings: Any) -> dict[str, Any]:
    """Validate every owner and source packet before sealing the catalog."""

    if not isinstance(bindings, list) or not 1 <= len(bindings) <= 100:
        raise ValueError("g1_team_campaign_registry_bindings_invalid")
    registry: dict[str, Any] = {
        "schema_version": REGISTRY_SCHEMA,
        "bindings": bindings,
        "registry_digest": "",
    }
    for row in bindings:
        if not isinstance(row, dict) or set(row) != BINDING_FIELDS:
            raise ValueError("g1_team_campaign_binding_invalid")
        owner = row["owner"]
        if (
            not isinstance(owner, dict)
            or set(owner) != {"user_id", "organization_id"}
            or any(not isinstance(value, str) or not value for value in owner.values())
        ):
            raise ValueError("g1_team_campaign_owner_invalid")
        binding = _selected_binding(registry, {
            "owner": owner,
            "source_packet_receipt_digest": row["source_packet_receipt_digest"],
        })
        setup = make_packet_planning_setup(
            source_packet_dir=Path(binding["source_packet_dir"])
        )
        if any(binding[field] != setup[field] for field in (
            "scene_id", "task_id", "source_packet_receipt_digest"
        )):
            raise ValueError("g1_team_campaign_binding_packet_mismatch")
    registry["registry_digest"] = digest(registry, digest_field="registry_digest")
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bindings-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not args.bindings_file.is_absolute() or args.bindings_file.is_symlink()
        or not args.bindings_file.is_file()
        or not args.output.is_absolute() or args.output.is_symlink()
        or not args.output.parent.is_dir()
        or args.output.parent.resolve() != args.output.parent
    ):
        parser.error("absolute paths and a real existing output directory are required")
    bindings = json.loads(args.bindings_file.read_text(encoding="utf-8"))
    registry = build_g1_team_campaign_registry(bindings)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(registry, stream, sort_keys=True, indent=2)
        stream.write("\n")
    for row in bindings:
        list_g1_team_campaign_setups(registry_path=args.output, owner=row["owner"])
    print(json.dumps({
        "registry_path": str(args.output),
        "registry_digest": registry["registry_digest"],
        "bindings": len(bindings),
        "provider_mutation_performed": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
