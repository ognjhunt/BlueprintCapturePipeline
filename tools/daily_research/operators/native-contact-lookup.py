#!/usr/bin/env python3
"""Existing private host controls for native lookups; absent scope never spends."""

import argparse
import json
import re
from pathlib import Path

from tools.daily_research.firestore import Bridge, FirestoreLedger
from tools.daily_research.native_contact_lookup import step
from tools.daily_research.runner import Refusal, canonical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "set-scope", "run"])
    parser.add_argument("--date")
    parser.add_argument("--scope-file", type=Path)
    parser.add_argument("--expected-scope-id")
    args = parser.parse_args()
    bridge = Bridge()
    try:
        if args.command == "run":
            result = step(bridge, args.date)
        else:
            with FirestoreLedger(bridge, lease_scope="research_release").lock():
                if args.command == "prepare":
                    inputs = bridge.call("native_contact_prepare", day=args.date)
                    result = {
                        "state": "native_contact_inputs_prepared",
                        "provider_calls": 0,
                        "inputs": [
                            {
                                "input_id": value["input_id"],
                                "provenance": value["provenance"],
                                "source_pages": len(value["sources"]),
                            }
                            for value in inputs
                        ],
                    }
                else:
                    if args.scope_file is None:
                        raise Refusal("native_contact_scope_file_required")
                    value = json.loads(args.scope_file.read_text())
                    bridge.call(
                        "native_contact_scope_set",
                        expected_scope_id=args.expected_scope_id,
                        value=value,
                    )
                    result = {
                        "state": "native_contact_scope_configured",
                        "enabled": value["enabled"],
                        "provider_calls": 0,
                    }
        print(canonical(result))
    except (OSError, ValueError, Refusal) as exc:
        # No upstream text, config contents or credentials are command output.
        code = str(exc) if isinstance(exc, Refusal) else "native_contact_operator_refused"
        if not re.fullmatch(r"(?:native_contact|contact_research|firestore)_[a-z_]{1,90}", code):
            code = "native_contact_operator_refused"
        print(canonical({"state": "blocked", "error": code}))
        return 1
    finally:
        bridge.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
