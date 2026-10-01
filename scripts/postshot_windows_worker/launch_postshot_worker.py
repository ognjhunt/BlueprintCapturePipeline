#!/usr/bin/env python3
"""Retired AWS Postshot worker command; historical receipts stay on disk."""
import json


def main() -> int:
    print(json.dumps({"status": "blocked", "blockers": ["aws_provider_integration_removed"],
                      "provider_mutations_performed": 0, "provider_absence_confirmed": False}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
