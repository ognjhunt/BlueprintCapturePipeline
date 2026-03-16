from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
CONTRACTS_SRC_DIR = REPO_ROOT.parent / "BlueprintContracts" / "src"

for candidate in (SRC_DIR, CONTRACTS_SRC_DIR):
    if candidate.is_dir():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
