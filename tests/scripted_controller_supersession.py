"""Deliberate, reviewed revisions of pinned scripted-controller sources.

A predecessor snapshot proves that later body additions changed no existing
controller byte. A revision that intentionally changes one is listed here with
its reason; only that exact digest substitution is accepted.
"""
from __future__ import annotations

import json

SUPERSEDED = {
    # Accept the tool link Isaac Sim 6 authors for TRLC (tn__*_link67_*) from its
    # source name link6-7; scripted_robot_grasp_frames._SOURCE_TOOLS.
    "sha256:d0ae2f89fcb2218523dbbbb5f11beb812c8d2bd5e5dd1a892af97f8cee20347d":
        "sha256:9f3ed5b722aba7bbfd19952de6d1712683de88a0465bbe8b030326fcda9978e9",
}


def current(snapshot):
    """The snapshot with each superseded digest replaced by its reviewed revision."""
    text = json.dumps(snapshot)
    for old, new in SUPERSEDED.items():
        text = text.replace(old, new)
    return json.loads(text)
