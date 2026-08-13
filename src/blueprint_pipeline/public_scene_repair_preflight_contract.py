"""Backend-neutral schema identities accepted by repair consumers.

The residual schema is retained solely so immutable historical receipts remain
replayable.  Importing this module does not import or enable any legacy repair
backend.
"""

from __future__ import annotations


CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA_VERSION = "public_scene_aura_exact_residual_preflight.v1"


__all__ = ["CALIBRATED_RESIDUAL_PREFLIGHT_SCHEMA_VERSION"]
