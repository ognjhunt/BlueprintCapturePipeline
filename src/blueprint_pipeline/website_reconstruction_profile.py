"""Provider-specific image capacity; never a global reconstruction frame cap."""
from __future__ import annotations

import json
import os
from typing import Any, Mapping


def reconstruction_profile(configured: Mapping[str, Any] | None = None) -> dict[str, Any]:
    # A future provider/model supplies its verified capability here. Atlas is
    # not assigned an invented hard limit or enabled before its API is available.
    selected_model = os.environ.get("WORLDLABS_DEFAULT_MODEL", "").strip()
    if configured is None and selected_model not in ("", "marble-1.1-plus") and not os.environ.get("BLUEPRINT_WEBSITE_RECONSTRUCTION_PROFILE_JSON"):
        raise ValueError("website_reconstruction_profile_required_for_selected_model")
    profile = dict(configured) if configured is not None else json.loads(
        os.environ.get("BLUEPRINT_WEBSITE_RECONSTRUCTION_PROFILE_JSON") or
        '{"provider":"world_labs","model":"marble-1.1-plus","max_input_images":8}')
    if not isinstance(profile, dict):
        raise ValueError("website_reconstruction_profile_invalid")
    if configured is None and selected_model and profile.get("model") != selected_model:
        raise ValueError("website_reconstruction_profile_model_mismatch")
    count = profile.get("max_input_images")
    if (not isinstance(profile.get("provider"), str) or not profile["provider"].strip()
            or not isinstance(profile.get("model"), str) or not profile["model"].strip()
            or isinstance(count, bool) or not isinstance(count, int) or count < 2):
        raise ValueError("website_reconstruction_profile_invalid")
    return profile
