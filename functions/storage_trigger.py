"""Storage trigger for NuRec-first swap orchestration."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from blueprint_pipeline.swap_orchestrator import run_swap_pipeline


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_DESCRIPTOR_PATTERN = re.compile(
    r"^scenes/(?P<scene_id>[^/]+)/captures/(?P<capture_id>[^/]+)/capture_descriptor\.json$"
)


def parse_descriptor_path(object_name: str) -> Optional[Dict[str, str]]:
    match = _DESCRIPTOR_PATTERN.match(object_name)
    if not match:
        return None
    data = match.groupdict()
    data["object_name"] = object_name
    return data


def on_storage_finalize(event: Dict[str, Any], context: Any) -> None:  # noqa: ARG001
    bucket = str(event.get("bucket") or "")
    object_name = str(event.get("name") or "")

    if not bucket or not object_name:
        logger.warning("Storage event missing bucket/name: %s", event)
        return

    parsed = parse_descriptor_path(object_name)
    if parsed is None:
        logger.debug("Ignoring non-descriptor object: gs://%s/%s", bucket, object_name)
        return

    descriptor_uri = f"gs://{bucket}/{object_name}"
    logger.info(
        "Triggering swap orchestrator for scene=%s capture=%s descriptor=%s",
        parsed["scene_id"],
        parsed["capture_id"],
        descriptor_uri,
    )

    run_swap_pipeline(descriptor_gcs_uri=descriptor_uri)
