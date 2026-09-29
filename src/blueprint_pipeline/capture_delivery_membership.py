"""Finite, exact source membership for an original website capture delivery."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath
from typing import Any

from .task_evaluation_scene_retirement_access import _document, _require

_GENERATION = re.compile(r"[1-9][0-9]{0,19}\Z", re.ASCII)
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z", re.ASCII)
_CRC = re.compile(r"[A-Za-z0-9+/]{6}==\Z", re.ASCII)
_ROW_KEYS = {"object_name", "relative_path", "generation", "size_bytes", "crc32c", "sha256"}


def _safe_path(value: Any) -> bool:
    return (type(value) is str and 0 < len(value.encode("utf-8")) <= 4096
            and not value.startswith("/") and "\\" not in value and "\x00" not in value
            and all(part not in ("", ".", "..") for part in value.split("/"))
            and PurePosixPath(value).as_posix() == value)


def validate_capture_delivery_membership(
    raw: bytes, *, selector: dict[str, Any], observation: dict[str, Any],
) -> dict[str, Any]:
    """Validate raw selected record bytes and all owner/marker/video joins.

    Caller must first read the record at selector.generation and verify its GCS
    metadata; this function additionally pins the exact downloaded bytes.
    """
    _require(type(raw) is bytes and 0 < len(raw) <= 65536,
             "capture_membership_bounds_invalid")
    _require(type(selector) is dict
             and set(selector) == {"object_name", "generation", "size_bytes", "sha256"}
             and _safe_path(selector["object_name"])
             and type(selector["generation"]) is str
             and _GENERATION.fullmatch(selector["generation"]) is not None
             and type(selector["size_bytes"]) is int
             and selector["size_bytes"] == len(raw)
             and type(selector["sha256"]) is str
             and selector["sha256"] == "sha256:" + hashlib.sha256(raw).hexdigest(),
             "capture_membership_selector_invalid")
    record = _document(raw)
    _require(set(record) == {"schema_version", "delivery_key", "source_finalize",
                             "producer_delivery", "raw", "derived"}
             and record["schema_version"] == "capture_delivery_membership.v1",
             "capture_membership_shape_invalid")
    marker = observation["completion_marker"]
    producer = observation["producer_delivery"]
    source = {"bucket": observation["bucket"], "object_name": marker["object_name"],
              "generation": marker["generation"]}
    key = hashlib.sha256(json.dumps([source["bucket"], source["object_name"],
                                     source["generation"]], separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()
    prefix = f"scenes/{observation['scene_id']}/captures/{observation['capture_id']}"
    delivery_prefix = f"{prefix}/deliveries/{key}/"
    _require(record["source_finalize"] == source and record["delivery_key"] == key
             and selector["object_name"] == delivery_prefix + "capture_delivery_membership.json",
             "capture_membership_delivery_mismatch")
    _require(record["producer_delivery"] == {
        "kind": producer["kind"],
        "receipt_object_name": producer["server_record"]["object_name"],
        "receipt_generation": producer["server_record"]["generation"],
    }, "capture_membership_receipt_mismatch")
    raw_rows, derived_rows = record["raw"], record["derived"]
    _require(type(raw_rows) is list and type(derived_rows) is list
             and 2 <= len(raw_rows) + len(derived_rows) <= 256,
             "capture_membership_rows_invalid")
    names: set[str] = set()
    destinations: set[str] = set()
    for kind, rows in (("raw", raw_rows), ("derived", derived_rows)):
        for row in rows:
            _require(type(row) is dict and set(row) == _ROW_KEYS
                     and _safe_path(row["object_name"])
                     and _safe_path(row["relative_path"])
                     and type(row["generation"]) is str
                     and _GENERATION.fullmatch(row["generation"]) is not None
                     and type(row["size_bytes"]) is int
                     and 0 < row["size_bytes"] <= (1 << 53) - 1
                     and type(row["crc32c"]) is str
                     and _CRC.fullmatch(row["crc32c"]) is not None
                     and type(row["sha256"]) is str
                     and _SHA.fullmatch(row["sha256"]) is not None,
                     "capture_membership_row_invalid")
            expected = (prefix + "/raw/" if kind == "raw" else delivery_prefix)
            relative = ("raw/" if kind == "raw" else f"deliveries/{key}/")
            _require(row["object_name"].startswith(expected)
                     and row["relative_path"] == relative + row["object_name"][len(expected):]
                     and row["object_name"] not in names
                     and row["relative_path"] not in destinations,
                     "capture_membership_row_identity_invalid")
            names.add(row["object_name"])
            destinations.add(row["relative_path"])
    marker_rows = [row for row in raw_rows if row["object_name"] == marker["object_name"]]
    video = producer["raw_video"]
    video_rows = [row for row in raw_rows if row["object_name"] == video["object_name"]]
    _require(len(marker_rows) == len(video_rows) == 1
             and all(marker_rows[0][field] == marker[field]
                     for field in ("generation", "size_bytes", "sha256"))
             and all(video_rows[0][field] == video[field]
                     for field in ("generation", "size_bytes", "crc32c")),
             "capture_membership_original_input_mismatch")
    if producer["kind"] == "website_browser_capture_delivery":
        _require(any(row["relative_path"] == "raw/manifest.json" for row in raw_rows)
                 and len(raw_rows) == 3, "capture_membership_browser_inputs_invalid")
    return record
