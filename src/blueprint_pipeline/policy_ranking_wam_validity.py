"""Label-blind action/motion diagnostics for frozen OSCAR policy rollouts.

RoboArena NPZs use NumPy object arrays.  This module never calls
``numpy.load(..., allow_pickle=True)``.  It inspects the pickle opcode globals,
uses a three-class restricted unpickler, and then validates every record into
plain finite numeric arrays before analysis.

The resulting motion correlation is a contradiction diagnostic, not a task
success score and not proof that the generated robot followed the action in 3D.
"""

from __future__ import annotations

import argparse
import importlib
import io
import json
import math
import pickle
import pickletools
import struct
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2  # type: ignore[import-not-found]
import numpy as np

from .common import write_json
from .policy_ranking_thesis import canonical_sha256, file_sha256


SCHEMA_VERSION = "policy_ranking_wam_action_motion_diagnostic.v1"
MAX_UNCOMPRESSED_NPY_BYTES = 10_000_000
ALLOWED_PICKLE_GLOBALS = {
    ("numpy.core.multiarray", "_reconstruct"),
    ("numpy._core.multiarray", "_reconstruct"),
    ("numpy", "ndarray"),
    ("numpy", "dtype"),
}
EXPECTED_RECORD_LENGTHS = {
    "cartesian_position": 6,
    "joint_position": 7,
    "gripper_position": 1,
    "action": 8,
}


class _RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if (module, name) not in ALLOWED_PICKLE_GLOBALS:
            raise pickle.UnpicklingError(f"forbidden_global:{module}.{name}")
        if name == "_reconstruct":
            return importlib.import_module(module)._reconstruct
        if (module, name) == ("numpy", "ndarray"):
            return np.ndarray
        return np.dtype


def _npy_payload(raw: bytes) -> tuple[bytes, tuple[int, ...], str]:
    if not raw.startswith(b"\x93NUMPY") or len(raw) < 12:
        raise ValueError("invalid_npy_magic")
    major = raw[6]
    if major == 1:
        header_len = struct.unpack("<H", raw[8:10])[0]
        offset = 10
    elif major in {2, 3}:
        header_len = struct.unpack("<I", raw[8:12])[0]
        offset = 12
    else:
        raise ValueError(f"unsupported_npy_version:{major}")
    header = raw[offset : offset + header_len].decode("latin1")
    if "'descr': '|O'" not in header and '"descr": "|O"' not in header:
        raise ValueError("expected_object_array")
    import ast

    parsed = ast.literal_eval(header.strip())
    shape = tuple(int(item) for item in parsed.get("shape", ()))
    return raw[offset + header_len :], shape, str(parsed.get("descr"))


def load_restricted_roboarena_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Return validated numeric arrays without permitting arbitrary pickle globals."""

    resolved = Path(path).resolve()
    with zipfile.ZipFile(resolved) as archive:
        names = archive.namelist()
        if names != ["data.npy"]:
            raise ValueError(f"unexpected_npz_members:{names}")
        info = archive.getinfo("data.npy")
        if info.file_size > MAX_UNCOMPRESSED_NPY_BYTES:
            raise ValueError("npz_member_too_large")
        raw = archive.read("data.npy")
    payload, declared_shape, _ = _npy_payload(raw)
    globals_seen = {
        tuple(str(arg).split(" ", 1))
        for op, arg, _ in pickletools.genops(payload)
        if op.name == "GLOBAL"
    }
    if not globals_seen.issubset(ALLOWED_PICKLE_GLOBALS):
        raise ValueError(f"forbidden_pickle_globals:{sorted(globals_seen - ALLOWED_PICKLE_GLOBALS)}")
    loaded = _RestrictedNumpyUnpickler(io.BytesIO(payload)).load()
    if not isinstance(loaded, np.ndarray) or loaded.dtype != object:
        raise ValueError("restricted_payload_not_object_array")
    if loaded.shape != declared_shape or loaded.ndim != 1 or not 1 <= loaded.size <= 10_000:
        raise ValueError("restricted_payload_shape")
    columns: dict[str, list[list[float]]] = {key: [] for key in EXPECTED_RECORD_LENGTHS}
    for record in loaded:
        if not isinstance(record, dict) or set(record) != set(EXPECTED_RECORD_LENGTHS):
            raise ValueError("invalid_record_keys")
        for key, width in EXPECTED_RECORD_LENGTHS.items():
            value = record[key]
            if not isinstance(value, list) or len(value) != width:
                raise ValueError(f"invalid_record_width:{key}")
            row = [float(item) for item in value]
            if not all(math.isfinite(item) for item in row):
                raise ValueError(f"nonfinite_record:{key}")
            columns[key].append(row)
    return {key: np.asarray(value, dtype=np.float64) for key, value in columns.items()}


def _generated_motion(video_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError("video_open_failed")
    previous: np.ndarray | None = None
    motion: list[float] = []
    frame_count = width = height = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            height, full_width = frame.shape[:2]
            width = full_width // 2
            generated = cv2.cvtColor(frame[:, :width], cv2.COLOR_BGR2GRAY)
            generated = cv2.resize(generated, (160, 120), interpolation=cv2.INTER_AREA)
            current = generated.astype(np.float32) / 255.0
            if previous is not None:
                motion.append(float(np.mean(np.abs(current - previous))))
            previous = current
            frame_count += 1
    finally:
        capture.release()
    if frame_count < 2:
        raise ValueError("insufficient_video_frames")
    return np.asarray(motion, dtype=np.float64), {
        "frame_count": frame_count,
        "generated_crop_pixels": [0, 0, width, height],
        "third_party_physical_pixels_decoded_for_metric": False,
    }


def _resample(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.asarray([], dtype=np.float64)
    edges = np.linspace(0, len(values), count + 1)
    rows = []
    for index in range(count):
        lo, hi = int(math.floor(edges[index])), int(math.ceil(edges[index + 1]))
        rows.append(float(np.mean(values[lo:max(lo + 1, hi)])))
    return np.asarray(rows, dtype=np.float64)


def analyze_action_motion(video_path: str | Path, npz_path: str | Path) -> dict[str, Any]:
    video = Path(video_path).resolve()
    action_file = Path(npz_path).resolve()
    arrays = load_restricted_roboarena_npz(action_file)
    action_magnitude = np.linalg.norm(arrays["action"][:, :7], axis=1)
    frame_motion, video_meta = _generated_motion(video)
    action_resampled = _resample(action_magnitude, len(frame_motion))
    correlation: float | None = None
    if np.std(action_resampled) > 1e-9 and np.std(frame_motion) > 1e-9:
        correlation = float(np.corrcoef(action_resampled, frame_motion)[0, 1])
    median_motion = float(np.median(frame_motion))
    result = {
        "video_sha256": file_sha256(video),
        "npz_sha256": file_sha256(action_file),
        "action_step_count": int(len(action_magnitude)),
        "action_dimension": int(arrays["action"].shape[1]),
        "mean_action_magnitude": float(np.mean(action_magnitude)),
        "cartesian_start_end_displacement": float(
            np.linalg.norm(arrays["cartesian_position"][-1, :3] - arrays["cartesian_position"][0, :3])
        ),
        "generated_adjacent_motion_mean": float(np.mean(frame_motion)),
        "generated_adjacent_motion_median": median_motion,
        "generated_adjacent_motion_p95": float(np.quantile(frame_motion, 0.95)),
        "generated_jump_to_median_ratio": float(np.max(frame_motion) / max(median_motion, 1e-9)),
        "action_motion_pearson": correlation,
        "security": {
            "numpy_allow_pickle_used": False,
            "restricted_pickle_globals": sorted(".".join(item) for item in ALLOWED_PICKLE_GLOBALS),
        },
        "claim_boundary": "pixel_motion_correlation_diagnostic_not_3d_action_following_or_task_success",
        **video_meta,
    }
    return result


def build_report(
    inventory: Mapping[str, Any], *, roboarena_root: str | Path
) -> dict[str, Any]:
    root = Path(roboarena_root).resolve()
    rows: list[dict[str, Any]] = []
    blockers: list[str] = []
    seen: set[tuple[str, str]] = set()
    for request in inventory.get("requests", []):
        if not isinstance(request, Mapping):
            continue
        identity = (str(request.get("session_id")), str(request.get("policy_id")))
        if identity in seen:
            continue
        seen.add(identity)
        session_id, policy_id = identity
        candidates = sorted((root / "evaluation_sessions" / session_id).glob(f"*_{policy_id}/*.npz"))
        if len(candidates) != 1:
            blockers.append(f"npz_resolution:{session_id}:{policy_id}:{len(candidates)}")
            continue
        try:
            metrics = analyze_action_motion(str(request["video_path"]), candidates[0])
        except Exception as exc:  # noqa: BLE001 - preserve fail-closed diagnostics
            blockers.append(f"analysis_failed:{session_id}:{policy_id}:{type(exc).__name__}")
            continue
        rows.append({"session_id": session_id, "policy_id": policy_id, **metrics})
    expected = len(seen)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed" if not blockers and len(rows) == expected else "blocked",
        "inventory_sha256": inventory.get("inventory_sha256"),
        "row_count": len(rows),
        "rows": rows,
        "blockers": sorted(set(blockers)),
        "benchmark_labels_seen": False,
        "third_party_physical_video_pixels_seen": False,
        "physical_proprioception_used_for_action_reference": True,
        "task_success_scored": False,
    }
    result["report_sha256"] = canonical_sha256(result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--roboarena-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    report = build_report(
        json.loads(Path(args.inventory).read_text(encoding="utf-8")),
        roboarena_root=args.roboarena_root,
    )
    write_json(Path(args.output), report)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
