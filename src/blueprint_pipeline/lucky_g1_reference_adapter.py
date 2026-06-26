"""Run the public Lucky Robots G1 manipulation challenge as a Blueprint adapter."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json

LUCKY_G1_REFERENCE_ADAPTER_SCHEMA_VERSION = "lucky_g1_reference_adapter_manifest.v1"
LUCKY_G1_REFERENCE_TRACE_SCHEMA_VERSION = "lucky_g1_reference_trace.v1"
DEFAULT_OUTPUT_RELATIVE = "pipeline/simulation_automation/lucky_g1_reference_adapter"
DEFAULT_REPO_URL = "https://github.com/luckyrobots/g1-manipulation-challenge.git"
REQUIRED_FILES = [
    "README.md",
    "run.py",
    "scene.xml",
    "g1.xml",
    "model_config.json",
    "walker.onnx",
    "croucher.onnx",
    "rotator.onnx",
    "right_reacher.onnx",
]


def _string(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _run_git(args: Sequence[str], cwd: Path | None = None) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def _default_cache_root() -> Path:
    return Path(os.environ.get("BLUEPRINT_LUCKY_G1_CACHE_DIR", "~/.cache/blueprint")).expanduser() / (
        "g1-manipulation-challenge"
    )


def _validate_lucky_root(root: Path) -> tuple[bool, list[str]]:
    missing = [name for name in REQUIRED_FILES if not (root / name).is_file()]
    if not (root / "assets").is_dir():
        missing.append("assets/")
    return not missing, missing


def resolve_lucky_g1_reference_root(
    *,
    lucky_root: str | Path | None = None,
    fetch_if_missing: bool = False,
    repo_url: str = DEFAULT_REPO_URL,
) -> tuple[Path | None, list[str], str | None]:
    candidates = [
        lucky_root,
        os.environ.get("BLUEPRINT_LUCKY_G1_CHALLENGE_ROOT"),
        _default_cache_root(),
    ]
    checked: list[str] = []
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        checked.append(str(root))
        ok, missing = _validate_lucky_root(root)
        if ok:
            commit = None
            try:
                commit = _run_git(["rev-parse", "HEAD"], cwd=root)
            except Exception:
                commit = None
            return root, [], commit
        if fetch_if_missing and root == _default_cache_root().resolve():
            ensure_dir(root.parent)
            if root.exists() and not (root / ".git").is_dir():
                return None, [f"cache_path_exists_but_is_not_git_repo:{root}"], None
            if not root.exists():
                _run_git(["clone", "--depth=1", repo_url, str(root)])
            else:
                _run_git(["fetch", "--depth=1", "origin", "main"], cwd=root)
                _run_git(["checkout", "FETCH_HEAD"], cwd=root)
            ok, missing = _validate_lucky_root(root)
            if ok:
                commit = _run_git(["rev-parse", "HEAD"], cwd=root)
                return root, [], commit
            return None, [f"missing_after_fetch:{item}" for item in missing], None
    return None, [f"missing_lucky_g1_challenge_root_checked:{path}" for path in checked], None


def _load_lucky_run_module(root: Path) -> Any:
    spec = importlib.util.spec_from_file_location("blueprint_lucky_g1_run", root / "run.py")
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable_to_load_lucky_run_py:{root / 'run.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(root))
    try:
        spec.loader.exec_module(module)
    finally:
        try:
            sys.path.remove(str(root))
        except ValueError:
            pass
    return module


def _contact_records(model: Any, data: Any, mujoco: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        geom_ids = [int(contact.geom[0]), int(contact.geom[1])]
        geom_names = [
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or f"geom_{geom_id}"
            for geom_id in geom_ids
        ]
        records.append(
            {
                "contact_index": index,
                "geom_ids": geom_ids,
                "geom_names": geom_names,
                "distance": round(float(getattr(contact, "dist", 0.0) or 0.0), 9),
                "right_hand_object_contact": (
                    any("right_hand" in name or "right_wrist" in name for name in geom_names)
                    and any(name in {"red_cylinder", "red_cap_top", "red_cap_bot"} for name in geom_names)
                ),
            }
        )
    return records


def _write_trace_video(
    *,
    out_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    generated_at: str,
) -> dict[str, Any]:
    manifest_path = out_dir / "lucky_g1_reference_video_manifest.json"
    try:
        from PIL import Image, ImageDraw
    except Exception as exc:  # pragma: no cover
        manifest = {
            "schema_version": "lucky_g1_reference_video_manifest.v1",
            "generated_at": generated_at,
            "status": "blocked_pillow_unavailable",
            "error": str(exc),
            "videos": [],
        }
        write_json(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path)}

    sampled = list(rows[:: max(1, len(rows) // 120)]) or list(rows)
    width, height = 720, 460
    points = []
    for row in sampled:
        for key in ("red_block_pose_xyz", "right_palm_pose_xyz"):
            value = row.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                points.append(value)
    if not points:
        points = [[0.0, 0.0, 0.0]]
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    min_x, max_x = min(xs) - 0.25, max(xs) + 0.25
    min_y, max_y = min(ys) - 0.25, max(ys) + 0.25

    def to_px(point: Sequence[float]) -> tuple[int, int]:
        x = (float(point[0]) - min_x) / max(0.001, max_x - min_x)
        y = (float(point[1]) - min_y) / max(0.001, max_y - min_y)
        return int(48 + x * (width - 96)), int(height - 50 - y * (height - 110))

    frames = []
    object_path: list[tuple[int, int]] = []
    for row in sampled:
        frame = Image.new("RGB", (width, height), (248, 249, 247))
        draw = ImageDraw.Draw(frame)
        draw.rectangle((0, 0, width, 42), fill=(24, 28, 34))
        draw.text((18, 13), "Lucky G1 walker/reacher adapter trace", fill=(245, 245, 245))
        red = row.get("red_block_pose_xyz") or [0.0, 0.0, 0.0]
        palm = row.get("right_palm_pose_xyz") or [0.0, 0.0, 0.0]
        red_px = to_px(red)
        palm_px = to_px(palm)
        object_path.append(red_px)
        if len(object_path) > 1:
            draw.line(object_path, fill=(198, 54, 44), width=3)
        draw.ellipse((red_px[0] - 10, red_px[1] - 10, red_px[0] + 10, red_px[1] + 10), fill=(210, 55, 45))
        draw.line((palm_px[0] - 14, palm_px[1], palm_px[0] + 14, palm_px[1]), fill=(36, 88, 190), width=4)
        draw.line((palm_px[0], palm_px[1] - 14, palm_px[0], palm_px[1] + 14), fill=(36, 88, 190), width=4)
        draw.text(
            (18, height - 30),
            f"phase={row.get('phase')} walker={row.get('walker_policy_executed')} "
            f"reacher={row.get('right_reacher_policy_executed')}",
            fill=(32, 35, 39),
        )
        frames.append(frame)

    gif_path = out_dir / "lucky_g1_reference_overview.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=90,
            loop=0,
            optimize=False,
        )
    manifest = {
        "schema_version": "lucky_g1_reference_video_manifest.v1",
        "generated_at": generated_at,
        "status": "complete" if gif_path.is_file() else "blocked_no_frames",
        "video_kind": "trace_derived_reference_animation",
        "frame_count": len(frames),
        "videos": [
            {
                "artifact_id": "lucky_g1_reference_overview_video",
                "path": str(gif_path),
                "format": "gif",
                "source": "lucky_g1_reference_trace",
            }
        ]
        if gif_path.is_file()
        else [],
    }
    write_json(manifest_path, manifest)
    return {**manifest, "manifest_path": str(manifest_path)}


def _blocked_manifest(
    *,
    out_dir: Path,
    generated_at: str,
    blockers: Sequence[str],
    repo_url: str,
) -> dict[str, Any]:
    manifest = {
        "schema_version": LUCKY_G1_REFERENCE_ADAPTER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "blocked",
        "repo_url": repo_url,
        "blockers": list(blockers),
        "official_lucky_walker_reacher_policy_assets_executed": False,
        "lucky_g1_reference_adapter_ready": False,
        "claim_boundary": {
            "official_lucky_walker_reacher_policy_assets_executed": False,
            "official_lucky_pick_place_physics_validated": False,
            "blueprint_tote_task_validated_by_lucky_assets": False,
        },
    }
    path = out_dir / "lucky_g1_reference_adapter_manifest.json"
    write_json(path, manifest)
    return {**manifest, "output_path": str(path), "artifacts": {"manifest": str(path)}}


def run_lucky_g1_reference_adapter(
    *,
    capture_root: str | Path,
    output_dir: str | Path | None = None,
    lucky_root: str | Path | None = None,
    fetch_if_missing: bool = False,
    repo_url: str = DEFAULT_REPO_URL,
    steps: int = 360,
) -> dict[str, Any]:
    generated_at = utc_now_iso()
    root = Path(capture_root).expanduser().resolve()
    out_dir = Path(output_dir).expanduser().resolve() if output_dir else root / DEFAULT_OUTPUT_RELATIVE
    ensure_dir(out_dir)
    lucky_path, blockers, commit = resolve_lucky_g1_reference_root(
        lucky_root=lucky_root,
        fetch_if_missing=fetch_if_missing,
        repo_url=repo_url,
    )
    if lucky_path is None:
        return _blocked_manifest(
            out_dir=out_dir,
            generated_at=generated_at,
            blockers=blockers or ["lucky_g1_challenge_root_unavailable"],
            repo_url=repo_url,
        )

    try:
        import mujoco
        import numpy as np
    except Exception as exc:  # pragma: no cover
        return _blocked_manifest(
            out_dir=out_dir,
            generated_at=generated_at,
            blockers=[f"missing_runtime_dependency:{type(exc).__name__}:{exc}"],
            repo_url=repo_url,
        )

    module = _load_lucky_run_module(lucky_path)
    config = json.loads((lucky_path / "model_config.json").read_text(encoding="utf-8"))
    model = mujoco.MjModel.from_xml_path(str(lucky_path / "scene.xml"))
    model.opt.timestep = 0.005
    if hasattr(module, "set_armature"):
        module.set_armature(model, config["joint_names"])
    data = mujoco.MjData(model)
    data.qpos[0] = -0.6
    data.qpos[2] = 0.76
    data.qpos[3:7] = [1, 0, 0, 0]
    for name, value in config["default_joint_pos"].items():
        if name in config["joint_names"]:
            data.qpos[7 + config["joint_names"].index(name)] = value
    mujoco.mj_forward(model, data)

    walker = module.ONNXPolicy(str(lucky_path / "walker.onnx"))
    croucher = module.ONNXPolicy(str(lucky_path / "croucher.onnx"))
    rotator = module.ONNXPolicy(str(lucky_path / "rotator.onnx"))
    right_reacher = module.ONNXPolicy(str(lucky_path / "right_reacher.onnx"))
    with open(os.devnull, "w", encoding="utf-8") as devnull, contextlib.redirect_stdout(devnull):
        controller = module.G1Controller(
            model,
            data,
            walker,
            croucher,
            rotator,
            config,
            right_reacher=right_reacher,
        )

    # Warm up the supplied policies and prove their I/O shapes are executable.
    walker(np.zeros(99, dtype=np.float32))
    right_reacher(np.zeros(36, dtype=np.float32))

    red_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red_block")
    palm_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_palm")
    camera_ids = {
        "head_cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "head_cam"),
        "wrist_cam": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_cam"),
    }
    actuator_count = int(model.nu)
    right_finger_actuator_count = len(getattr(controller, "right_finger_actuators", []))
    right_arm_joint_count = len(getattr(controller, "right_arm_indices", []))

    phase_plan = [
        ("walk_to_source", 80, {"lin_vel_x": 0.25, "lin_vel_y": 0.0, "ang_vel_z": 0.0}),
        (
            "reach_to_object",
            90,
            {"lin_vel_x": 0.0, "reach_active": True, "reach_target": [0.34, -0.20, 0.06]},
        ),
        (
            "close_hand",
            60,
            {"reach_active": True, "grip_closed": True, "reach_target": [0.34, -0.20, 0.04]},
        ),
        (
            "lift_attempt",
            80,
            {"reach_active": True, "grip_closed": True, "reach_target": [0.30, -0.18, 0.24]},
        ),
        (
            "place_release_attempt",
            70,
            {"reach_active": True, "grip_closed": False, "reach_target": [0.24, -0.42, 0.12]},
        ),
    ]

    rows: list[dict[str, Any]] = []
    contact_sample_count = 0
    hand_object_contact_count = 0
    walker_action_seen = False
    reacher_action_seen = False
    step = 0
    for phase, phase_steps, settings in phase_plan:
        controller.lin_vel_x = float(settings.get("lin_vel_x", 0.0))
        controller.lin_vel_y = float(settings.get("lin_vel_y", 0.0))
        controller.ang_vel_z = float(settings.get("ang_vel_z", 0.0))
        if "reach_active" in settings:
            controller.reach_active = bool(settings["reach_active"])
            controller.input_mode = "reach" if controller.reach_active else "walk"
        if "reach_target" in settings:
            controller.reach_target = np.array(settings["reach_target"], dtype=np.float32)
        if "grip_closed" in settings:
            controller.grip_closed = bool(settings["grip_closed"])
        for phase_step in range(phase_steps):
            target_pos = controller.step()
            controller.apply_pd_control(target_pos)
            mujoco.mj_step(model, data)
            walker_action_seen = walker_action_seen or bool(np.linalg.norm(controller.last_action) > 0.0)
            reacher_action_seen = reacher_action_seen or bool(
                controller.reach_active and np.linalg.norm(controller.last_arm_action) > 0.0
            )
            contacts = _contact_records(model, data, mujoco)
            contact_sample_count += len(contacts)
            hand_object_contact_count += sum(1 for record in contacts if record["right_hand_object_contact"])
            if step % 10 == 0 or phase_step == phase_steps - 1:
                rows.append(
                    {
                        "schema_version": LUCKY_G1_REFERENCE_TRACE_SCHEMA_VERSION,
                        "step": step,
                        "sim_time_s": round(float(data.time), 9),
                        "phase": phase,
                        "walker_policy_executed": walker_action_seen,
                        "right_reacher_policy_executed": reacher_action_seen,
                        "grip_closed": bool(controller.grip_closed),
                        "reach_target_pelvis_xyz": [
                            round(float(value), 6) for value in controller.reach_target.tolist()
                        ],
                        "base_pose_xyz": [round(float(value), 6) for value in data.qpos[:3].tolist()],
                        "right_palm_pose_xyz": [
                            round(float(value), 6) for value in data.site_xpos[palm_site_id].tolist()
                        ],
                        "red_block_pose_xyz": [
                            round(float(value), 6) for value in data.xpos[red_body_id].tolist()
                        ],
                        "last_walker_action_norm": round(float(np.linalg.norm(controller.last_action)), 6),
                        "last_reacher_action_norm": round(float(np.linalg.norm(controller.last_arm_action)), 6),
                        "contacts": contacts,
                    }
                )
            step += 1

    trace_path = out_dir / "lucky_g1_reference_trace.jsonl"
    with trace_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    video_manifest = _write_trace_video(out_dir=out_dir, rows=rows, generated_at=generated_at)

    red_initial_z = rows[0]["red_block_pose_xyz"][2] if rows else None
    red_final = rows[-1]["red_block_pose_xyz"] if rows else None
    red_max_lift = 0.0
    if rows and red_initial_z is not None:
        red_max_lift = max(float(row["red_block_pose_xyz"][2]) - float(red_initial_z) for row in rows)
    grab_logic_executed = any(bool(row.get("grip_closed")) for row in rows)
    place_logic_executed = any(row.get("phase") == "place_release_attempt" for row in rows)
    policy_assets_executed = bool(walker_action_seen and reacher_action_seen)
    manifest = {
        "schema_version": LUCKY_G1_REFERENCE_ADAPTER_SCHEMA_VERSION,
        "generated_at": generated_at,
        "status": "complete" if policy_assets_executed and grab_logic_executed and place_logic_executed else "blocked",
        "repo_url": repo_url,
        "repo_commit": commit,
        "lucky_root": str(lucky_path),
        "model": {
            "scene_xml": str(lucky_path / "scene.xml"),
            "g1_xml": str(lucky_path / "g1.xml"),
            "model_config": str(lucky_path / "model_config.json"),
            "nq": int(model.nq),
            "nu": actuator_count,
            "right_arm_joint_count": right_arm_joint_count,
            "right_finger_actuator_count": right_finger_actuator_count,
            "head_camera_available": camera_ids["head_cam"] >= 0,
            "wrist_camera_available": camera_ids["wrist_cam"] >= 0,
        },
        "policies": {
            "walker": {
                "path": str(lucky_path / "walker.onnx"),
                "input_dim": 99,
                "output_dim": 29,
                "executed": walker_action_seen,
            },
            "right_reacher": {
                "path": str(lucky_path / "right_reacher.onnx"),
                "input_dim": 36,
                "output_dim": 7,
                "executed": reacher_action_seen,
            },
        },
        "official_lucky_walker_reacher_policy_assets_executed": policy_assets_executed,
        "grab_logic_executed": grab_logic_executed,
        "place_logic_executed": place_logic_executed,
        "lucky_g1_reference_adapter_ready": policy_assets_executed,
        "metrics": {
            "trace_sample_count": len(rows),
            "contact_sample_count": contact_sample_count,
            "hand_object_contact_count": hand_object_contact_count,
            "red_block_max_lift_delta_m": round(float(red_max_lift), 6),
            "red_block_final_pose_xyz": red_final,
        },
        "artifacts": {
            "lucky_g1_reference_trace": str(trace_path),
            "lucky_g1_reference_video_manifest": str(video_manifest["manifest_path"]),
            "lucky_g1_reference_overview_video": str(out_dir / "lucky_g1_reference_overview.gif")
            if video_manifest.get("status") == "complete"
            else None,
        },
        "claim_boundary": {
            "official_lucky_walker_reacher_policy_assets_executed": policy_assets_executed,
            "official_lucky_pick_place_physics_validated": False,
            "blueprint_tote_task_validated_by_lucky_assets": False,
            "blueprint_proxy_tote_physics_claimed_by_this_adapter": False,
            "generated_world_rank_fidelity_result_proven": False,
        },
        "blockers": []
        if policy_assets_executed
        else ["lucky_walker_or_right_reacher_policy_execution_missing"],
    }
    output_path = out_dir / "lucky_g1_reference_adapter_manifest.json"
    write_json(output_path, manifest)
    return {**manifest, "output_path": str(output_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Lucky G1 reference adapter proof")
    parser.add_argument("--capture-root", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--lucky-root")
    parser.add_argument("--fetch-if-missing", action="store_true")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--steps", type=int, default=360)
    args = parser.parse_args(argv)
    result = run_lucky_g1_reference_adapter(
        capture_root=args.capture_root,
        output_dir=args.output_dir,
        lucky_root=args.lucky_root,
        fetch_if_missing=args.fetch_if_missing,
        repo_url=args.repo_url,
        steps=args.steps,
    )
    print(result["output_path"])
    print(result["status"])
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
