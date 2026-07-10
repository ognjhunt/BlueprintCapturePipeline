# SPEC-12: Make CPU safety gates actually run (`pxr`/`mujoco` in the canonical env + CI)

> [!WARNING]
> **SUPERSEDED FOR CURRENT LAUNCH STATUS.** This file is historical evidence, not a current completion or launch decision.
> Use the [current 107-gap ledger](/docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json) and the [July 9 source audit](/docs/PUBLIC_LAUNCH_SC3_QUALITY_GAP_AUDIT_2026-07-09.md). Do not infer current status from “proposed,” “implemented,” or “fixed” wording below.

- Status: Proposed
- Priority: **P1 — major** (protects against paid-GPU spend on broken renders)
- Area: canonical `.venv`, CI workflow, `tests/test_cpu_env_contract.py` consumers

## Problem

The no-GPU dry-render / scene-placement / POV-framing gates exist specifically to catch
the invisible-robot and pitched-down-crop render bugs *before* paid GPU spend — but they
`pytest.importorskip` on `pxr` and `mujoco`:

- `tests/test_local_render_preview.py` (7× `importorskip("pxr")`: 89, 275, 353, 383, 422, 442, 471)
- `tests/test_scene_placement.py:618,682,716` (`pxr`)
- `tests/test_robot_eval_job_orchestrator.py:6899-6900`, `tests/test_manipulation_task_stack.py:42,137` (`mujoco`/`trimesh`)

Per `docs/cpu-work-audit-2026-06-29.md` (CRIT-02), `pxr` (usd-core) and `mujoco` are
missing from the canonical `.venv`, so these gates **skip green**. The repo has since
added `tests/test_cpu_env_contract.py`, which fails loudly when the extras are absent —
verified in this audit: a fresh `pip install -e .` produces 8 env-contract failures and
~80 skips, exactly as designed. The remaining gap is that the canonical env and CI don't
guarantee the extras, so the contract test is the only tripwire and a runner without it
in scope still skips the real gates.

## Why this matters for launch

Every GPU render run costs real money and burns operator time; the two known historical
render bugs (invisible robot, pitched-down crop) are exactly what these gates catch. A
green suite that silently skipped them is a false safety signal during the launch push.

## Proposed fix

1. **Canonical env installs the dev extra by default:** `uv sync --extra dev` (or
   `pip install -e '.[dev]'`, per `docs/DEV_SETUP.md`) becomes the documented + scripted
   single path (Makefile target `make env`); `cpu_env_doctor` verifies it. Note:
   `.[geometry,cloud]` alone is insufficient — the env contract also requires `cv2`
   (`opencv-python-headless`), which ships in the `dev`/`runtime` extras, and without it
   ~32 video/WAM validation tests skip.
2. **CI job matrix includes a leg with the dev extra installed** that runs the
   dry-render/placement/POV suites and **fails if they skip**: run with
   `-W error::pytest.PytestUnraisableExceptionWarning` plus a skip-budget check
   (e.g. `--strict-skip` wrapper or asserting `skipped == 0` for those files via
   `pytest --collect-only`/junitxml parsing).
3. **Skip-visibility:** the env-contract test already covers interpreter drift; extend it
   to write `cpu_gate_status.json` (ran/skipped per gate family) that the launch gates
   (SPEC-11) consume, so "CPU gates actually executed" becomes a machine-checked
   precondition for any GPU-spend run packet (`first_gpu_*` flows).

## Acceptance criteria

- [ ] Fresh-clone `make env && pytest` runs the dry-render/placement/POV gates with zero skips attributable to missing `pxr`/`mujoco`.
- [ ] CI fails when those suites skip.
- [ ] `first_gpu_run_packet` / GPU preflight refuses to proceed unless `cpu_gate_status.json` shows the gate families executed at current HEAD.
