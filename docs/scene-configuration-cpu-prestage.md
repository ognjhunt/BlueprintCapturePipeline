# Scene configuration without a GPU for authoring

## What actually needs the Isaac host

| Stage | Capability | Needs | Class today |
| --- | --- | --- | --- |
| 1 | observed appearance object removal (website: prepared background pass-through) | CPU | `no_spend` |
| 2 | collision object excision (website: pass-through) | CPU | `no_spend` |
| 3 | rigid replacement authoring (Astra CAD via build123d/OCP, Blender program, appearance review) | CPU + OpenAI spend | `gpu_canary` |
| 4 | replacement static qualification (SimReady static USD checks, pxr) | CPU | `no_spend` |
| 5 | replacement native import qualification (Isaac Sim import) | GPU | `gpu_canary` |
| 6 | scene assembly (native task scene assembly in Isaac) | GPU | `no_spend` |

`gpu_canary` means "may spend on a paid provider", not "needs a GPU". Stage 3
rents an RTX 4090 only because the six-stage chain runs as one unit inside
the Isaac container. The first website-origin run spent about nine minutes
pulling the Isaac image before the first OpenAI call, and failed at stage 3
on a rights-record field; the GPU minutes bought nothing.

## The split

1. **Prestage on the control plane (CPU).** Execute stages 1-4 with
   `execute_scene_configuration_stage_chain(..., stage_limit="stage-4")` into
   `<run>/runtime_output/stages` with a bounded CPU deadline. Checkpoints keep
   their exact output paths, hydrated envelope, configuration bytes and deadline.
   A split-phase binding records the prefix boundary; ordinary same-process
   resumes retain the original deadline contract.
2. **Paid run adopts the prefix.** The bundle carries `runtime_output/stages`
   from the prestage and restores the complete input/output capsule at the same
   per-run logical paths. Changing only the basename cannot relocate real files.
   The first native continuation pins its own deadline, which retries cannot
   extend. Inside the container the runner finds the four
   `completed_stage_checkpoint.json` files, prints
   `BLUEPRINT_SCENE_CONFIGURATION_STAGE_ADOPTED` for each, and executes only
   stages 5 and 6. Nothing about stage results, digests or the admitted
   adapter chain changes; the paid run still owns the one provider mutation.
3. **Same accounting.** OpenAI spend from the prestage flows through the
   existing budgeted invoker and inference reservations; the controls and
   policy runs are unchanged.

## How the prestage works (`task_evaluation_scene_configuration_cpu_prestage.py`)

Set `BLUEPRINT_SCENE_CONFIGURATION_CPU_PRESTAGE_STAGE_LIMIT=stage-4` in the
control-plane environment. Inside `run_scene_configuration_vast`, after the
OpenAI stage gates and runtime secrets are composed and before anything is
staged for a provider:

1. `prestage_stage_limit` admits the limit only when every scheduled adapter
   is CPU work and the bundle's replacement backend is Astra (the only backend
   that seals same-root checkpoints). `carried_completed_stage_count` must be 0.
2. `prepare_stage_prefix_before_gpu` extracts the sealed bundle to the paid
   run's exact logical path, `<work_dir>/task_evaluation_scene_configuration_provider_bundle`
   (`BLUEPRINT_SCENE_CONFIGURATION_CPU_PRESTAGE_WORK_DIR`, default `/workspace`,
   the Vast onstart's `WORK_DIR`), and runs the bundle's own entrypoint
   (`run_task_evaluation_scene_configuration_provider.sh`) with the paid run's
   composed environment plus `BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT`, its
   own bounded deadline (`prestage_ttl_seconds`: the scheduled stages'
   allowances, no Isaac allowance) and the host Python on `PATH`. The
   entrypoint materialises the bundled `astra_asset_authoring` wheelhouse
   runtime under `runtime_output/.venv` exactly as it does on the GPU host.
   Secrets stay outside the work dir; the disk budget role is `cpu_prestage`;
   one prestage runs at a time (`<work_dir>/.cpu-prestage.lock`).
3. The result must be `completed_prefix` for the same run id and source commit
   with a sealed `astra_split_stage_resume_binding.v1`. The runner's prefix
   checkpoint zip (marker, completed `stages/<id>/` trees and
   `stages/astra_same_run_resume_binding.json`) is copied to
   `job/cpu_prestage_capsule.zip` with a digest-bound
   `cpu_prestage_transport.json` (work dir, output root, run id, bundle sha,
   authority digest, prefix deadline). The receipt is sealed at
   `job/cpu_prestage_receipt.json`; the work dir is cleared.
4. The capsule is staged like the ArtiFixer pretraining capsule
   (`key_prefix=blueprint/arm-decision-proof-v1/cpu-prestage`) and the runtime
   environment carries `BLUEPRINT_SCENE_CONFIGURATION_STAGE_PREFIX_CAPSULE_{URL,SHA256,BYTES}`.
   A failed prestage raises before any provider allocation.
5. On the GPU host the runner calls `consume_stage_prefix_capsule` before
   `completed_astra_prefix`: it downloads exactly the bound bytes, verifies the
   digest, marker and transport record, refuses a capsule bound to another
   run id or output root, and restores only `stages/` members into the empty
   output root. The chain then adopts stages 1-4
   (`BLUEPRINT_SCENE_CONFIGURATION_STAGE_ADOPTED`), binds its own native
   continuation deadline, skips Astra tool installation and executes stages
   5-6 only.

Tests: `tests/test_task_evaluation_scene_configuration_cpu_prestage.py`
(including a rehearsal that seals a real prefix through the real chain and
archive writer, restores it at the same path and continues to stage 6).

## Host requirements (control plane)

- `/workspace` must be a real directory (not a symlink; `.resolve()` must be
  identity) writable by the `blueprint` service user, ideally a bind mount
  onto the work volume. A stale root-owned
  `/workspace/task_evaluation_scene_configuration_provider_bundle` from an
  earlier manual run must be removed first.
- `python3` on the service `PATH` must be 3.12 (the wheelhouse manifest pins
  it); the entrypoint builds the provider runtime from the bundle, so the
  host venv does not need the Astra profile itself.
- `bash`, `timeout`, `bwrap` (present).
- Capacity: 4 vCPU / 7 GB. Blender appearance review may prove slow; measure
  the first prestage (`job/cpu_prestage_entrypoint.log`) before deciding on a
  CPU worker class.

## Remaining

- Deploy with Astra: batch this with #2028-#2030, provision `/workspace`,
  set the stage-limit env, then one controller-origin website run. Do not
  deploy over a paid run.
