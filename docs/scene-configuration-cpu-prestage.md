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

## Done

- `stage_limit` in the provider runtime and the bundle runner
  (`BLUEPRINT_SCENE_CONFIGURATION_STAGE_LIMIT`). Prefix time admission counts
  only the scheduled stages.
- Exact-path split-phase checkpoint binding. A real-file fixture exercises
  prefix restoration at its original logical path, adoption of stages 1–4, and
  rejection of path changes or deadline extensions. This is local contract
  evidence; the cross-host capsule and live CPU authoring remain unproven.
- Website rights admission carries the driver gate fields (the stage-3 failure).

## Remaining

- Control-plane prestage entrypoint: materialise the packaged Blender runtime
  and the `astra_asset_authoring` python profile on the host (build123d, OCP
  and pxr are already in the venv), run the runner with the limit under the
  staged bundle's `runtime_output`, and seal the prefix receipt.
- Bundle builder: include input bytes and `runtime_output/stages` from the
  prestage in a digest-bound capsule, restore the exact logical paths on the
  native host, and verify actual artifact and hydrated-input bindings before
  adoption. Do not rewrite signed receipts or weaken their path checks.
- Launch ordering: activation submits the paid run only after the prestage
  receipt is sealed; a failed prestage never rents a GPU.
- Capacity: the control plane is 4 vCPU / 7 GB; Blender appearance review may
  need a CPU worker class or a cheap CPU-only provider instance if it proves
  slow. Measure the first prestage before deciding.
