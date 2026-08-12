# Production Stage-Worker Interface Inventory

Scope: inventory only. This records what exists today for three production
capabilities — visual splat inspection, object removal, and rights-routed CAD
generation — and what is missing to make each a state-machine-owned adapter with
pinned inputs, pinned outputs, and a retained receipt.

It does not implement, design, or schedule any of them. Nothing here authorizes
a paid run.

Audited at `main` = `ea296fd7c`. All claims below are file-grounded.

## 1. The seam already exists

There is a real dispatcher-owned adapter seam. It is not an agent skill surface
and not a supervisor shell command.

- Module: [`task_evaluation_prelaunch_skills.py`](../../src/blueprint_pipeline/task_evaluation_prelaunch_skills.py)
- Schemas: `task_evaluation_prelaunch_skill_plan.v1`,
  `task_evaluation_prelaunch_skill_execution.v1`
- Validated during profile validation:
  [`task_evaluation_launch_dispatcher.py:271`](../../src/blueprint_pipeline/task_evaluation_launch_dispatcher.py:271)
- Executed by the dispatcher, and a non-`passed` status blocks the launch:
  [`task_evaluation_launch_dispatcher.py:864`](../../src/blueprint_pipeline/task_evaluation_launch_dispatcher.py:864)

Enforced properties of the seam (from the module, not from prose):

| Property | Where |
| --- | --- |
| Inputs resolve only to profile-declared `immutable_inputs` names | `_profile_input_path` |
| Fixed argv, `shell=False`, no agent-chosen arguments | `_execute_room_survey:334`, `_execute_cad_inspection` |
| Per-step timeout, bounded 1–900 s | `_MIN_TIMEOUT_SECONDS`, `_MAX_TIMEOUT_SECONDS` |
| At most 8 steps per plan | `_MAX_STEPS` |
| Typed errors only, `^[a-z0-9_]+(?::[a-z0-9_.-]+)?$` | `_SAFE_ERROR`, `_typed_error` |
| Output digest verified against the tool's own self-digest | `_execute_room_survey:369` |
| Immutable receipt, replay-bound, written once | `_write_immutable`, `execute_prelaunch_skill_plan:458` |
| No secret read, no provider mutation, no allocator call | module docstring + absence of any provider import |

**The template for any new adapter is therefore already set**: add the adapter id
to `_SUPPORTED_ADAPTERS`, add a field-exact `_validate_step` branch, add an
`_execute_*` that shells a pinned module through fixed argv, and verify a
digest-bound output. The CAD adapter additionally shows how to pin an external
skill tree by `expected_commit` / `expected_tree` rather than trusting it
(`_execute_cad_inspection:391`), which is the pattern the remaining three need.

## 2. Capability status

### 2.1 Visual splat inspection — PARTIAL (non-visual only)

Present in the seam: `interiorgs_room_survey`, which runs
[`public_scene_viewpoint_survey.py`](../../src/blueprint_pipeline/public_scene_viewpoint_survey.py).

That module states its own limit: it is selection evidence that "does not
render, edit, or admit a scene." It produces room-centre overview *camera poses*
and a room-scoped object inventory. It is geometric survey, not visual
inspection.

Rendered-frame review surfaces exist but none is reachable from the state
machine:

- `public_scene_simready_visual_review.py`
- `public_scene_aura_human_review.py`
- `public_scene_aura_native_render.py`
- `episode_visual_evidence.py`

**Missing:** an adapter that renders splat frames from pinned viewpoints and
retains them as digest-bound review evidence.

**Structural obstacle, not just absent code:** splat rendering wants a GPU, and
the seam forbids provider mutation and caps a step at 900 s. So this capability
does not fit the prelaunch seam as written. It needs either a CPU-only render
path admitted into this seam, or a separate allocator-owned paid stage with its
own spend/TTL/teardown/provider-zero gates. That choice is a real design
decision and is deliberately left open here.

### 2.2 Object removal — ABSENT from the seam

`_SUPPORTED_ADAPTERS` contains no object-removal adapter.

What exists instead, all outside the state machine:

| Surface | Nature |
| --- | --- |
| `public_scene_inpaint360_adapter.py` | prepares commands; retained receipt is `status: prepared_unexecuted` |
| `public_scene_aura_adapter.py`, `public_scene_infusion_adapter.py` | candidate adapters |
| `adp_inpaint360_interiorgs_vast.py`, `adp_aura_interiorgs_vast.py` | paid Vast execution lanes |

The paid lanes are provider-bound by construction — they require teardown and
provider-zero before a receipt is valid
(`public_scene_inpaint360_execution.py:120-129`,
`public_scene_aura_execution.py:143`). They therefore **cannot** sit under the
prelaunch seam, which forbids provider mutation.

**Missing:** either (a) a paid object-removal stage owned by the state machine
with its own allocator boundary, or (b) a CPU-only *suppression-volume* adapter.
Option (b) is the one that fits the existing seam: suppression volumes are
applied at render/package time as a digest-bound volume receipt rather than by
deleting splats, so they are local, deterministic, and reversible.

### 2.3 Rights-routed CAD generation — INSPECTION ONLY, NO RIGHTS ROUTING

Present in the seam: `earthtojake_step_inspection`, which calls
`capture_cad_inspection` with a `cad_skill_root` pinned by `expected_commit` and
`expected_tree`.

The module docstring states the limit precisely: it "inspects an already
admitted STEP artifact; it does not generate geometry or promote a CAD candidate
to measured truth."

Two distinct things are missing, and they should not be conflated:

1. **Generation.** No adapter generates CAD. The only generation-adjacent module
   is `public_scene_simready_control.py`.
2. **Rights routing.** There is no rights-routing layer for CAD at all. The
   nearest modules serve different purposes:
   - `consent_normalization.py`, `consent_takedown.py` — capture consent, not
     asset rights class
   - `reconstruction_worker_license_inventory.py` — SBOM/license mirror
   - the only `rights_admitted` flag in the tree is a single ScanNet++ access
     field (`public_scene_scannetpp_access.py:183`)

**Missing:** a rights-class routing decision — which generator a request is
allowed to reach, given the source asset's rights — plus a receipt binding the
generated geometry to that decision. Today nothing records why a given generator
was permitted for a given input.

## 3. Summary

| Capability | Seam adapter | Gap |
| --- | --- | --- |
| Visual splat inspection | `interiorgs_room_survey` (non-visual) | no rendered-frame adapter; needs a GPU-vs-CPU seam decision |
| Object removal | none | paid lanes structurally excluded; suppression-volume adapter is the seam-compatible option |
| Rights-routed CAD generation | `earthtojake_step_inspection` (inspection only) | no generation adapter **and** no rights-routing layer |

Common to all three: the adapter contract, the pinning pattern, and the receipt
discipline already exist and are proven by two shipped adapters. What is missing
is per-capability, not architectural — except for visual splat inspection and
paid object removal, where the seam's no-provider-mutation rule forces an
explicit choice between admitting a CPU-only path and standing up a second,
allocator-owned stage boundary.
