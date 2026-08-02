# External Scene Task Evaluation

This lane turns an authorized external reconstruction, such as a Scaniverse
GLB/PLY/SPZ export, into a claim-bounded Task Evaluation Run. It is a reusable
pipeline contract, not a 506 Lenox special case.

## Durable flow

1. Render digest-bound scene views and inspect available geometry.
2. Propose visible objects, affordances, and task candidates. Proposals may come
   from a model, detector, or agent, but they cannot authorize themselves.
3. Bind each candidate to a 3D scene region with a digest-bound method and
   spatial uncertainty.
4. Evaluate visibility, scale status, reach support, collision support, and the
   scene-frame registration.
5. `scene_task_target_pipeline` deterministically authorizes a bounded target or
   abstains when visual-to-3D evidence is weak.
6. Place the default Franka Panda (or Unitree G1 only for a humanoid task),
   compile the exact external-scene Isaac request, and run the canonical paid
   allocator when paid execution is separately authorized.
7. Rehash static scene/contact evidence, robot-only visibility evidence, and
   policy traces independently. A policy-only failure does not invalidate a
   qualified scene/contact observation.
8. `external_scene_task_evaluation` routes each claim to the best qualified
   method in the current catalog. Unqualified Isaac, analytic, learned-world-
   model, or physical methods remain candidates only and force abstention.

Missing source video does not block this support-asset lane. It lowers the claim
ceiling: the reconstruction is not raw Blueprint capture truth, metric scale is
not independently established, and physical or deployment claims remain false.

## Replaceable analyzer seam

`rendered_scene_task_target_orchestrator` is the reusable bridge between a
semantic proposal backend and deterministic target authorization. The backend
emits a `rendered_scene_task_proposal_set.v1` containing visible object labels,
affordances, task families, confidence, supporting view IDs, and one 2D binding
box per proposal. The orchestrator then:

1. rehashes every rendered RGB observation and the analysis splat;
2. rejects analyzer self-authorization and replay/digest drift;
3. backprojects each proposed box into the splat with explicit uncertainty;
4. binds collision, frame-registration, metric-scale, and reach status; and
5. calls the deterministic target gate, returning either a bounded target or a
   `no_qualified_3d_task_target` abstention.

The semantic backend may be a local detector, hosted vision model, or agent and
remains replaceable. A backend being unavailable is not permission to invent a
target. Static schemas live at
`docs/schemas/rendered_scene_task_proposal_set.v1.schema.json` and
`docs/schemas/rendered_scene_task_target_orchestration.v1.schema.json`.

### Automatic analyzer invocation

Call `compile_rendered_scene_task_target_with_analyzer` when the pipeline owns
proposal generation. It builds a `rendered_scene_task_analyzer_request.v1` from
the exact RGB and splat digests, invokes a replaceable backend, requires the
backend result to echo the request digest, and only then enters the 2D-to-3D
binding and deterministic qualification flow. Local file paths are passed in a
separate runtime-input object and are not part of the portable request digest.

`CommandRenderedSceneAnalyzer` is the generic executable adapter. It uses JSON
stdin/stdout, never invokes a shell, imposes a timeout and output-size contract,
and converts missing, failed, timed-out, invalid, or oversized backend output
into a deterministic analyzer abstention. The command returns proposals only;
Blueprint constructs analyzer provenance and forbids self-authorization.

An unavailable backend therefore yields no proposal and no selected target,
with `rendered_target_analyzer_*` blocker codes. A result replayed from a
different view/splat request is rejected as
`rendered_target_analyzer_request_digest_mismatch`. The corresponding schemas
are `docs/schemas/rendered_scene_task_analyzer_request.v1.schema.json` and
`docs/schemas/rendered_scene_task_analyzer_run.v1.schema.json`.

The supported command entrypoint accepts a digest-bound
`rendered_scene_task_target_pipeline_request.v1`; executable analyzers must be
explicitly authorized in that request:

```bash
python -m blueprint_pipeline.rendered_scene_task_target_orchestrator \
  --request <rendered-scene-target-pipeline-request.json> \
  --output <rendered-scene-target-orchestration.json>
```

This command is the reusable future-run path. It executes the configured
analyzer, creates the proposal-set provenance itself, binds proposals to 3D,
and emits either one qualified bounded-sim target or a structured abstention.

### Private local TorchVision backend

`torchvision_rendered_scene_analyzer` is the first in-repo semantic backend for
this seam. It runs Faster R-CNN ResNet-50 FPN v2 over every rendered RGB view,
maps taskable COCO objects to the default Franka Panda task catalog (or the G1
catalog when the request explicitly selects a humanoid), and emits the strongest
detection for each visible taskable object. It does not upload scene images,
download a checkpoint during analysis, bind its own 3D target, or authorize a
task.

The official checkpoint is operator-provisioned outside the request and must
match this pinned SHA-256 before inference:

```text
dd69338a24b8d7381807e247652bdc356325bcbaf1cd3e092e00e0a1a58706bf
```

Inspect the portable analyzer identity and contract digest with:

```bash
python -m blueprint_pipeline.torchvision_rendered_scene_analyzer --print-contract
```

Configure the generic command adapter with an absolute local checkpoint path:

```json
{
  "analyzer_id": "blueprint_local_torchvision_coco_detector",
  "implementation_version": "1",
  "analyzer_contract_digest": "<digest printed by --print-contract>",
  "command_execution_authorized": true,
  "candidate_may_self_authorize": false,
  "command": [
    "python",
    "-m",
    "blueprint_pipeline.torchvision_rendered_scene_analyzer",
    "--weights",
    "/absolute/path/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth"
  ]
}
```

Without a task-context filter, the backend proposes taskable visible objects and
the deterministic target gate ranks only successfully bound, qualified
candidates. A request may narrow the proposal vocabulary with
`task_context.allowed_object_labels` (for example, `["sink"]`) without changing
model confidence or bypassing qualification. Missing or altered weights,
replayed RGB, unsupported robots, inference failure, and no taskable detections
all produce structured `torchvision_analyzer_*` abstentions.

The subsequent Franka placement solver first searches the nominal standoff
annulus. When that pose is collision-probe clear but analytically outside the
arm envelope, it may search the smaller profile-defined gap for a clear analytic
reach-rescue candidate. Such a pose is marked
`collision_clear_analytic_reach_rescue_candidate`; a below-nominal standoff is
preserved as `placement_below_nominal_standoff_range`. This improves runtime
placement without upgrading metric reach, contact, safety, or physical claims.

## Required commands after a runtime

Build robot visibility evidence from isolated robot-only RGB/depth artifacts:

```bash
python -m blueprint_pipeline.external_scene_robot_placement_evidence \
  --verification-request <external-request.json> \
  --runtime-result <isaac-runtime-result.json> \
  --runtime-artifact-root <validated-runtime-root> \
  --result-out <visual-placement-evidence.json>
```

Compile and execute the evidence-bounded route:

```bash
python -m blueprint_pipeline.external_scene_task_evaluation \
  --verification-request <external-request.json> \
  --runtime-result <isaac-runtime-result.json> \
  --independent-qualification <independent-qualification.json> \
  --visual-placement-evidence <visual-placement-evidence.json> \
  --target-analysis <scene-task-target-result.json> \
  --robot-placement-result <robot-placement-result.json> \
  --output-root <task-evaluation-output>
```

The result is expected to be partial whenever any requested claim lacks a
digest-bound qualification. “Partial” means the supported claims are real and
bounded; it does not mean the task, policy ranking, or deployment succeeded.
