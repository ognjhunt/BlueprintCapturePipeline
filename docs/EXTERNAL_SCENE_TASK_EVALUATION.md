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
