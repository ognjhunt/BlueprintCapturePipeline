# ARKitScenes Scripted Inspection Run

`blueprint_pipeline.arkitscenes_inspection_run` prepares the zero-spend local
portion of a development-only ARKitScenes inspection Task Evaluation Run. It
verifies the retained public-dataset source bytes, compiles an admitted
provider-derived source profile, regenerates a visible-object target under
`rendered_scene_task_target_orchestration.v1`, builds depth-derived geometry
candidates, proposes the official Isaac Franka placement, and freezes the
existing five scripted inspection controllers.

The source is always labeled **public-dataset proxy**. It is not Blueprint Raw
Contract V3.2 or iPhone-route evidence. Depth-derived geometry remains support
only; dataset-declared meters, coordinate conventions, collision, placement,
and reset stability retain separate qualification states.

Run the retained scene with:

```bash
python -m blueprint_pipeline.arkitscenes_inspection_run \
  --scene-root /path/to/arkitscenes/40958756 \
  --selected-compilation /path/to/arkitscenes/40958756/compiled/<id>/arkitscenes_raw_proxy_compilation.json \
  --output-root output/arkitscenes-40958756-scripted-inspection \
  --implementation-source-commit-sha "$(git rev-parse HEAD)" \
  --view-id arkitscenes-40958756-0001638877 \
  --allowed-object-label sink \
  --torchvision-weights /path/to/fasterrcnn_resnet50_fpn_v2_coco-dd69338a.pth \
  --maximum-spend-usd 1.00 \
  --hard-ttl-seconds 3600 \
  --paid-runtime-authority user_explicit_thread_authorization_YYYY-MM-DD
```

The authority flag records a human grant but never bypasses placement or
collision gates. On a host without a local NVIDIA Isaac runtime, the command retains the exact
scene, target, placement, camera, outcome, controller, spend, TTL, and output
requirements, then emits a typed digest-bound terminal abstention. It does not
launch paid compute. A later authorized runtime must execute exactly five
controllers from one matched reset, retain every trace and render, and
normalize the Isaac result before a controller ranking or Decision Envelope can
be emitted.

Even after execution, the result is a single-scenario, simulation-only scripted
controller comparison. It is not learned-policy evidence and proves no physical
success, deployment, safety, or transfer.
