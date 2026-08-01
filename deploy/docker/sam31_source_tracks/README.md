# SAM 3.1 source-track canary image

This image is a bounded GPU worker for source-frame semantic tracks. It is
separate from the customer-facing SAM privacy service and does not expose an
HTTP server.

The image pins the official SAM code revision and the CUDA/PyTorch compatibility
floor used by that revision. The gated `sam3.1_multiplex.pt` checkpoint is not
baked into the image. The paid-resource adapter must download it with an
ephemeral Hugging Face token, verify the exact configured SHA-256 digest, unset
the token, enable offline mode, and only then invoke
`blueprint_pipeline.sam31_source_track_canary_worker`.

The worker output is 2D source-frame track evidence. It does not establish raw
capture authority, metric object placement, collision geometry, physics, task
success, deployment readiness, or comparative policy ranking.
