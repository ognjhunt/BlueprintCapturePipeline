# Reconstruction worker image

This is the pinned, headless Linux/amd64 CUDA candidate for COLMAP pose work and
gsplat/3DGRUT appearance training. It preserves hidden-held-out isolation: the
image healthcheck does not accept a dataset path and scientific evaluation runs
in the independent evaluator.

The Dockerfile is a build recipe, not a build receipt. Until a native Linux GPU
build resolves the image digest and passes the runtime healthcheck, its status is
`candidate_unbuilt`. Paid builds and GPU canaries must enter through
`python -m blueprint_pipeline.paid_resource_allocator`.
