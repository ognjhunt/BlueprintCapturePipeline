# Reconstruction worker image

This is the pinned, headless Linux/amd64 CUDA candidate for COLMAP pose work and
gsplat/3DGRUT appearance training. It preserves hidden-held-out isolation: the
image healthcheck does not accept a dataset path and scientific evaluation runs
in the independent evaluator.

The runtime trainer is invoked explicitly with `python -m
blueprint_pipeline.reconstruction_gaussian_trainer`. It consumes a typed,
digest-bound request plus a candidate-only COLMAP export. The launcher disables
3DGRUT test rendering and extra metrics; hidden held-out views remain outside
the worker input and are scored only by the independent evaluator.

The Dockerfile is a build recipe, not a build receipt. Until a native Linux GPU
build resolves the image digest and passes the runtime healthcheck, its status is
`candidate_unbuilt`. Paid builds and GPU canaries must enter through
`python -m blueprint_pipeline.paid_resource_allocator`.

Ubuntu 22.04 does not supply the manifest-pinned Python 3.11.9 runtime from its
default package repositories. The image builds the official CPython 3.11.9
source tarball only after verifying its pinned SHA-256 digest; do not substitute
an unpinned PPA or a distro `python3.11` package.

The FFmpeg 6.1.1 source archive is likewise bound to an exact SHA-256 digest
before extraction. A versioned URL alone is not accepted as source integrity.
The Python requirements lock is also digest-checked inside the image build and
bound into the worker-stack manifest; changing it requires updating both pins.
Regenerate the fully resolved Linux/amd64 Python 3.11 CUDA 12.4 hash lock with
`python scripts/compile_reconstruction_worker_lock.py`. The generator requires
the repository-pinned uv version, an immutable cutoff, binary distributions,
and hashes for direct and transitive packages. Only the pure-Python `asciitree`
and `antlr4-python3-runtime` source distributions are allowed; their builds use
the separately hash-locked pip/setuptools/wheel bootstrap with build isolation
disabled. `--check` validates both locks' recorded input, target, cutoff, exact
versions, and hashes without package-index access.

The baseline 3DGRUT loss imports the fused-SSIM CUDA extension unconditionally.
The image therefore builds that extension from its upstream pinned commit and
checks the source revision and import during the embedded healthcheck. NCore,
SlangTorch, and Hydra imports are checked for the same trainer-startup reason.

Prepare the deterministic, exact-source build archive with
`python -m blueprint_pipeline.reconstruction_worker_build_packet`; the archive
does not build or push anything. Its manifest is accepted only by the canonical
`cpu-build` allocator, which applies the independent paid-resource controls.
Preparation requires explicit worker-stack and license-review receipt files;
the latter must limit the image to a private internal build and cannot be
inferred from operator or agent prose.

Generate the current repository-pinned candidate manifest before compiling the
license inventory:

```bash
python -m blueprint_pipeline.reconstruction_worker_contracts \
  --source-commit <clean-immutable-sha> \
  --output <worker-stack-manifest.json>
```

The command performs no build, provider call, or authority grant. It records
`candidate_unbuilt` and binds the exact source SHA, pinned components, model
assets, GPU capabilities, hidden-held-out isolation, and self-grading boundary.
