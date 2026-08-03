# Isolated-component license inventory

Components that are deliberately **not** part of the Blueprint runtime
environment (and therefore absent from `uv.lock`, the CI CycloneDX SBOM, and
`docs/runtime_dependency_license_policy.json`, which must exactly mirror the
runtime SBOM). Per the build-on-top adoption mechanics (§5 of
`docs/build_on_top_audit_2026-08-02.md`), every component below must be
recorded here with an exact pin **before its first execution** in any
Blueprint lane. Review owner: @ognjhunt.

| Component | Exact pin | Environment | License | Reviewed | Notes |
| --- | --- | --- | --- | --- | --- |
| skypilot[vast] | `0.13.0` (PyPI, released 2026-07-22) | `.venvs/skypilot` via `scripts/setup_skypilot_venv.sh`; CLI only (`BLUEPRINT_SKYPILOT_BIN`) | Apache-2.0 | 2026-08-02 | Never importable from the runtime; would downgrade Click/Uvicorn/Pillow. Usage telemetry disabled via `SKYPILOT_DISABLE_USAGE_COLLECTION=1`. |
| nerfstudio | `1.1.5` (PyPI) | `.venvs/splatfacto` via `scripts/setup_splatfacto_venv.sh` (bakeoff arm G1) | Apache-2.0 | 2026-08-02 | PyPI 1.1.5 splatfacto hardcodes gsplat `DefaultStrategy`; pins `gsplat==1.4.0`. |
| nerfstudio (git) | `nerfstudio-project/nerfstudio@50e0e3c70c775e89333256213363badbf074f29d` | same venv, arm G2 only | Apache-2.0 | 2026-08-02 | Required for the MCMC arm: `SplatfactoModelConfig.strategy = "mcmc"` exists only past the 1.1.5 release. Exact commit pin; imports `gsplat.strategy.MCMCStrategy`. |
| gsplat | `1.4.0` (PyPI wheel/build) | `.venvs/splatfacto` | Apache-2.0 | 2026-08-02 | The version nerfstudio pins exactly. Distinct from the GPU worker image's vendored gsplat `937e2991…` (v1.5.3) and from NVIDIA's `usd-convert-gsplat` oracle — do not conflate. |
| c2patool | pin at first production use (record the release tag + binary sha256 here) | external binary via `BLUEPRINT_PTDP_C2PATOOL_BIN` | Apache-2.0 OR MIT (contentauth) | 2026-08-02 (pre-registration) | Sidecar-only stamping; hermetic tests never require it. Not yet executed in any production lane. |
| google-cloud-tasks | `2.23.0` | `functions/requirements.txt` (Cloud Functions env — pre-existing) and any future control-plane deploy env for `cloud_tasks_dispatch.py` | Apache-2.0 | 2026-08-02 | Lazy-imported; absent from the runtime base. Fixture lane only until promotion. |

Rules:

1. Exact `name==version` (or commit) pins only; floating versions are not
   reviewable.
2. A component moving into the runtime environment leaves this file and
   enters `docs/runtime_dependency_license_policy.json` through the normal
   fail-closed flow (lock regeneration + exact-version policy entry).
3. Fail-closed at runtime: the code paths consuming these components
   (`skypilot_provisioner`, `c2pa_stamping`, `cloud_tasks_dispatch`, the
   splatfacto arm runner) each degrade to explicit
   `unavailable`/`blocked` statuses when the pinned component is absent —
   never to a silent fallback.
