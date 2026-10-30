# Cosmos DROID runtime and Nano integration

The beta harness can prepare both exact public NVIDIA DROID policies against the existing Panda/Robotiq embodiment. Nano is a separate candidate with its own checkpoint, plain-text prompt and resource floor. Edge retains its model revision, source revision, JSON prompt, native 32-row output and eight-row simulator execution. The standalone server bootstrap now uses exact offline-installable artifacts for both candidates.

## Immutable inputs

- Source: [`NVIDIA/cosmos-framework@2f603cb114ff8b335e116060444d0b6caee3a85e`](https://github.com/NVIDIA/cosmos-framework/tree/2f603cb114ff8b335e116060444d0b6caee3a85e). Retained ZIP: 4,756,151 bytes, SHA256 `d0825b630396f62b58f4c8875066cc52ceadaf6432c9491602e83ef808a9eddc`, 899 exact files. The source tree retains five <=1 MiB parts; bounded reassembly checks the same full ZIP SHA256 before staging. The privileged source verifier limits remain unchanged. The ZIP uses deterministic packaging of the immutable upstream archive; upstream source is unmodified.
- Nano checkpoint: [`nvidia/Cosmos3-Nano-Policy-DROID@805c0d6d46196ecdc789213a6de72303c383895b`](https://huggingface.co/nvidia/Cosmos3-Nano-Policy-DROID/tree/805c0d6d46196ecdc789213a6de72303c383895b). Retained inference/rights inventory: 39 files, 32,936,198,308 bytes. Thirty small files were downloaded and hashed; nine weight-file hashes come from publisher LFS metadata. The full 43-file repository includes optional promotional images/video that are not staged.
- Nano inventory canonical SHA256: `1f87ceffb8f1db9e0a3b6cf76e23d64e2d01047c8e39ffe92691f386aa2a3e88`. Edge's existing 21-file inventory and 9,154,401,203-byte model identity remain unchanged.
- Server interpreter: CPython 3.13.9 standalone build 20251014; 33,517,986 bytes, SHA256 `c0ccda275948c79996e46993c2c5476ff5cca606dee530f1dea712179131b348`. Regular/archive member expansion is 98,092,081 bytes, bounded at 128 MiB.
- Production dependencies: 109 exact server wheels, plus five thin-client wheel records (four selected for the actual native Python minor), 5,652,022,212 deduplicated external bytes including interpreter. Canonical manifest digest: `sha256:3eb229d970c775596eefe3f9cec35fb9e9232649e7a87b616a55ee9c03dc5460`; raw manifest SHA256: `b999ed98b7c73da5c06f1c7f47ec5bea50148dbf1d422435686f2a1f59fa3ba6`.

The CUDA server uses publisher-locked Torch 2.10.0+cu128, TorchVision 0.25.0+cu128, Flash Attention 2.7.4.post1+cu128.torch210, FlashAttention3 1.0.3+cu128.torch210, NATTEN 0.21.6.dev6+cu128.torch210, Transformers 4.57.6 and NumPy 2.2.6. The locked Flash Attention/NATTEN wheels require CPython 3.13. No runtime resolver, training-stack sync, source compilation or post-staging hub download is used. The prebuilt ANTLR wheel reuses the existing retained source/license provenance.

## Why isolate the interpreter

The exact current native image is `nvcr.io/nvidia/isaac-sim:6.0.1@sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb`. Its public OCI config (`sha256:2d4ebfef053f740b634e30723e38ed736255f677553425ef35552437d74c3b21`) identifies Ubuntu 24.04. Our fixed launch envelope installs distro `python3`; the previous Cosmos bootstrap selected `/usr/bin/python3`. This is evidence of a 3.12-versus-locked-cp313 compatibility problem, not a measured execution of this image. No provider was allocated for this audit.

The server now ignores host Python and clears Isaac import/shared-library paths. The thin client uses the actual running native interpreter, selecting the exact cp311 or cp312 msgpack wheel and pure-Python OpenPI/WebSocket/typing wheels. It verifies installed transport bytes against retained wheels and replaces a previously imported different transport package. NumPy and the simulator remain in Isaac's environment.

## Actual CPU evidence and limits

[`cosmos-droid-cpu-proof-2026-10-30.json`](cosmos-droid-cpu-proof-2026-10-30.json) retains the complete CPU wheel inventory and result. The reproducible script is `scripts/audit_cosmos_droid_inference_imports.py`.

Using the exact CPython 3.13.9 distribution and matching Torch/TorchVision CPU builds, 85 bounded public wheels (667,707,106 bytes) supported:

- Import of the actual pinned RoboLab server and its inference modules.
- Local checkpoint config loading, source-supported local tokenizer dispatch and actual Qwen3VL tokenizer creation.
- Three 224×224 transport views, publisher composition to 540×640, and the publisher's final `resolution=480` transform to a `[3,33,544,736]` video tensor.
- Exact `[33,64]` conditioned action input, preserving seven observed joints and applying the publisher's `1 - gripper` conditioning convention. Returned actions are inverted back by the same official server; the client adds no second normalization.
- Plain-text Nano prompt and an actual OpenPI msgpack/NumPy request roundtrip, with zero attempted network connections.

The publisher explicitly overrides OpenPI client's declared `numpy<2` requirement in `pyproject.toml`. Its selected NumPy 2.2.6 works in the measured codec roundtrip, but `pip check` reports that declared requirement conflict. It is recorded rather than hidden or resolved by changing the publisher's numeric stack.

No policy model was constructed, no weights were loaded, no CUDA kernels or inference were executed, and no simulator result is claimed. Nano's 80 GB and Edge's unchanged 48 GB allocation floors are conservative co-resident policy/Isaac budgets, not measured model minima. Native failures remain individual candidate reasons.

## Rights, budget and admission

The exact Nano model card states “This model is ready for commercial and non-commercial use” and identifies OpenMDW 1.1. Its retained README SHA256 is `ad6a745e4f6424603f31958b74bdbb10a4b9a95769b2b1a9b9b9ffb203e10177`. The exact source LICENSE is OpenMDW 1.1, SHA256 `6bd3fdb9356edb6e4c1f00ad9cd6639a1cf06ca0a415a5071fd70d68799209e6`. Neither candidate needs an API credential; ambient Hugging Face credentials are not forwarded.

Each candidate's sealed runtime binds the snapshot inventory, source ZIP/manifest, dependency manifest, interpreter, prompt mode and GPU floor. Native preparation validates this contract. Every external model/interpreter/wheel byte is reserved before fetching, with full size/hash checks and the existing absolute provider deadline. Source ZIP and manifests are already charged as sealed bundle bytes. Transfer estimates also retain the conservative 64 GiB cold-image allowance and 8,000,000,000-byte output bound, plus any external simulator source layer. These ceilings are accounted through the existing canonical allocator and the $15 site screening budget.

The composed compiler/planner/bundle regression prepares Pi, GR00T, Edge, Nano and MolmoAct2 DROID at $3 maximum per candidate ($15 total fixture reservation), with the actual third camera and independent manifests. At a smaller site budget, the ordinary planner retains a deferred reason for each candidate that does not fit. A historical two-camera scene keeps the other candidates and reports the missing actual third view for both Cosmos policies.

This followup depends on the shared `local_policy_provisioning.py` helper from the Molmo runtime integration, including standalone interpreter isolation fix `2a9b4dbaa`. It does not modify the PiPER or Molmo dependency locks and does not widen historical ADP pair authorization.
