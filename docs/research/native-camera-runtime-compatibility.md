# Native camera and controller startup compatibility

This change separates the task runtime's required capabilities. It does not
change the dedicated public-body importer, its controller smoke, or an existing
asset-preparation grant. No GPU execution was performed for this change.

The retained batch03 logs showed Kit refusing NuRec-utils' dependency on
`omni.isaac.ml_archive`. The full task launcher had the same unconditional
NuRec-utils/Pink enablement. Both pinned IsaacLab experiences exclude the ML
archive and bundled Warp. Enabling the ML archive would also introduce its
Torch 2.11 dependency into the retained Torch 2.10 runtime.

## Exact source and activation

The source receipt still binds IsaacLab
`ffff603eafc6b74264a5261cc0183d6a65390d78`, IsaacSim 6.0.1 image digest
`sha256:b1c542b2ecc549b3d1ebb78c25664aa3bacba1709e6ad8e0a68e09426d57dedb`,
and the separately provisioned Torch `2.10.0+cu128` / Warp `1.13.0` bytes.

- [Official NuRec rendering helper](https://github.com/isaac-sim/IsaacSim/blob/045ca8b59622b99a408092124377c66346e8d9c2/source/extensions/isaacsim.replicator.nurec_utils/isaacsim/replicator/nurec_utils/rendering_setup.py),
  its adjacent `config/nurec_config.yaml`, and the repository Apache-2.0 license
  are retained unchanged in `config/isaacsim_nurec_601`. The sealed manifest
  checks every byte before loading. The helper imports no metrics or Torch
  module. This is reuse of the official Python helper; it does not claim the
  NuRec-utils extension was enabled. Its stage classification and 800-step
  rendering warmup remain unchanged.
- [Official Pink extension](https://github.com/isaac-sim/IsaacSim/blob/045ca8b59622b99a408092124377c66346e8d9c2/source/extensions/isaacsim.robot_motion.pink/config/extension.toml)
  and five required experimental extensions are bound to their exact configs,
  active Python Git blobs, and the bundled Franka URDF. A private runtime
  overlay preserves their original IDs, Python, plugins, assets, and every
  dependency except four explicit `omni.warp.core` edges. Those edges name a
  small provider that requires the retained Warp 1.13 module and path.
  Both exact parent/child experiences allow required Kit pip; ML, Newton, and
  bundled Warp remain excluded. Original SDK/source bytes are not modified.
- Same-ID `--ext-path` precedence is checked after Kit startup against each
  resolved extension path. Python namespaces and the controller class must
  also resolve to the verified image source. Required Pink startup constructs
  the real `PinkIKController` using its bundled Franka robot and frame task;
  this constructor check does not claim a solve or robot movement.

| Caller or capability | Requested startup |
| --- | --- |
| Team named-arm control, native DLS sweep, ordinary composition/destination | `joint`; no SDK Pink |
| G1/SONIC and G1 composition/destination | `g1_joint`; append exact image pure-Python Pink bundles, retain external Pinocchio priority, no SDK Pink |
| Legacy Franka construction, controls, policy, runtime preflight, controlled policy | `pink_franka`; verified compatibility overlay and real controller construction |
| NuRec volume / ParticleField appearance | SPG plus existing renderer settings, exact official helper |
| USD geometry / no site appearance | No requested SPG or NuRec-utils extension |
| Cameras disabled | Pinned headless experience, not the rendering child |

Scoped Arena embodiment imports remain in place: ordinary joint callers do
not import every unrelated embodiment. G1 itself needs pure-Python Pink even
when SONIC is selected; its loaded dependency origins are reported separately
from SDK controller activation.

After every task launch, readback checks the actually loaded Torch/Warp
versions, file origins and hashes, conflicting extension flags, and a small
CUDA tensor round trip with shared Torch/Warp storage. SPG enablement,
renderer settings, and schema state are actual Kit reads, not inferred from
launch arguments. A failure retains its observed identity diagnostic.

## Bounded camera diagnostic

`native_camera_runtime_diagnostic.run_bounded_camera_process` is shipped in
native policy bundles and can be called by a preflight on an already admitted
GPU worker. It takes that worker's verified provisioning receipt, a fresh
output directory, absolute deadline, and explicit controller backend. It
does not allocate a provider or alter the spend contract.

The parent bounds startup, rendering, and shutdown to at most 90 seconds,
caps the deadline at the provider deadline minus its existing 600-second
upload/teardown reserve, and leaves five seconds for child close. It owns a
new process group and terminates that group on timeout: the pinned Isaac
`python.sh` launches Python without `exec`. A previous result cannot be
reused, and nonzero process exit cannot count as an overall pass.

The child renders only an explicitly synthetic 64x64 cube scene for at most
120 updates. It records a finite, nonconstant RGB CUDA tensor and digest,
then seals the result before SDK close. It reports no task execution, policy
execution, or captured-site appearance evidence. It is not proof of NuRec
scene fidelity or a candidate's task score. Existing task-specific render
and scoring evidence remains necessary.

## Validation and remaining native evidence

CPU tests exercise exact upstream metadata/bytes, controller selection,
module/path tampering refusal, actual provider ZIP/extracted helper imports,
whole-archive static closure, and real shell/child timeout cleanup. A built
wheel contains the exact helper, YAML, manifest, and license. Actual historical
runtime staging remains below its unchanged 8 MiB aggregate cap. The new
rendering experience fixture is unchanged IsaacLab source, covered by the
adjacent `isaaclab_headless_ffff603e.LICENSE.txt` BSD-3-Clause license.

SDK doubles exercise orchestration only. Real Kit duplicate-ID resolution,
external-Warp compatibility, Pink construction and GPU camera output still
require the bounded diagnostic on the pinned GPU image. Neither these tests
nor public-body import/controller results establish that native camera proof.
