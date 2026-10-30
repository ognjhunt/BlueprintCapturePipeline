# Public robot batch03 importer bootstrap repair

The retained batch03 output reports **0/19 imported**, including both URDF
configurations and the KUKA MJCF configuration. Each body log reaches the
official Isaac Lab converter and fails importing `isaacsim.asset.importer.*`:
`ModuleNotFoundError: No module named 'isaacsim.asset'`. Earlier in every log,
Kit cannot resolve NuRec's excluded `omni.isaac.ml_archive` dependency. These
are bootstrap failures; they establish no body-specific conversion result,
controller motion, task result, or policy result. Raw production logs remain
outside this repository.

The preparation worker reused the site-scene launcher, which unconditionally
requested NuRec, SPG and Pink even with cameras disabled and no site appearance.
Removing those requests alone is insufficient: the pinned official URDF and
MJCF importer extensions themselves require two other excluded dependencies.

## Exact source evidence

- Isaac Lab commit
  [`ffff603eafc6b74264a5261cc0183d6a65390d78`](https://github.com/isaac-sim/IsaacLab/blob/ffff603eafc6b74264a5261cc0183d6a65390d78/apps/isaaclab.python.headless.kit)
  excludes `omni.kit.pip_archive`, `omni.isaac.ml_archive`,
  `isaacsim.pip.newton`, and `omni.warp.core`. Its exact headless experience is
  retained as a development fixture with its BSD-3-Clause license; SHA-256
  `e5d1f4e581f177526a1b90414d1139242c89234dc4eaef8c16fee5c931c788c6`.
- Isaac Sim 6.0.1 commit
  [`045ca8b59622b99a408092124377c66346e8d9c2`](https://github.com/isaac-sim/IsaacSim/tree/045ca8b59622b99a408092124377c66346e8d9c2)
  provides URDF importer 3.11.2 and MJCF importer 3.10.1. Both require
  `isaacsim.pip.newton`; it and asset-transformer rules require
  `omni.kit.pip_archive`. NuRec utilities 0.1.2 separately require
  `omni.isaac.ml_archive`. The selected dependency edges, exact source URLs,
  and source-file hashes are retained in
  `tests/fixtures/isaacsim_601_importer_dependencies.json`.
- The same Sim revision's
  [`deps/pip_newton.toml`](https://github.com/isaac-sim/IsaacSim/blob/045ca8b59622b99a408092124377c66346e8d9c2/deps/pip_newton.toml)
  deliberately does not bundle Warp. Newton 1.2.1 requires Warp >=1.13.0;
  the retained native source packet already supplies Warp 1.13.0 and
  Torch 2.10.0+cu128. No dependency, image, or source revision is upgraded.

## Repair and validation boundary

`deployment_embodiment_isaac_launch` verifies the existing native source receipt,
then uses the exact headless experience. Only the isolated converter process
gets a derived copy permitting Kit's pip archive and Newton and explicitly
enabling its official URDF or MJCF extension. The original source file remains
unchanged. Relocated `${app}` search paths still point to verified Isaac Lab
source. ML archive and bundled Warp remain excluded. The separate controller
process uses the original experience and all four upstream exclusions.

Startup retains the selected experience bytes/hash and loaded extension,
module path, Torch and Warp version readback. Exceptions are flushed before
Isaac's process-terminating fast shutdown. The existing parent verifies the
import checkpoint and retains each body outcome. Official converter settings,
geometry, dynamics modeling, source/licensing checks, deadlines, and task/policy
claim boundaries are unchanged. The site-scene rendering launcher is unchanged.

Local tests replay the two selected upstream dependency conflicts, exercise
URDF/MJCF and controller startup configuration, verify original source retention,
reject conflicting loaded dependencies, and rerun the real subprocess-exit
checkpoint/smoke regressions. These use development SDK doubles. This machine
has no Isaac Sim image or GPU: **actual Kit resolution, native USD conversion,
and controller stepping still require the next bounded native batch**. A
successful dependency replay does not establish that those operations succeed.
