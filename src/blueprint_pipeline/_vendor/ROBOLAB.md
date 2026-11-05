# RoboLab source provenance

Source: https://github.com/NVlabs/RoboLab/tree/b6cfcf6ccd2113c4cd1216d35b33024a7d5691da
Release metadata: 0.3.2. License: Apache-2.0 (unaltered LICENSE inside archive).

The deterministic ZIP contains all upstream robolab Python sources, unchanged,
plus LICENSE. robolab-source.v1.json records every source hash. The sealed provider
bundle installs these sources offline and verifies the loaded files before use.
No policy weights, scenes, benchmark tasks, vendor credentials or assets are fetched
by this dependency step. The run uses the upstream run_episode and InferenceClient.

Blueprint registers its existing native task environment through a small Python
protocol adapter. Its coroutine exposes policy-query and actual native-step
boundaries so RoboLab drives each step while native code retains calibrated
actuator conversion, camera/request evidence and the independent task scorer.
RoboLab's optional ground-truth export is disabled. Its startup resets acknowledge
the native scientific reset and cannot perturb that validated initial state.

The upstream scene importer targets IsaacLab 2.x and rigid object scenes. Blueprint's
IsaacLab 3.x articulated/capture-backed scene builder remains the compatibility
boundary. We do not substitute a RoboLab benchmark scene or success predicate.

Third-party imports on the run path (`robolab.eval.episode`,
`robolab.eval.base_client` and the modules they import) and where each comes from.
`tests/test_robolab_import_closure.py` recomputes this list from the archive and
fails when a module has no provider:

- Shipped with RoboLab as unchanged PyPI wheels. Each wheel is SHA-256 checked
  when the bundle is staged and again on the worker before RoboLab is used
  (`verify_loaded_robolab`), and the preflight records which copy was imported.
  - ImageIO 2.37.2 (BSD-2-Clause, vendored here). RoboLab imports its video
    utility even when recording is disabled.
  - OpenCV, `opencv-python-headless` 4.10.0.84 (Apache-2.0; the wheel also
    bundles FFmpeg and other third-party libraries under their own licenses,
    listed in its `LICENSE-3RD-PARTY.txt`). This exact version is approved in
    `docs/runtime_dependency_license_policy.json`. `robolab.eval.episode`
    imports `cv2` at module level and calls it only to display frames when not
    headless; Blueprint always runs headless. The 50 MB wheel is not kept in git:
    the bundle builder downloads the exact PyPI file once, checks its size and
    SHA-256, and keeps it in the control plane's pinned wheel cache
    (`pinned_wheel_cache.py`). The wheel is the `cp37-abi3` manylinux2014 x86_64
    build against the NumPy 2.0 C API, so it loads on CPython 3.12 with NumPy 2.x
    (verified with NumPy 2.3.1, the version in the G1 runtime packet).
- Named in the native dependency contract and installed from the runtime
  source packet or Isaac Sim: NumPy, torch, h5py, PyYAML, tqdm, and the Isaac Lab
  packages. NumPy is not a wheel in the base packet; it comes from the Isaac Sim
  6.0.1 image (the pinned Isaac Lab requires NumPy 2 or later).
- Isaac Sim itself: `omni`, `pxr`, `isaacsim`.
- SciPy: not contracted or shipped. Blueprint's own wrist-camera code imports it
  in the same Isaac process (`native_task_direct_camera_aim.py`).

The RoboLab import preflight runs after Isaac starts and before policy clients.
