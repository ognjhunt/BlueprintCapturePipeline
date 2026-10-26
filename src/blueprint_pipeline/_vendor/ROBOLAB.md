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

ImageIO 2.37.2 is shipped as its unchanged PyPI wheel (BSD-2-Clause license
included in the wheel). RoboLab imports its video utility even when recording
is disabled. The wheel and installed bytes are verified offline. NumPy, Pillow,
OpenCV, SciPy, h5py, tqdm and torch remain supplied by the existing native simulator
runtime; RoboLab import preflight runs after Isaac starts and before policy clients.
