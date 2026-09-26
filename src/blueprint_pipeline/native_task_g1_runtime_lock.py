"""Pinned Linux CPython 3.12 wheel closure for native G1 control imports.

The Isaac Sim image supplies Pink but not its Pinocchio backend.  These wheels
are staged as immutable packet inputs rather than fetched on a paid instance.
The complete Pinocchio/Coal/CMeel closure is required: the Python modules live
inside ``cmeel.prefix/lib/python3.12/site-packages`` and their native libraries
live in the sibling ``cmeel.prefix/lib`` tree.
"""

from __future__ import annotations


G1_RUNTIME_DEPENDENCY_WHEELS = (
    # SONIC runs its encoder and decoder on CUDA. The task-neutral packet's
    # CPU-only onnxruntime wheel is replaced for this G1 profile.
    {
        "filename": "onnxruntime_gpu-1.24.4-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl",
        "package": "onnxruntime-gpu",
        "version": "1.24.4",
        "license_spdx": "MIT",
        "pure_python": False,
        "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel-0.61.0-py3-none-any.whl",
        "package": "cmeel",
        "version": "0.61.0",
        "license_spdx": "BSD-2-Clause",
    },
    {
        "filename": "cmeel_assimp-6.0.5-0-py3-none-manylinux_2_28_x86_64.whl",
        "package": "cmeel-assimp",
        "version": "6.0.5",
        "license_spdx": "BSD-3-Clause",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel_boost-1.90.0-0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "package": "cmeel-boost",
        "version": "1.90.0",
        "license_spdx": "BSL-1.0",
        "pure_python": False,
        "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel_console_bridge-1.0.2.3-0-py3-none-manylinux_2_28_x86_64.whl",
        "package": "cmeel-console-bridge",
        "version": "1.0.2.3",
        "license_spdx": "Zlib",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel_octomap-1.10.0-5-py3-none-manylinux_2_28_x86_64.whl",
        "package": "cmeel-octomap",
        "version": "1.10.0",
        "license_spdx": "BSD-3-Clause",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel_qhull-8.0.2.1-1-py3-none-manylinux_2_28_x86_64.whl",
        "package": "cmeel-qhull",
        "version": "8.0.2.1",
        "license_spdx": "Qhull",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel_tinyxml2-11.0.0-0-py3-none-manylinux_2_28_x86_64.whl",
        "package": "cmeel-tinyxml2",
        "version": "11.0.0",
        "license_spdx": "Zlib",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel_urdfdom-6.0.0-0-py3-none-manylinux_2_28_x86_64.whl",
        "package": "cmeel-urdfdom",
        "version": "6.0.0",
        "license_spdx": "BSD-3-Clause",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "cmeel_zlib-1.3.2-0-py3-none-manylinux_2_28_x86_64.whl",
        "package": "cmeel-zlib",
        "version": "1.3.2",
        "license_spdx": "Zlib",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "coal-3.0.3-0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "package": "coal",
        "version": "3.0.3",
        "license_spdx": "BSD-3-Clause",
        "pure_python": False,
        "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
    },
    {
        "filename": "eigenpy-3.13.0-0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "package": "eigenpy",
        "version": "3.13.0",
        "license_spdx": "BSD-2-Clause",
        "pure_python": False,
        "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
    },
    {
        "filename": "libcoal-3.0.3-0-py3-none-manylinux_2_28_x86_64.whl",
        "package": "libcoal",
        "version": "3.0.3",
        "license_spdx": "BSD-3-Clause",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "libpinocchio-4.1.0-0-py3-none-manylinux_2_28_x86_64.whl",
        "package": "libpinocchio",
        "version": "4.1.0",
        "license_spdx": "BSD-3-Clause",
        "pure_python": False,
        "wheel_tag": "py3-none-manylinux_2_28_x86_64",
    },
    {
        "filename": "numpy-2.3.1-cp312-cp312-manylinux_2_28_x86_64.whl",
        "package": "numpy",
        "version": "2.3.1",
        "license_spdx": "BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0",
        "pure_python": False,
        "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
    },
    {
        "filename": "pin-4.1.0-0-cp312-cp312-manylinux_2_28_x86_64.whl",
        "package": "pin",
        "version": "4.1.0",
        "license_spdx": "BSD-3-Clause",
        "pure_python": False,
        "wheel_tag": "cp312-cp312-manylinux_2_28_x86_64",
    },
    {
        "filename": "protobuf-6.33.6-py3-none-any.whl",
        "package": "protobuf",
        "version": "6.33.6",
        "license_spdx": "BSD-3-Clause",
    },
)
