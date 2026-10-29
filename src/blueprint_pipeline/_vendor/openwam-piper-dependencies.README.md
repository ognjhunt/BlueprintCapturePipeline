# OpenWAM PiPER dependency provenance

`openwam-piper-dependencies.v1.json` is an inference-only subset of
OpenWAM commit `3950b6d9ea848dc980a82a89b4554af2c082544a`'s
`docker/requirements-cu128.txt`. Wheel digests and sizes came from exact PyPI
release metadata or the PyTorch CUDA 12.8 wheel index. CPU validation fetched
and hashed every non-CUDA dependency. Large production CUDA wheels were **not**
downloaded here; their full bytes must be verified during staging.

The server interpreter is the exact Astral python-build-standalone CPython
3.12.12 distribution, release 20251014. Its archive was downloaded, hashed,
extracted, and used for the CPU audit. Python 3.12 matches the publisher's
NumPy/SciPy pins. The Isaac client has an independent CPython 3.11 WebSocket
wheel and imports no server-side Torch dependencies.

Two deliberate substitutions are recorded: headless OpenCV at the same
4.14.0.94 version avoids GUI system-library dependencies; the existing thin
client keeps WebSockets 16.0, while the server uses publisher-pinned 16.1.1.

## ANTLR pure-Python wheel

PyPI publishes ANTLR Python runtime 4.9.3 only as a source archive. The retained
`antlr4_python3_runtime-4.9.3-py3-none-any.whl` is a deterministic package of its
unmodified `src/antlr4/**` files. No source setup script or native compiler was
run. The manifest retains the exact source URL, size and SHA-256. Metadata is
the archive's `PKG-INFO`, installed as `.dist-info/METADATA`.

The upstream archive omits its standalone license file. The wheel additionally
retains the complete BSD 3-clause license from ANTLR release 4.9.3 commit
`e4c1a74c66bd5290364ea2b36c97cd724b247357`; its exact source and hash are recorded.
Individual source copyright notices remain unchanged.

Repacking uses Python's ZIP writer with sorted member names, DEFLATE level 9,
timestamp 1980-01-01 00:00:00, Unix mode 0100644 and create_system=3. WHEEL has
Wheel-Version 1.0, Generator blueprint-pure-python-repack-v1, Root-Is-Purelib
true and Tag py3-none-any. RECORD is sorted CSV with LF endings and each file's
unpadded URL-safe SHA-256 and length; its own row has empty digest and size.
The deployment does not rebuild this wheel. It verifies the retained bytes
and installs the wheel offline with hashes.

The local audit receipt and reproduction command are documented in
`docs/research/openwam-piper-dependencies-2026-10-29.md`. CPU import evidence
does not establish CUDA inference, model VRAM requirements, or task success.
