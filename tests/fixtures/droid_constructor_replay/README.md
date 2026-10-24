# Retained DROID constructor replay

Development-only regression input from Vast instance 57520002, sealed source
7db0e7d6139642f47916c9f05af8f8c5f3b37de5. `retained_input.json` is the exact
cell-zero robot input extracted from its immutable bundle. It contains seven
named arm resets; the GA setter requires thirteen articulation-ordered values.

The upstream files are unmodified source at the immutable commits and paths in
`upstream.json`, with SHA-256 records. Arena is Apache-2.0 (`LICENSE.md`);
Isaac Lab is BSD-3-Clause (`LICENSE_ISAACLAB_BSD`). GA's
`franka_stack_events.py` reexports `set_default_joint_pose` from `stack_events.py`.

`test_droid_retained_constructor_replay.py` executes the released constructor
classes and reset function bodies with CPU configuration/asset and kernel
stand-ins. It reproduces the 13-versus-7 assertion, then checks our real factory
preserves the seven-arm reset event and leaves gripper defaults untouched in
both releases. The existing thirteen-value setter path is also preserved.
These tests establish compatibility only, not native stepping, observation,
policy execution, contact correctness or runtime qualification.
