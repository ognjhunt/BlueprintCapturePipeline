# Native Task Execution Admission

Native construction consumes one exact execution-admission receipt, not a
generic statement that a USD opened in Isaac.

The production ordering is:

1. An authoring agent emits registered USD bytes and explicit collision intent:
   moving rigid body, contact role, source-derived dimensions, cooking
   approximation, and geometric envelope.
2. `prepare_native_task_execution_candidate` reopens those exact bytes and runs
   the deterministic GPU-collision audit before provider authority exists.
3. The immutable native-import bundle carries the candidate and exact runtime
   image digest.
4. The exact-runtime worker attaches the complete co-present set to a native
   physics scene, requests GPU broadphase and GPU dynamics, initializes PhysX,
   steps eight zero-gravity frames, and reads every declared collider and
   approximation back.
5. Paired construction bindings v2 retain the candidate, native result, and
   per-object readback.
6. Final packet materialization compiles the complete scene plan and seals
   `native_task_execution_admission.v1` over the final packet digest, scene-plan
   digest, runtime result, registered USD, runtime image, and collision intent.
7. Live-profile construction and the canonical allocator refuse a v2 packet
   without that matching admission receipt.

Any change to the registered asset, cooking approximation, collision intent,
native runtime result, final packet, or scene plan invalidates admission.

## Claim boundary

Admission proves only that the exact task bytes passed provider-zero geometry
qualification, native GPU collision cooking/readback, a bounded physics step,
and final packet compilation. It does not prove contact quality, joint behavior
under the task load, robot reachability, controls success, policy success,
physical equivalence, or a physical outcome.

## Compatibility

Historical `paired_target_native_construction_bindings.v1` packets remain
readable. New materialization emits v2 whenever the native result contains the
candidate digest and qualified GPU physics readback. That v2 marker makes the
new admission mandatory at profile build and allocator execution, so migration
does not reinterpret old evidence while new scenes fail closed.
