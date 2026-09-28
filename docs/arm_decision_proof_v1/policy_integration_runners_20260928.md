# Policy integration runner proof boundary

ADP-011 / day 7 admission and ADP-050 / day 28 execution/delivery.
Owner-authorized extension: model upload, customer-hosted controlled observation
access, simulator controller adapters, and high-level skill traces.

The complete acceptance checklist is maintained in the sibling WebApp repo at
`docs/robot-team-policy-integration-acceptance-20260928.md`. These modules are
prerequisites for that goal, not a terminal production receipt.

## Initial model profile

`onnx_state_mlp_cpu_v1` uses ONNX 1.23.0 and ONNX Runtime 1.30.0. Install the
`policy_model_cpu` extra in the isolated policy worker only. The WebApp stores
private bytes and metadata; it never deserializes an uploaded inference graph.

Compatibility is deliberately explicit: at most 16 MiB, embedded weights,
ONNX IR at most 10, standard opset 7–17, static shapes, an approved bounded
MLP operator set, one float32 `[1, state_width]` input and one float32
`[chunk_rows, action_width]` output. The interface binds named state fields in
order and action channels with units and accepted limits. Model bytes must
match the server-computed SHA-256 and size before inference. No custom operator,
external initializer file, graph/function extension, or dynamic shape is
accepted. Process isolation and resource ceilings remain separately required.

The Blueprint-owned linear model in `tests/test_policy_model_onnx.py` actually
executes inference and produces different actions for different states. This
is CPU inference evidence, not scene execution, task success, or a customer
model result. Native LeRobot/safetensors, OpenPI and GR00T require their own
approved framework/configuration/embodiment profiles; ONNX is an optional
portable path, not a universal robotics checkpoint format.

## Controlled observation access

`controlled_policy_observations.py` constructs a new wire object from trusted
simulator inputs. It exports only an opaque request id, task instruction,
approved camera pixels and approved finite robot-state fields. Images are
re-encoded losslessly from pixels to discard metadata. No local path, scene
manifest, calibration URI, asset URL, mesh, texture or scoring object is copied.

The trusted evidence writer must retain the exact returned PNG bytes and
observation digest, including a frame manifest and derived review video. The
HTTPS call allows only operator-approved origins, refuses redirects, bounds
request/response size and time, and accepts the contract's fixed numeric action
tensor. The response cannot contain a policy-authored success label. The
simulator must separately execute actions and score the resulting state.

This keeps scene files and scoring infrastructure inside Blueprint. Visible
frames still disclose information; the boundary is controlled observation
access, not a promise of zero information disclosure or impossible reconstruction.

## Skill traces

A submitted ordered skill sequence now normalizes to `submitted_unexecuted`
with unknown success and no invented motor commands. This fixes the previous
reference replay's unconditional `success=true`. Actual task-bound execution
evidence remains a separate input. Controller-plugin package upload alone
likewise is not controller or simulator execution proof.

## Remaining release gates

Production model upload/auth/store readback; approved runner materialization
and dedicated worker dispatch; real observations and simulator steps; native
controller adapter execution; separate HTTPS proof; independent task outcome;
durable owner-scoped result/download; exact release/deployment identities; and
terminal worker/image/artifact cleanup are all still required by the active goal.
