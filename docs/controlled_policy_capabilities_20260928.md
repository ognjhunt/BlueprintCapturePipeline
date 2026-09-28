# Controlled policy integration capabilities

ADP-011/day 7 admission and ADP-050/day 28 execution. Owner: Nijel Hunt.
Observed blocker: private model uploads and customer controllers previously
had no compatible handoff to an isolated, observation-only execution path.
Completion artifacts: the model service/image packaging, controlled executor,
orchestrator hooks, and matching WebApp runtime/admission changes.

The owner requested capability building, commit, merge and deployment, then
explicitly restored live proof runs. New test development and unrelated
debugging remain deferred. Fixes directly needed for the live proof are in
scope. Source integration alone does not establish production execution,
task success or physical outcomes.

## Worker integration

Configure `ControlledPolicyExecutor` on the trusted simulator host, then pass it
as `controlled_policy_executor` to `build_robot_eval_job` or
`run_robot_eval_job_request_inbox`. This is a callable supplied by runtime code;
request JSON cannot select imports, shell commands, private scene paths or a
different security profile.

The prepared `run_robot_eval_worker` and inbox
`run_live_pipeline_control_plane` entry points also accept this trusted callable
and an explicit `allow_policy_execution` gate. They forward both through the
existing orchestration path; downloaded manifests cannot enable them. The
existing `BLUEPRINT_ALLOW_POLICY_EXECUTION` environment gate still applies.
`execute_robot_eval_request_as_evaluation_run` carries the callable through the
canonical Evaluation Run handoff. For compiled runs, register a
`RobotEvalEvaluationRunExecutor(controlled_policy_executor=executor)` in the
existing `EvaluationRunExecutionRegistry` and pass that registry to
`execute_evaluation_run`. Default registries and CLI entry points remain
unconfigured and block controlled execution.

Its `task_contract` hook resolves the frozen, rights-bound company-policy v2
observation/action contract from Blueprint's task configuration. Its
`environment_factory` creates the virtual robot adapter for each exact scenario
run. The adapter supplies fresh approved PNG frames, numeric state and the task
instruction, applies validated action channels in their declared units, reports
observed motor steps, and stops on exit. Independent outcome scoring remains
separate. Controller plugins implement the same interface in an isolated image;
no customer Python is imported into the simulator process.

`ControlledSimulatorAdapter` wraps the existing Isaac episode environment:
configure exact camera/state bindings, state units, measured control frequency,
and the measured native action translator
(including joint order, units, control rate and gripper convention), plus the
trusted task-terminal and controller-stop functions. It encodes fresh RGB PNG
frames, rejects incompatible shapes/types, and calls the native simulator step
for each validated action. These bindings are operator configuration, never
customer-selected asset paths or executable imports.

For customer-hosted policies, configure operator-approved HTTPS origins and an
optional protected credential resolver. Credentials stay in HTTP headers and
are excluded from retained policy-input records. The wire object contains only
camera frames, robot state, instruction, an opaque request ID and a synthetic
flag. Frames expose scene information: this is controlled observation access,
not a promise against inference or reconstruction from visible content.

The `sandbox_factory` hook invokes
`execute_company_policy_sandbox_preobservation` with a trusted
`authorize_scene_access` function and the provided `qualified_session` callback.
Existing admission, dedicated-worker attestation, image/credential binding,
network denial measurements and synthetic conformance run before this callback.
The callback receives only the fixed Unix-proxy transport. The executor also
validates the observation projection before forwarding it and retains its
signed terminal cleanup receipt after the session. No callback means the
original synthetic-only behavior.

`ControlledSandboxFactory` provides that concrete plan-to-executor handoff.
Configure its frozen `ControlledSandboxConfiguration` with the exact release,
worker, proxy image, security profiles, registry allowlist and private receipt
directory. Its admission resolver retrieves the retained receipt for the owned
job and exact post-packaging contract; a customer-supplied receipt is not an
authority source. A separate upstream authorizer must approve execution before
any container mutation. The factory generates a unique attempt, builds the
existing runsc plan, obtains independently signed boot evidence for that exact
attempt, and invokes the measured sandbox executor. It passes job context to the
scene-access authorizer after qualification and retains a separate terminal
receipt for each attempt. Missing authorities or boot evidence fail closed.

## Model handoff

Configure `PrivateModelImageBuilder` on a dedicated trusted builder with the
private bucket, approved registry/repository, storage client, source checkout,
private scratch root and receipt sink. It binds the uploaded object generation,
size, SHA-256 and interface to the frozen task, builds an image from the fixed
Dockerfile, publishes and reads back an OCI digest, and removes its temporary
build context and local tag. It copies no credentials, storage URI, scene assets
or scoring code into the image. Graph loading and inference occur inside the
isolated worker using the explicit `onnx_state_mlp_cpu_v1` profile.

Pass this builder as `model_image_builder` to the controlled executor. Model
container command, CPU mode, port and user are selected by the approved runner
profile. Native LeRobot, OpenPI and GR00T packages require additional explicit
runner profiles; they are not silently coerced into the state-only ONNX profile.

Only a host configured for this path publishes an execution offer with
`policy_execution_profiles=("controlled_observation_v1", "onnx_state_mlp_cpu_v1")`.
Without that signed offer the WebApp preserves the upload and blocks paid
execution. The controlled path cannot fall through to legacy manifest export,
Docker launch commands or reference replay.

## Skill traces

WebApp accepts ordered `blueprint.skill_trace.v1` steps and stores task-bound
intent through the authenticated run-owner skill-trace route. Submitted steps
cannot provide their own outcome, motor-action evidence or spend authorization.
The existing high-level trace normalization preserves unexecuted intent and
excludes it from execution coverage.

## Live proof boundary

The sanitized production and dedicated-worker evidence is retained in WebApp
[`docs/robot-team-policy-live-proof-20260928`](https://github.com/ognjhunt/Blueprint-WebApp/tree/2c3dafd06a86d6fa705b6fab7534d25bfbaa090e/docs/robot-team-policy-live-proof-20260928).
The deployed Pipeline at `16d8551b413fe1a65fe70b3f77ad73ea0b42b287`
queried a separate authenticated HTTPS policy and validated 15 rows of actions
from synthetic approved camera/state inputs. No scene or scoring files were sent.
The production-uploaded ONNX model was packaged into an immutable private image
and actually inferred in a dedicated no-mount runsc VM, with policy-container
cleanup read back. WebApp skill intent and owner checks were exercised live.

A retained historical `completed_unqualified` development result was restored
from its verified archive and downloaded through the actual WebApp ticket path
with the expected SHA-256. It is deliberately unlisted-public and proves neither
private result denial nor execution of this newly uploaded policy.

These are component proofs, not a completed native Task Evaluation Run.
Production model planning selected zero eligible tasks and refused execution
preparation. A configured trusted native environment, authorized frozen task,
qualified sandbox factory and signed execution offer are still needed before
new-policy motor execution, independent outcome and private delivery can be
claimed. Do not publish those profiles solely because source hooks exist.

Live registry lease upload exposed an undefined KMS wrapping-field serialization
error in WebApp. The targeted repair merged in
[WebApp PR #750](https://github.com/ognjhunt/Blueprint-WebApp/pull/750); production
KMS/Firestore write/read/decrypt replay and deployed route replay passed.
The production Render worker delivered the durable outbox; Pipeline retained
its exact admission. The canonical executor then claimed the production
single-use lease, pulled and verified the private image, completed nine network
denial probes and synthetic conformance, and removed containers, image bytes
and credential ciphertext. A repeat credential claim returned 409.

Use the separate Blueprint proxy Dockerfile under
`deploy/docker/company_policy_proxy/`. Its worker bootstrap access is distinct
from the customer image lease and is removed before that lease is claimed.
The dedicated development proof VM and temporary HTTPS service were deleted.
The sandbox was operator-invoked with an owner-authorized development boot key;
a production native simulator, task authority and execution offer remain
unconfigured. No real observation was sent by this qualification run.
