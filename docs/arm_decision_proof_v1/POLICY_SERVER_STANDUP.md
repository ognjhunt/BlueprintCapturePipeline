# ADP-009D policy server standup

Backlog item: **ADP-009D — Deterministic variation, abstention, and full rehearsal**
(`IMPLEMENTATION_BACKLOG.md`). Day gate unblocked: the two-candidate Franka Task
Evaluation Run cannot start until the Isaac runtime can query a real DROID policy,
and today it cannot — `adp009d_isaac_runtime.py` emits
`"candidate_policy_queried": false` on every path.

This is design and contract only. No paid GPU is allocated, no checkpoint is
downloaded, and nothing here is a claim that a policy has run.

Contract module: `src/blueprint_pipeline/adp009d_policy_server_standup.py`.
Tests: `tests/test_adp009d_policy_server_standup.py`.

---

## 1. Recommendation

**One Vast instance, one container, two processes, two interpreters, loopback
transport.** In the contract module this is
`topology.mode = "shared_worker_separate_interpreter"`.

```
Vast instance (1x 48 GB NVIDIA, ssh_direct)
└── container  nvcr.io/nvidia/isaac-sim:6.0.0-dev2@sha256:c3e7bef5…
    ├── /isaac-sim/python.sh  →  adp009d_isaac_runtime.py      (Kit + PhysX + RTX)
    │                             └── policy client ──┐
    ├── /opt/blueprint-policy-env/bin/python          │  127.0.0.1:<port>
    │     └── candidate policy server  ◄──────────────┘
    └── /workspace/blueprint_adp009d_checkpoint/      (materialized on the worker)
```

The reason it is two interpreters and not two containers is that there is no
second container available: `adp009d_franka_vast.run_adp009d_native_microcheck_vast`
routes through `adp_isaac_lab_arena_vast.run_arena_native_control_vast`, which
launches with `vast_launch_mode="ssh_direct"` and runs exactly one
`runtime_entrypoint` (`provider_runtime/run_adp_arena_provider_runtime.sh`) in one
`container_image`. A sidecar container would be a new transport seam, not a
configuration change.

The reason it is two interpreters and not one is §2 — and it is not a new idea:
this repo already ships an Isaac Sim image running a GR00T policy server, a
client, and Isaac in three separate interpreters on one GPU worker.

---

## 2. Can the policy server share Isaac's container?

**Yes — the container. No, not the interpreter.** Splitting those two questions
is the whole answer, and this repo has already answered both, in production.

### 2.0 It is already done, and consolidating it is a build blocker

The shipped `groot_oscar_closed_loop` image runs **four Python environments in
one Isaac Sim container**: `/isaac-sim/python.sh`, `/opt/oscar-venv`,
`/opt/gr00t-venv`, and `/opt/runpod-serverless-venv`. A GR00T *policy server*,
an OSCAR client, and Isaac Sim run as three concurrent processes in three
different interpreters on one GPU worker. This is the exact shape ADP-009D needs.

It is not incidental — it is a governed contract.
`scripts/verify_groot_oscar_thin_architecture.py:133-138` fails the build if
anyone merges them:

```python
if "/opt/robot-venv" in foundation:
    blockers.append("foundation_uses_unproven_consolidated_robot_environment")
if foundation.count("uv venv /opt/oscar-venv") != 1:
    blockers.append("foundation_oscar_environment_not_isolated")
if foundation.count("uv venv /opt/gr00t-venv") != 1:
    blockers.append("foundation_groot_environment_not_isolated")
```

And `oscar_isaac_closed_loop_eval.py:8061-8066` states the rationale outright:

> The sealed worker deliberately keeps GR00T, OSCAR, and Isaac in separate
> interpreters.

**Correction to an earlier assumption of mine:** those venvs are built with
`uv venv --python 3.10` while Isaac's interpreter is 3.12, *in the same
container* (`groot_oscar_closed_loop/Dockerfile:202, 232`). The policy
environment therefore does **not** need to match Isaac's Python minor. I had
initially written that rule into the contract; it is removed, because it would
have forbidden the topology that already works. Isolation is the gate, not
version parity.

### 2.1 What is verified about Isaac's Python

| Fact | Evidence |
| --- | --- |
| Isaac's interpreter is invoked as `/isaac-sim/python.sh` | `adp009d_native_microcheck_bundle.py` `ENTRYPOINT`, line 69; `adp009d_native_microcheck_worker.py` `ISAAC_PYTHON` |
| It is **CPython 3.12** | The worker installs `h5py-3.16.0-cp312-cp312-manylinux_2_28_x86_64.whl`, `msgpack-1.1.0-cp312-cp312-…whl`, `pyzmq-27.0.1-cp312-abi3-…whl` into it (`adp009d_native_microcheck_worker.py:33-68`). A cp312 wheel cannot install into any other minor. |
| It is **pip-mutable, and we already mutate it** | `_install_commands` issues five `[str(ISAAC_PYTHON), "-m", "pip", "install", …]` steps (`adp009d_native_microcheck_worker.py:196-224`) |
| It already has **torch** | `import torch` in `_assert_arm_pose` and `_build_environment` (`adp009d_isaac_runtime.py:244, 613`) |
| It already has **pyzmq + msgpack** | `ARENA_REMOTE_TRANSPORT_URLS`; `_preflight_environment_imports` imports `zmq` and `msgpack` (`adp009d_isaac_runtime.py:749-790`) |
| Its install profile is **fail-closed against expansion** | `_validate_install_commands` raises `adp009d_runtime_install_profile_expanded` (`adp009d_native_microcheck_worker.py:224-249`) |
| The final `pip freeze` is retained as `isaac_runtime_lock.txt` | `adp009d_native_microcheck_worker.py` `finally:` block |

So the container is not sealed, and installing into Isaac's interpreter is a
thing this repo already does deliberately and pins by URL and hash.

### 2.2 Why the interpreter still should not be shared

The blocker is not permission, it is **which accelerator framework each server
needs**, and that differs per candidate:

| Candidate | Server framework | Client import | Transport |
| --- | --- | --- | --- |
| `pi05_droid` | **JAX** | `from openpi_client import websocket_client_policy` | `openpi_websocket_msgpack_numpy` |
| `groot_n17_droid` | PyTorch | `from gr00t.policy.server_client import PolicyClient` | `nvidia_groot_zmq_msgpack` |
| `groot_n16_droid` | PyTorch | same | `nvidia_groot_zmq_msgpack` |
| `cosmos3_edge_policy_droid` | PyTorch | `from openpi_client import websocket_client_policy` | `openpi_websocket_msgpack_numpy` |

JAX for pi05 is verified, not assumed: `openpi_policy_ranking_gpu_job._gpu_runtime_evidence`
does `import jax`, calls `jax.devices()`, and raises the blocker
`jax_gpu_device_not_present`; the cleanup path calls `jax.clear_caches()`
(`openpi_policy_ranking_gpu_job.py:116-139, 302-306`).

Installing `jax[cuda]` into Isaac's interpreter would pull a second set of
`nvidia-*` CUDA wheels next to the ones Isaac's own pinned torch resolved
against, inside an interpreter whose install plan is explicitly guarded against
expansion. Installing GR00T or `cosmos_framework` instead would resolve *torch*
against Isaac's pin. Either way the failing artifact is Isaac Sim, mid-run, on a
paid worker.

JAX in particular has **never** been put inside an Isaac container in this repo.
It lives only in `deploy/docker/policy_ranking_openpi/`, on a plain
`nvidia/cuda:12.2.2` base, in its own `uv venv --python 3.11.9` — and that image
sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80` because it owns the whole GPU. See §9.

The refusal to share an environment is also explicit elsewhere:

```sh
# scripts/setup_cosmos3_edge_env.sh
if [[ "$edge_env" == "$repo_root/.venv" || "$edge_env" == "$repo_root/.venv/"* ]]; then
  echo "Cosmos dependencies must not use the core .venv" >&2
  exit 2
fi
uv venv --python 3.12 "$edge_env"
```

### 2.3 The one case where sharing is defensible

`groot_n17_droid` and `groot_n16_droid` speak ZMQ + msgpack, and Isaac's
interpreter already has `pyzmq` and `msgpack` installed for Arena's own remote
transport. The **client** side of the GR00T candidates therefore needs nothing
new inside Isaac's interpreter — only the *server* side needs the GR00T stack.
That is why the client always runs in Isaac's process and only the server is
moved out.

`shared_worker_shared_interpreter` remains in `SUPPORTED_TOPOLOGIES` for the case
where a future image already ships the framework, but the contract requires
`accelerator_framework_installed_into_isaac_interpreter: false` to admit it —
i.e. it is admissible only when nothing had to be installed.

---

## 3. Environment isolation

Follow the pattern the shipped image already uses, and copy its guard rails.

```sh
# inside the container; uv installs itself if absent, per
# unitree_groot_n17_sonic_provider_smoke._install_uv
uv venv /opt/blueprint-policy-env --python <candidate minor> --seed
VIRTUAL_ENV=/opt/blueprint-policy-env uv sync --frozen --project <candidate-source>
# Blueprint goes in with --no-deps so it cannot perturb the resolved set,
# exactly as deploy/docker/policy_ranking_openpi/Dockerfile:59 does.
VIRTUAL_ENV=/opt/blueprint-policy-env uv pip install --no-deps -e /opt/blueprint
```

`--system-site-packages` is deliberately absent. The repo does have a
`--system-site-packages` path (`oscar_wam_provider_bundle.py:3416-3441`) but it
is off by default, and it is not the policy-server pattern.

Isolation is **measured, not assumed**, because a leak has already broken a lane
here once: `groot_oscar_worker_startup_script.py:72-100` raises
`groot_runtime_isolation_failed` on three checks — exact `sys.prefix`, no foreign
path on `PYTHONPATH`, no foreign path on `sys.path` — and then separately asserts
`accelerate` resolved from the right venv, because it once did not.

The standup mirrors those checks:

```json
"interpreter_isolation": {
  "isaac_site_packages_on_policy_sys_path": false,
  "policy_interpreter_prefix_exact": true,
  "policy_sys_path_digest": "sha256:…"
}
```

Refusals: `policy_server_standup_policy_env_not_isolated`,
`policy_server_standup_policy_interpreter_prefix_inexact`,
`policy_server_standup_policy_sys_path_digest_invalid`.

I still could not read what `/isaac-sim/python.sh` itself exports (see §10),
which is precisely why the contract requires the measurement rather than a
version rule. Launch the server with a scoped environment — the existing launcher
sets only `PYTHONPATH` explicitly before `Popen`
(`unitree_groot_n17_sonic_provider_smoke.py:987-1020`).

If ADP-009D ends up baking this into an image rather than building it at runtime,
note that `scripts/verify_groot_oscar_thin_architecture.py` counts
`uv venv /opt/<name>-venv` occurrences and will need a matching assertion.

---

## 4. Checkpoint materialization

**On the worker, never locally.** The orchestrating machine has 2.9 GiB free
(measured); the smallest frozen checkpoint is `nvidia/GR00T-N1.6-DROID` at
6,573,569,204 bytes and the largest is `gs://openpi-assets/checkpoints/pi05_droid`
at 12,429,488,598 bytes. Nothing in the set fits.

Materialization happens in the container, before either server binds:

- credentials resolve via `blueprint_pipeline.model_access_env.normalize_model_access_env()`,
  which bridges `HF_TOKEN` / `HUGGINGFACE_HUB_TOKEN` / `NGC_API_KEY` aliases and
  their `*_FILE` forms. It writes no secret to any artifact
  (`raw_secret_written_to_artifacts: false`), and no standup field carries a
  token value.
- the byte count and identity are bound to the already-frozen inventory in
  `adp009d_policy_candidate_admission.EXPECTED_CANDIDATES`: `checkpoint_repository`,
  `checkpoint_revision`, `checkpoint_total_bytes`, `checkpoint_inventory_digest`.
  The receipt restates none of these — it is checked against them.
- a `materialization_digest` covers the bytes that actually landed. Without it
  the receipt would prove which checkpoint was *named*, not which one was served.

The contract fails closed on a short download
(`policy_server_standup_checkpoint_materialized_bytes_mismatch`) because several
checkpoint formats load happily from a truncated tree and then serve weights that
are not the frozen ones — a silent wrong-model run is worse than a hard stop.

`checkpoint.materialized_on` must be `"gpu_worker"` and
`checkpoint.orchestrator_bytes` must be `0`.

---

## 5. Startup ordering

`STARTUP_PHASES` is frozen and the receipt must list it exactly, in order:

1. `worker_admitted`
2. `policy_environment_created`
3. `checkpoint_materialized`
4. `checkpoint_verified`
5. `isaac_runtime_started`
6. `policy_server_started`
7. `policy_server_endpoint_accepting`
8. `identity_metadata_verified`
9. `inference_round_trip_verified`
10. `isaac_policy_query_enabled`

Two orderings carry the argument.

**Checkpoint before either server (3-4 before 5-6).** The download is the long
pole on a per-second-billed worker, and a checkpoint that fails its digest check
should never cost a Kit boot. It also overlaps naturally with the existing Arena
git clone and pip install, which the worker already does before Kit starts.

**Isaac before the policy server (5 before 6).** Isaac is the *fragile* consumer.
`adp009d_isaac_runtime.main` wraps `_run` in `except Exception` — which catches
Python failures and writes a structured `blocked` result, but cannot catch a
native abort from Kit or PhysX. A policy server that cannot fit its weights
raises an ordinary allocation error inside its own process, which we can catch,
attribute, and turn into a typed blocker. Letting the recoverable process claim
memory *second* is what makes co-residency measurable instead of fatal. The
contract inverts to `policy_server_standup_startup_phase_order_invalid` if these
are swapped.

The consequence is that the Isaac runtime must idle-wait on the policy endpoint
between phases 5 and 10, rather than assuming it. That is the one real code seam
this plan adds to `adp009d_isaac_runtime.py`: a bounded readiness poll after
`_build_environment` and before the first policy query.

Do not invent that poll. Two working templates exist, and both check liveness
*and* reachability rather than just sleeping:

- shell, `groot_oscar_digitalocean_closed_loop_job.py:1663-1692` — background the
  server, then loop on `/proc/<pid>` existence **plus** a TCP connect to
  `127.0.0.1:<port>`, 900 s deadline, 5 s poll, distinguishing
  `groot_policy_server_exited_before_ready` from `groot_policy_server_not_ready`;
- Python, `unitree_groot_n17_sonic_provider_smoke.py:1022-1108` — same two-part
  check via `_tcp_ready`, with a 30 s heartbeat phase marker and three typed
  terminal outcomes (`…_exited_before_listening`, `…_startup_timeout`, listening).

Both also `trap`/reap the server process on exit, which is what
`teardown.policy_server_process_terminated` in the receipt records.

Note that the current entrypoint runs `set +e` and falls back to writing a
`blocked` result if the runner exits without one
(`adp009d_native_microcheck_bundle.ENTRYPOINT`). A policy-server supervisor added
to that script must preserve that property: the run must still return a
structured result when the server dies.

---

## 6. Readiness detection

**A loaded model is not a working server.** The existing precedent is exactly the
bug to avoid — `cosmos_edge_droid_policy_server.serve_identity_bound_policy`
writes

```python
"status": "model_loaded_ready_to_serve",
```

to `policy_server_startup.json` and only *then* calls `server.serve_forever()`.
That record cannot distinguish a server that answers from one that never binds
its port, that returns empty metadata, or that returns a chunk the ADP-009D
action adapter would reject.

`validate_policy_server_standup` therefore refuses `ready: true` unless the
receipt carries a completed round trip:

| Field | Refusal |
| --- | --- |
| `inference_round_trip` present | `…_inference_round_trip_missing` |
| `completed: true` | `…_inference_round_trip_incomplete` |
| `observation_digest`, `action_digest` | `…_round_trip_<field>_invalid` |
| `observation_adapter_schema_version == adp009d_droid_observation.v1` | `…_round_trip_observation_adapter_invalid` |
| `action_shape[1] == 8` | `…_round_trip_action_width_invalid` |
| `action_shape[0] >= 8` (`DROID_OPEN_LOOP_HORIZON`) | `…_round_trip_action_rows_insufficient` |
| `action_finite: true` | `…_round_trip_action_nonfinite` |
| `latency_ms > 0` | `…_round_trip_latency_invalid` |
| `listening_socket_confirmed: true` | `…_endpoint_socket_unconfirmed` |

Binding the round trip to `adp009d_droid_observation.v1` matters: a round trip on
a hand-built array proves the server answers *something*, not that it answers
what Isaac will actually send. The observation the probe sends should come from
`build_droid_observation` on a real Isaac frame.

Latency is required because the loop's budget is already fixed:
`isaac_steps_per_droid_action()` returns `1` — `sim.dt = 1/120` with
`decimation = 8` is exactly DROID's 15 Hz, one action row per environment step.
An unmeasured round trip leaves the 66.7 ms/step question open until the paid
matrix is already running.

The endpoint is loopback-only (`127.0.0.1`, `localhost`, `::1`), matching
`openpi_droid_policy_runtime.serve_identity_bound_policy`, which raises
`openpi_policy_server_must_be_loopback_only`. A routable policy port on a rented
machine is an open inference endpoint serving weights Blueprint does not own.

---

## 7. Teardown

The Vast instance is destroyed after each run and nothing persists, so teardown
is about what must *not* survive and what must be proven:

- `teardown.policy_server_process_terminated: true` — the server must be reaped
  before the runtime returns, or the entrypoint can hang past the TTL holding a
  billed instance.
- `teardown.checkpoint_retained_on_orchestrator: false` — the checkpoints are
  public and re-materializable from their frozen digests; copying 12.4 GB back to
  a host with 2.9 GiB free would fail the run for a file we do not need.
- `teardown.provider_zero_required_after_return: true` — matches the existing
  bundle manifest field and the paid-lane guard.

The existing hard caps apply unchanged: `max_hourly_rate_usd=1.00`,
`hard_cap_usd=4.00`, `hard_ttl_seconds=14_400`, zero retries
(`adp009d_franka_vast.py`). Adding a second process does not add a second budget.

---

## 8. Failure modes each choice guards against

| Choice | Failure it prevents |
| --- | --- |
| Separate interpreter | JAX/torch CUDA wheels resolving against Isaac's pins and breaking Kit mid-run |
| Measured `sys.path` + exact `sys.prefix` | The `accelerate`-out-of-the-wrong-venv class of failure, already seen here once |
| JAX preallocation disabled | XLA taking 80 % of the card and aborting Isaac natively mid-run |
| Worker-side materialization | A 12.4 GB download onto a host with 2.9 GiB free |
| Byte-count equality | A truncated checkpoint that loads and serves the wrong weights |
| Digest-bound checkpoint identity | A receipt that proves which checkpoint was *named*, not which was served |
| Isaac starts before the server | An unrecoverable native abort instead of a catchable Python OOM |
| Round-trip readiness | The `model_loaded_ready_to_serve` gap — a bound-but-broken server |
| Action shape + width check | A chunk the Isaac action adapter rejects, discovered mid-matrix |
| Per-candidate transport | Serving GR00T over a websocket, stranding its own ZMQ client |
| Loopback-only | An open inference endpoint on a rented machine |
| Frozen phase list | A skipped verification presented as a fast path |
| Forbidden outcome keys | A standup receipt asserting a task outcome that cannot yet exist |

---

## 9. Biggest risk

**If `pi05_droid` is one of the two frozen candidates, JAX will preallocate the
GPU out from under Isaac Sim.**

JAX claims a fraction of the device on first use, and Blueprint's own OpenPI
image sets `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`
(`deploy/docker/policy_ranking_openpi/Dockerfile:39-40`) because on that image
JAX owns the entire GPU. Co-resident with Isaac Sim, that setting takes 80 % of
a 48 GB card and leaves Isaac to fail — as a **native abort**, since
`adp009d_isaac_runtime.main` can only catch `Exception`. The paid run dies with
no structured blocker.

This risk is specific and asymmetric:

- It affects **only** `pi05_droid`. The GR00T and Cosmos candidates are PyTorch,
  which allocates on demand, and GR00T is already proven co-resident with Isaac
  in the `groot_oscar_closed_loop` image.
- JAX has **never** run inside an Isaac container in this repo. Every existing
  JAX lane is a standalone `nvidia/cuda` image. So this combination is genuinely
  untested here, not merely undocumented.
- The startup ordering in §5 does not save us. Isaac starts first, but JAX's
  fraction is taken at first use, after Isaac has already built its scene — so
  the failure lands mid-run rather than at startup.

Mitigation, encoded in the contract: `shared_worker_*` topologies with a `jax`
server framework must record
`accelerator_memory_guard.xla_python_client_preallocate: false`, or the standup
is refused with `policy_server_standup_jax_preallocation_not_disabled`. A smaller
`MEM_FRACTION` is not accepted in its place — it narrows the race rather than
removing it.

If disabling preallocation costs pi05 too much throughput to hold 15 Hz, the
answer is `separate_worker` (the goal prompt's own fallback: one L40S-class 48 GB
worker for Isaac, one RTX 4090-class 24 GB worker for the policy server), not a
tuned fraction.

**Second risk — Isaac's bundled torch version.** It is not pinned anywhere in
this repo, and the release image's import matrix
(`groot_oscar_model_cache_s3_remote_executor.py:730-759`) deliberately probes
torch only inside `gr00t-venv` and `oscar-venv`, never through
`/isaac-sim/python.sh`. If ADP-009D ever wants the shared-interpreter topology
it needs that version. The cheapest way to get it costs nothing: every
micro-check already retains `isaac_runtime_lock.txt` from
`/isaac-sim/python.sh -m pip freeze`. **Read the torch line out of the last
completed ADP-009D run's lock file** before designing any install step.

**Third risk — Cosmos is unusable in this scene regardless of standup.** The
`cosmos3_edge_policy_droid` candidate needs three camera views and the ADP-009D
scene deliberately has two: `adp009d_droid_observation` raises
`droid_observation_third_view_outside_frozen_two_camera_contract` and the runtime
sets `external_camera_2 = None`. A Cosmos policy server would stand up
successfully and then have nothing to serve. That is a programme decision, not a
standup problem, but it should be settled before any Cosmos checkpoint is pulled.

---

## 10. Verified vs. assumed

**Verified from source in this repo:**

- The entrypoint, container image digest, and single-container `ssh_direct`
  transport.
- That a policy server, a client, and Isaac Sim **already run concurrently in
  three separate interpreters in one Isaac Sim container**, and that merging them
  is a build blocker (`verify_groot_oscar_thin_architecture.py:133-138`).
- That the policy environment's Python minor need **not** match Isaac's: 3.10
  venvs ship beside Isaac's 3.12 today.
- Isaac's interpreter path, its CPython **minor** version (3.12, from cp312 wheel
  pins), that it is pip-mutable, and that we already mutate it under a
  fail-closed profile guard.
- That the ADP-009D Isaac runtime imports `torch`, and that Isaac's interpreter
  has `pyzmq` and `msgpack` installed for Arena's remote transport.
- That pi05 needs JAX, the other three need PyTorch, and that **JAX has never run
  inside an Isaac container here** — every JAX lane is a standalone
  `nvidia/cuda` image with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.80`.
- Each candidate's client import path and transport.
- The loopback-only rule and the `model_loaded_ready_to_serve` readiness gap.
- Working readiness-poll and process-reaping templates, and the runtime isolation
  gate (`groot_runtime_isolation_failed`) added after a real cross-venv failure.
- That `uv` is installed on the fly when absent
  (`unitree_groot_n17_sonic_provider_smoke._install_uv`).
- 15 Hz / one-row-per-step timing, the 8-wide action row, and the 8-row open-loop
  horizon.
- 2.9 GiB free on the orchestrating machine (measured `df`).

**Not verified — assumed, and marked as such:**

1. **What `/isaac-sim/python.sh` exports.** I did not open the container: doing so
   means a GPU allocation or a ~20 GB image pull onto 2.9 GiB. No file in the repo
   reads or documents the script's contents. The plan turns this into a required
   measurement rather than a guess.
2. **Isaac's bundled torch version**, and whether it satisfies GR00T or
   `cosmos_framework`. Not pinned anywhere; the release image's probe matrix
   deliberately never imports torch through `python.sh`. See §9.
3. **The CPython patch version and the real interpreter path** behind `python.sh`.
   Only the minor is evidenced, inferred from cp312 wheel pins and the Ubuntu
   24.04 base. No `/isaac-sim/kit/python/bin/python3` reference exists in the repo.
4. **VRAM split.** No measurement exists. `min_gpu_ram_mb=46_000` is what we
   *request*, not what Isaac and a policy server *use*. The contract requires the
   observed split rather than accepting the advertised size. The illustrative
   numbers in the test fixture are placeholders, not observations.
5. **Whether disabling XLA preallocation still lets pi05 hold 15 Hz.** The §9
   mitigation is untested; the fallback is `separate_worker`.
6. **The GR00T server entrypoint at the pinned revision** `b9955401…`. The
   recorded launch shapes (`gr00t/eval/run_gr00t_server.py --model-path …
   --embodiment-tag … --port 5550`) are both for the Unitree Sonic embodiment,
   not DROID, so the DROID flags may differ.
7. **pi05's served action-chunk row count.** `openpi_droid_policy_runtime` accepts
   10 or 15 rows; the ADP-009D inventory pins no row count for pi05. The contract
   only enforces `>= 8`.
8. **Whether the `gs://openpi-assets` pull needs credentials.** Stated public;
   not tested.
9. **Download and load wall-clock** on a Vast worker, which sets how much of the
   4 h TTL the standup consumes before the matrix can start.
10. **The port numbers** in this document. 5555/5550 (GR00T) and 8000 (openpi) are
    the defaults each runtime declares; ADP-009D has not chosen its own.
