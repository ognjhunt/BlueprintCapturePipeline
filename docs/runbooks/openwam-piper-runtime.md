# Public OpenWAM PiPER runtime

The public `OpenWAM/OpenWAM-Alpha-Real-RoboDojo-Piper` checkpoint is a separate
candidate on the prepared `agilex_piper_bimanual` embodiment. The candidate is
`openwam_alpha_piper_bimanual`; its interface is `openwam_piper_eef20.v1` and its
runtime profile is `openwam_alpha_real_piper.v1`. Its registration explicitly
uses the existing `customer_controller` route's controller-artifact option.
It does not replace any of the original 84 candidates/74 deployment route
pairs or the unrelated single-Franka OpenWAM candidate. PiPER is a body used by
Dream Labs; this public alternative is not evidence that Dream Labs uses
OpenWAM, endorses it, or provides its rights.

Installation requires retained native PiPER model/import receipts and genuine
scoped rights. The protected operator registry is the installation authority;
`registration_from_prepared_inputs` alone does not install a production
registration. The narrow operator-door registration operation resolves those
retained identities and updates the protected registry without launching a
provider. Deploying this runtime creates no registration and spends nothing.

The initial simulation configuration uses the prepared source's two independent
six-joint arms and parallel grippers, native control at60Hz, policy actions
at30Hz, and one action row per native policy query. The upstream model predicts
32 rows and executes an explicit buffer of8. Native control holds each policy
target for two steps, preserving the task's independent sampling/scoring rate.
The policy receives overhead, left-wrist and right-wrist RGB plus each arm's
measured pose in its own base frame. Upstream owns its384×320 composite and
normalization; no task predicate or scorer state enters the policy.

Per-arm base/TCP transforms are explicit. The initial identity transforms are a
simulation assumption: PiPER's source link6-to-gripper-base transform is
identity, while the publisher's exact recording TCP/firmware was not specified.
The prepared camera geometry and30Hz policy cadence are also assumptions,
not a claimed match to training calibration. They are retained in the sealed
configuration so later calibrations create a distinct configuration identity.

The allocator requests at least80GB GPU memory. This is a conservative setting;
peak OpenWAM-plus-Isaac memory and GPU latency have not been measured. Existing
site budgets, rate ceilings, transfer/storage charges, zero-retry allocation,
independent watchdog and teardown remain authoritative. An unaffordable offer,
missing compatible GPU, native failure, startup timeout or model failure stays
an individual candidate result/reason. No default budget or rate is raised.

The sealed bundle contains the exact upstream source archive and all541 source
file hashes. It does not need a Git checkout on the GPU. The isolated server
interpreter and inference dependencies have a fixed file inventory, byte sizes
and SHA256 hashes; there is no package resolver or source build on the provider.
The worker fetches only those dependency files and the eight fixed checkpoint
files from revision `12bfca6fb289174ab5a0f4f7eb363dc77ed265f0`. Every complete
file is verified before atomic publication and loading. Full checkpoint weights
were not loaded in CPU verification.

The transfer quote includes the actual bundle, any separately staged native
runtime packet, exact interpreter/wheel/checkpoint bytes, a64GiB cold image
allowance, and the8GB output allowance. All provisioning and startup run inside
the same admitted absolute provider deadline. No Hugging Face token is
forwarded. Model loading uses `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1` and
`DIFFSYNTH_SKIP_DOWNLOAD=true`. The server listens only on loopback; its exact
checkpoint/configuration handshake completes before the native worker starts.
The existing receipt-bound server teardown terminates that exact process.

CPU tests cover source/manifest tampering, local versus HTTPS dispatch, exact
checkpoint and robot binding, independent route accounting, bounded file
downloads, sealed preparation and allocator transfer parameters. They establish
software integration, not policy effectiveness or physical validation.
