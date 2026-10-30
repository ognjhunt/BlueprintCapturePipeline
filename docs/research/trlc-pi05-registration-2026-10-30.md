# Community TRLC Pi0.5 registration

`install-public-trlc-policy` registers one fixed community checkpoint on the
retained `trlc_dk1_bimanual` body. It uses the existing operator-door source/model
verification and serialized registry compare-and-swap operation. It downloads
nothing, invokes no model, allocates no compute, and grants no task/capture
permission. Registration is not native execution evidence.

The candidate is `djkesu/pi05_trlc_rabc_3cam_alpha025_20260319_v2_step10000_merged`
at `10466a133c74018465c6839b7003b700775e4f69`. It is a community checkpoint,
not a TRLC vendor policy. Its training involved T-shirt folding. Running it on
`rigid_pick_place` is an explicitly out-of-distribution simulation diagnostic;
no cloth physics, folding benchmark result or physical performance is implied.
The original company/controller and generic custom-checkpoint catalog rows stay
separate. This variant uses `pi05_custom_checkpoint` and an empty company list.

## Exact request

Use the twelve fixed public-policy fields documented in
[the PiPER contract](openwam-piper-registration-2026-10-29.md), with:

| Field | Value |
|---|---|
| `schema_version` | `trlc_pi05_policy_install_request.v1` |
| `candidate_id` | `pi05_trlc_rabc_10466a13` |
| `team_id` | `blueprint-public-models` |
| `checkpoint_id` | `pi05-trlc-rabc-10466a13` |
| `policy_profile` | `pi05_trlc_rabc_joint14.v1` |

Bind the actual installed release, actual retained preparation publication,
fresh registry digest, stable installation ID and the current user's instruction
to run public policies in internal simulation. Do not invent publication digests
or copy a prior G1 task approval. The request accepts no tokenizer/source path,
URL, code, endpoint, command, credentials or new budget.

```sh
python3 scripts/operator_door.py install-public-trlc-policy \
  --request-file trlc-pi05-install.json --wait
```

The five-minute transient has `PrivateNetwork=yes` and `AF_UNIX` only. The CLI
waits six minutes by default. The service-user builder binds the prepared body
and exact runtime profile; independently installed stdlib root code reconstructs
the manifest, scoped rights, tokenizer evidence and complete expected
registration. It inserts/replaces only this stable candidate ID in registry
defaults and existing profiles. Identical retries preserve the original audit;
a changed registry digest refuses replacement instead of losing another update.

Read the result at
`/var/lib/blueprint-operator-door/policy-registrations/<installation_id>/receipt.json`.
The receipt states zero allocations and `native_policy_execution_validated=false`.
A disabled default catalog row preserves missing-registration and unverified
retained-tokenizer reasons before the first successful installation. Once a
scene can be planned, normal screening retains an individual result or reason.

## Tokenizer and public terms

The installer verifies only the six files in the fixed PaliGemma tokenizer
inventory, revision `35e4f46485b4d07967e7e9935bc3786aad50687c`, inventory SHA256
`5020414c0d782bb33cdf1f53b220569c83aff55356d2d920f7741657eb0fe830`.
It reads the existing fixed retained directory
`/var/lib/blueprint/task-evaluation-inputs/g1-841757-development/pi-tokenizer-f1eaa13`.
The historical directory name identifies bytes, not inherited task authority.
Each file's size and SHA256 or publisher Git-blob identity is verified; SHA256
also seals the resulting neutral `offline_tokenizer` record. Missing, altered
or symlinked files fail with a specific reason. No gated download or Hugging Face
credential is requested. Unlisted host receipts are not consumed or staged;
the runtime stages only the six listed files into its verified model environment.

The community card declares Apache-2.0. Its base merge metadata names
`lerobot/pi05_base`, whose card declares Gemma/PaliGemma; the exact training-base
revision remains unknown. Apache additions do not remove inherited terms.
`_vendor/trlc_pi05_notices/` retains the community/base cards, Apache text,
complete Gemma terms and prohibited-use policy, the required NOTICE and source
provenance. Website terms are complete article-text extractions with links and
original HTML hashes; website navigation/scripts are omitted. These fixed bytes
are verified by the installed profile and preserved in cloud runtime staging.

Rights are scoped to the current user-authorized internal simulation and the
public terms, including use restrictions, applicable law and notice conditions.
They assert no new human review, reused G1 task authority, vendor authorization,
model redistribution, public model service, official ranking, self-grading,
ADP evidence or physical validation. Operators must preserve the applicable
conditions; this integration does not grant permissions beyond them.

## Mechanical and inference boundary

Native control is 60 Hz; policy control is 30 Hz with 50 predicted/executed rows.
The exact robot manifest retains its source camera order, mounts, poses and FOV;
image sizes become 800×600 overhead and 960×540 at each wrist. Policy image order
is separately fixed to overhead, right wrist, left wrist. Camera calibration
and dataset joint zeros relative to the public URDF remain explicit simulation
assumptions. Both native grippers retain 0 open / −0.045 closed; closure channels
remain independent. Full 28D actions are unnormalized before selecting the 14
joint/gripper channels; unused EEF outputs remain diagnostics, not IK input.

The pinned compatible inference source is HumanoidArena's LeRobot subtree at
`68479287a784a69be9ce6ad739311d2f11f75ef9`, not a claim to the private training
fork. The 48 GB GPU floor is conservative and unmeasured. Exact checkpoint/source,
notices, tokenizer and isolated dependency verification, strict weight loading,
actual native motion and task results remain the runtime's work inside the
existing allocator, transfer/storage/TTL bounds and aggregate site budget.
