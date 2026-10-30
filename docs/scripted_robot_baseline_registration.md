# Fixed scripted robot baseline registration

`install-scripted-robot-baselines` registers the Blueprint scripted pick/place
controller for every compatible retained body in one exact public asset
preparation batch. It runs no simulator, model server, provider call or allocator
and creates no capture, task, budget or vendor authority. Installing a release
or upgrading the door does not submit a registration request.

The operation enumerates these thirteen fixed configurations itself:

- `i2rt_yam_bimanual`
- `agilex_piper_bimanual`
- `franka_research_3_bimanual`
- `universal_robots_ur5e_bimanual`
- `kinova_gen3` and `kinova_gen3_bimanual`
- `ufactory_xarm7` and `ufactory_xarm7_bimanual`
- `almond_axol`
- `trlc_dk1_bimanual`
- `universal_robots_ur5e_robotiq_2f85_sim` and `universal_robots_ur10e_robotiq_2f85_sim`
- `franka_research_3_robotiq_2f85_sim`

The three single-arm Robotiq configurations use a separate sealed controller
profile; the original ten controller source identities remain unchanged.
Source-verified actuated grippers and the corrected Axol/TRLC closure directions
are required to construct the particular controller binding. Missing imports,
failed source/import verification and incompatible tool profiles have individual
unavailable rows; other bodies continue. Existing registrations are preserved
for an unavailable body. The receipt is not a claim that any robot ran.

An authenticated operator with `operate` scope supplies exactly these fields:

| Field | Binding |
| --- | --- |
| `schema_version` | `scripted_robot_baseline_install_request.v1` |
| `installation_id` | Stable identifier for this installation and retries |
| `release_commit` | Exact installed pipeline commit, 40 lowercase hex characters |
| `preparation_authorization_id` | Existing retained public asset preparation ID |
| `preparation_publication_digest` | Exact digest from its retained batch publication |
| `team_id` | `blueprint-owned-baselines` |
| `policy_profile` | `blueprint_scripted_pick_place.v1` |
| `expected_registry_digest` | Current `/etc/blueprint/team-evaluation-registry.json` digest |
| `authorization_reference` | Actual user instruction to run simulation baselines |
| `request_digest` | SHA256 of canonical JSON excluding this field |

Canonical JSON uses sorted keys, compact separators, UTF-8, `ensure_ascii=False`
and `allow_nan=False`, with the `sha256:` prefix. No body list, paths, URLs,
commands, code, endpoints or credential fields are accepted. Read the batch from
`/var/lib/blueprint/robot-models/batches/<preparation_authorization_id>.json`
through the operator door and use its exact publication digest. Do not invent a
successful import or a task authority to fill missing fields.

```sh
python3 scripts/operator_door.py install-scripted-robot-baselines \
  --request-file scripted-baselines-install.json --wait
```

Reuse the same installation ID and identical request for a lost-response retry.
A changed batch, release, registry or intentional retry of an unavailable body
uses a new request identity bound to the newly observed records. The operation
key is derived from the installation ID. A fixed ten-minute transient bounds
the operation; the CLI's default wait is eleven minutes.

The service-user builder re-verifies each retained source configuration,
imported USD, dependency closure and manifest. It preserves native camera poses,
optics, reset, gains, cadence and allowed chunk size. It builds a one-row scripted
policy specification from the reviewed source-tool grasp frames and records its
assumptions. Prepared proposal retries retain their original per-body outcomes.

The independently installed stdlib root code imports no pipeline checkout. It
verifies its exact three-file controller source inventory and MIT license
against a fixed reviewed profile, reconstructs every registration field, and
checks the full source-tool/TCP/grasp composition. Numeric matrix comparison
allows at most 1e-9 absolute rounding tolerance. No helper-selected extra policy,
endpoint, model path, calibration claim or rights field can be installed.
The profile snapshot is parity-tested against the controller code and all thirteen
source-frame definitions; changing controller code requires refreshing that
reviewed snapshot and installing the matching door release.

The shared PiPER/YAM installation lock serializes operations. Before/after
registry snapshots and an intent are durably retained before the final compare
and atomic replacement. Exact retries recover a completed replacement without
applying it again. A conflicting registry digest fails without overwriting the
observed change. Noncooperating administrator writers must not race the final
comparison and replacement. Each successful registration is inserted into
`derived_profiles.defaults.screening_candidates` and existing explicit profiles,
so subsequent ordinary task screening enumerates it automatically. Other
candidates, rights and settings remain intact.

Read the final receipt and audit at
`/var/lib/blueprint-operator-door/policy-registrations/<installation_id>/` through
the door. It retains all thirteen `results`, `registered_count`, source-bound registry
digests and `claim_ceiling=scripted_baseline_registration_only`. Its
`candidate_policy_queried`, `scripted_controller_invoked` and
`native_policy_execution_validated` are false, and allocation count is zero.

Controller rights bind Blueprint's MIT source license and the user's explicit
simulation request. They grant no rights over third-party robot models; the
existing source/import evidence remains attached. They claim no private vendor
relationship, learned policy inference, self-grading, official ranking, ADP or
physical validation. Ordinary task privacy, rights and per-site budget checks
still apply. Results retain the separate scripted evidence kind described in
[the controller contract](scripted_robot_baseline.md).

Validation covers actual localhost authorization and idempotent submission,
fixed networkless shell privilege separation, all-thirteen profile parity, partial
batch handling, code/license/model/policy tamper, registry conflict, concurrent
retry and recovery after replacement. A separate local check used retained
public YAM source, the actual configuration/import/binding and both installation
stages with a development converter producing real USD structure: one registered
body, nine individual unavailable rows, zero allocations. That converter was not
Isaac, and these checks establish no native execution or task success.
