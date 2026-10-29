# Fixed public PiPER registration through the operator door

`install-public-piper-policy` adds the fixed OpenWAM public PiPER baseline to
the operator-owned screening registry after a public-asset preparation batch
has retained that exact body. It does not allocate compute, query a policy,
call a provider, restart a worker, or establish capture/task authority.
Ordinary screening still enforces scene rights and its existing budget.

The authenticated `operate` request carries only these fields:

| Field | Constraint |
| --- | --- |
| `schema_version` | `openwam_piper_policy_install_request.v1` |
| `installation_id` | Stable 1–64 character alphanumeric/underscore/hyphen identity; reuse on retries. |
| `release_commit` | Exact installed 40-character lowercase Git SHA. |
| `preparation_authorization_id` | Existing retained preparation identity, same restricted ID grammar. |
| `preparation_publication_digest` | Exact `publication_digest` from the retained model-store batch record. |
| `candidate_id` | `openwam_alpha_piper_bimanual` |
| `team_id` | `blueprint-public-models`; this independent baseline does not assert Dream Labs endorsement. |
| `checkpoint_id` | `openwam-alpha-piper-12bfca6f` |
| `policy_profile` | `openwam_alpha_real_piper.v1` |
| `expected_registry_digest` | Current self-digest of `/etc/blueprint/team-evaluation-registry.json`. |
| `authorization_reference` | Printable reference to the actual user's instruction to run public models, 1–300 characters. Do not invent a vendor approval or put secrets here. |
| `request_digest` | Canonical SHA256 over every other field, using sorted keys, compact separators, UTF-8 and no NaN. |

No URL, path, endpoint, credential environment, code, command, budget or arbitrary
profile field is accepted. After reviewing the current registry digest and the
retained batch publication through ordinary read operations, submit the locally
sealed JSON:

```sh
python3 scripts/operator_door.py install-public-piper-policy \
  --request-file openwam-piper-install.json --wait
```

The CLI chooses a stable operation key from `installation_id`. The endpoint and
root spool validate the same exact request contract. The five-minute transient
has no network, hides credential paths, and runs active-pipeline validation as
`blueprint`. The active release must equal `release_commit`; the builder uses
the fixed retained PiPER publication and model store, validates the original
import/source closure, and writes a proposal under:

`/var/lib/blueprint/pipeline-control-plane/public-policy-registrations/<installation_id>/`

The body uses native control60Hz, action chunk one row; the policy is explicitly
assumed30Hz with native hold stride2. This preserves the existing task-scoring
cadence contract. The pinned model profile retains the camera/frame/cadence
assumption reference and upstream execution horizon8. Initial identity TCP
transforms are a source-supported simulation assumption, not a claim about the
publisher's recording firmware calibration.

Privileged installation runs only the independently installed stdlib door
module under `python3 -I -S`. It independently reconstructs the complete allowed
registration from the fixed profile, exact model-card bytes, retained PiPER
model/source records, and expected native manifest. An arbitrary helper JSON
blob cannot add endpoint/credential/command fields or select another file/body.
The operation inserts or replaces only this candidate under
`derived_profiles.defaults.screening_candidates` and existing explicit profiles;
all other registry fields remain unchanged.

The public rights record identifies the pinned model card's `Apache-2.0`
declaration and exact hash, checkpoint identity, actual user-request reference,
and scope. It explicitly disclaims private-vendor authorization, physical
validation, policy self-grading, official ranking and ADP evidence permission.
Registration does not assert downloaded weights or successful learned inference.

A privileged lock serializes these installation operations. The expected
registry digest detects unrelated changes observed before replacement;
noncooperating administrator tools must not race the final comparison and
atomic replacement. Before replacement, the installer
durably retains the request, before/after registry snapshots and installation
intent. It preserves the registry's ownership/mode and atomically replaces the
file, then stores a receipt. A retry after replacement but before receipt
publication recognizes the exact after-digest; matching retries return the
same result and a changed request with the same ID is refused.

Audit records live under:

`/var/lib/blueprint-operator-door/policy-registrations/<installation_id>/`

These are root-owned and readable by the door group; proposals remain private
to `blueprint`. Existing door read roots cover both locations while existing
secret-content checks remain active. An installer upgrade only installs code,
the fixed profile/card, and empty directories. It creates no request or run.

Validation includes strict request/identity tests, actual HTTP CLI scope and
retry tests, real shell execution against controlled stubs, a subprocess check
of the isolated privileged entrypoint, registry-field preservation, helper
tamper refusal, model-byte and symlink checks, concurrent retries, recovery
after replacement, and existing door upgrade rollback tests. Fixtures remain
`development_only`; none establish imported native policy execution.

A local source-derived check additionally prepared the retained public PiPER
source, generated real USD bytes through the existing fake SDK fixture, and ran
the actual proposal builder and independently reconstructed installer together.
Both completed with zero allocations. This checks their data compatibility;
the conversion was not an Isaac run and no learned model was loaded.
