# Fixed public MolmoAct2 YAM registration

`install-public-yam-policy` adds the exact public MolmoAct2 bimanual YAM baseline
from a retained `i2rt_yam_bimanual` asset preparation. It uses the same scoped
operator-door installation and durable audit machinery as
[the PiPER operation](openwam-piper-registration-2026-10-29.md), preserving its
request and receipt formats. Installation never downloads weights, queries a
model, allocates compute, restarts a worker or supplies capture/task authority.

The operation accepts the same exact twelve request fields described there,
with these fixed values:

| Field | Value |
| --- | --- |
| `schema_version` | `molmoact2_yam_policy_install_request.v1` |
| `candidate_id` | `molmoact2_bimanual_yam` |
| `team_id` | `blueprint-public-models` |
| `checkpoint_id` | `molmoact2-bimanual-yam-8dcbed66` |
| `policy_profile` | `molmoact2_bimanual_yam.v1` |

The request must name the current installed release, existing preparation
publication, current registry digest and actual user instruction to run public
models. Reuse the same installation ID and identical sealed request on retries:

```sh
python3 scripts/operator_door.py install-public-yam-policy \
  --request-file molmoact2-yam-install.json --wait
```

The service-user proposal builder verifies retained source/import/model bytes.
It preserves the source-configured camera mounts and poses, sets the three RGB
views to 640×360 with horizontal FOV 69.4° overhead and 87° at each wrist, and
uses native 60 Hz with a 30-row chunk. The policy runs at an assumed 30 Hz and
executes the first 25 rows. Its exact binding reference labels these optics,
poses and cadence as initial simulation assumptions, including the overhead
D405 CAD model versus simulated D435i optics. These are not publisher or site
calibration. The requested 48 GB GPU floor is conservative and unmeasured for
the combined Blueprint simulator/model process.

The independently installed stdlib root verifier reconstructs the complete
registration from its own fixed profile, retained body, manifest and official
license evidence. It accepts neither arbitrary helper-produced registration
fields nor caller-selected paths, URLs, code, credentials or compute commands.
Only the selected YAM candidate changes in registry defaults and existing
explicit profiles. Existing PiPER and other candidates remain intact. The
shared installer lock serializes these operations; noncooperating administrator
writers must not race the final comparison and atomic replacement.

The checkpoint is `allenai/MolmoAct2-BimanualYAM` at
`8dcbed66f2380e4393189c303ea72488eb9e63c2`. Its Hugging Face card has no license
field. Rights evidence instead retains exact bytes of the official
[Ai2 README](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/README.md#7-license)
and [Apache 2.0 text](https://github.com/allenai/molmoact2/blob/6070080a20321b4f498ab30f28e1d09ac465edb7/LICENSE).
The README explicitly lists this checkpoint and states, “This model is licensed
under Apache 2.0.” Its research/educational intended-use statement and Ai2
Responsible Use Guidelines reference remain recorded. The profile verifies:

| Evidence | Bytes | SHA256 |
| --- | ---: | --- |
| README | 19,456 | `5b3cc2e6780938fbbf5d5f72f43368ff210b69b8d6cc0ff6daff8bfbdeef5909` |
| LICENSE | 11,357 | `c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4` |

The scoped rights record binds the exact inference-file identity, that evidence
and the user's request. It makes no private vendor authorization, deployment
company relationship, physical validation, official ranking, self-grading or
ADP claim. Registration alone does not assert downloaded weights, successful
native inference or a compatible authorized task. Ordinary screening retains
its rights and per-task budget checks and individual candidate outcomes.

Tests cover both fixed operation identities, real localhost CLI authorization
and retries, actual shell privilege separation with controlled stubs, exact
runtime/profile parity, preservation of all other registry fields and camera
poses, and rejection of changed model/license/proposal fields. They are
`development_only` and launch no compute. Installer upgrades only install
code, fixed profiles/evidence and empty directories; they create no request.

A local compatibility check also prepared the retained public YAM source, used
a development-only converter to produce real USD structure, and ran the actual
proposal builder and independent root verifier together, including idempotent
retry. It performed zero allocations. This was not an Isaac conversion or
learned-policy execution. The full operator-door regression suite and the
companion-runtime/profile parity test passed.
