# PhysX configuration sentinels in native reset evidence

Scope: ADP-009D, day-28 reusable Franka rehearsal. V23, frozen at
`ed97253b9dbe5c9978c9e2f98927993923091ef4`, retained complete lighting and
scene-asset channels but reported `physics:ValueError` and
`colliders:ValueError`. Its complete-reset continuation gate correctly refused
that evidence. This repair changes only native configuration serialization and
failure diagnostics; it does not change that gate, actions, scoring, or receipts
from earlier runs.

## Observed SDK facts

Read-only inspection of instance `50461698` before teardown found Isaac Sim
`6.0.1-rc.7+release.42383.32955d8d.gl` and the installed extension
`omni.usd.schema.physx-110.1.13+110.1.2.lx64.r.cp312.u7f4`.
Its resource root was
`/isaac-sim/extscache/omni.usd.schema.physx-110.1.13+110.1.2.lx64.r.cp312.u7f4/plugins/PhysxSchema/resources`.

| Resource | SHA-256 | Relevant declarations |
| --- | --- | --- |
| `schema.usda` | `eaef57fc794f6114d3641c00bbc3f3141abc9f8bfe8480d61d4bafc4a33a1904` | Lines 171–180 and 781–804 |
| `generatedSchema.usda` | `4a7638c3890522bbe7c25600c0090e6cb7f403550ccb8be7641bf84034213b83` | Lines 192, 432, and 446 |
| `plugInfo.json` | `e274f45c131722ff7909a200e11e35dab014c39cfd522c015e177b3895f3a2c0` | `PhysxSceneAPI` and `PhysxCollisionAPI` are single-apply schemas |

| Attribute | USD type | Exact default | Configuration meaning |
| --- | --- | --- | --- |
| `physxScene:maxBiasCoefficient` | `float` | `inf` | Maximum solver-bias coefficient |
| `physxCollision:contactOffset` | `float` | `-inf` | Simulation selects the offset from shape extent |
| `physxCollision:restOffset` | `float` | `-inf` | Simulation selects a suitable rest offset |

These attributes are on the existing reset reader's selected namespaces. A CPU
OpenUSD 0.26.5 process with a codeless schema containing these exact declarations
reproduced `ValueError` from the original serializer for all three defaults.
No live worker, model, simulator process, or provider state was modified.

## Serialization boundary

Only these three exact attribute/type/token combinations are additionally
retained, and only when their owning PhysX API is applied. Their output records
the schema, configuration meaning, token, and `measured_numeric_value: false`.
The source is explicitly `usd_schema_fallback` or
`usd_authored_configuration`; only fallback and authored default resolution are
admitted. An authored documented sentinel retains its native configuration
meaning without being represented as a numeric measurement.

Arbitrary infinity/NaN measurements, unknown attributes, opposite-sign tokens,
unsupported USD types, and attributes without the owning API still fail.
The existing center-of-mass exception remains fallback-only: authored nonfinite
COM values still fail. A finite setting or a fallback/authored distinction
changes paired reset comparison. No computed contact distance is invented.

Future attribute conversion failures preserve the coarse channel/error gap and
add the attribute path, USD type, resolve source, and exception category. They
do not retain the value or the SDK exception's message. Existing sealed receipts
are unchanged.

## Verification and remaining uncertainty

The new CPU cases use a fresh interpreter per codeless-schema registration so
USD's process-wide schema cache cannot make tests order-dependent. They exercise
the real USD stage and native reset reader, complete-reset and paired-comparison
checks, both configuration sources, unsupported values/types/APIs, numeric
measurement refusal, and diagnostic redaction.

```sh
python -m pytest -q -m 'not gpu' tests/test_native_physx_reset_serialization.py tests/test_native_usd_reset_serialization.py tests/test_policy_scientific_reset.py
python -m pytest -q tests/test_native_task_arena_policy_canary_lifecycle_rehearsal.py tests/test_provider_runtime_import_closure.py
```

V23's saved gaps contain only exception categories, so they do not establish the
exact failing attribute or its resolve source. The static schema facts and CPU
reproduction are established; a later native run must still prove every required
channel complete. No live success, policy success, or qualified comparison is
claimed by this repair.
