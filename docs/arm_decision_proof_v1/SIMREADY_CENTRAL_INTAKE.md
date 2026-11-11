# SimReady Central intake

This ADP-009D / Day-35 precursor adds a licensed-catalog source to the existing
static, external-validator and native-import boundaries. Catalog producer tests
do not qualify a Blueprint task, and native import does not prove physical
equivalence. The existing authored-graph path remains available.

## Reproducible selection

Select an exact catalog commit and a package path from its submission index:

```bash
python -m blueprint_pipeline.simready_central \
  --catalog-revision aca433cdd810b903ab0cfc34c8ac366f1d28acfb \
  --submission nvidia/simready-assets --asset food_beverage_can_b01 \
  --accepted-license CC-BY-4.0 --output-root output/catalog-can
python -m blueprint_pipeline.simready_central_qualification \
  --intake-receipt-path output/catalog-can/intake.json \
  --output-path output/catalog-can/static.json
```

The adapter resolves the dataset revision from the selected catalog URL. It
retains the entire package, including the original directory name, metadata,
license, textures, materials and all physics variants. It checks the publisher
BOM, immutable repository-tree hashes, byte lengths, and USD dependencies in
inactive variants as well as the selected composition. External dependencies,
path traversal, changed bytes and ambiguous selections block reuse. No
conversion or physics-variant selection is performed. `verify_intake()` reopens
every byte before each downstream boundary.

Use `--catalog-intake-receipt-path` with the existing
`blueprint_pipeline.external_simready_validation` command. Its catalog request
does not invent a capture ID or claim that third-party content is capture-derived.
The pinned published validator is supported by
`scripts/run_simready_validator_worker.py --installed-tier-version 2026.8.0` alongside the existing
source-tree installation. The environment must carry explicit compatible
licenses and version/source identities, as before. Missing reports fail closed.

`normalize_producer_results()` retains each source JSON pointer, inherited
timestamp, original details and boolean outcome. False, unknown and absent
results remain distinct. Current-validator optional failures stay failed
warnings; they do not become passes.

## Native boundary

```bash
python -m blueprint_pipeline.paired_target_native_import_bundle \
  --catalog-static-qualification output/catalog-can/static.json \
  --runner scripts/run_paired_target_native_import_probe.py \
  --output-root output/catalog-can/native-bundle \
  --implementation-commit <clean-published-40-character-commit>
```

This reuses the native execution-candidate audit and complete paid probe worker.
The bundle retains all package files in their original relative layout. The
worker checks dependencies before simulator startup and verifies actual PhysX
collider cooking, including native SDF resolution. Only catalog-inapplicable
agent-authored colors and authored-graph registration are omitted. Materials,
collision structure, byte identity, native execution, and teardown remain required.

The existing native-qualification join explicitly accepts the catalog static
receipt alongside the authored-graph receipt. A task-neutral catalog import
does not establish task registration, task physics, or task suitability. A task
freeze may be supplied to static qualification when there is an actual intended
task; it must not be invented for intake.

The canonical `paid_resource_allocator gpu-canary` remains the only launcher.
`materialize_paired_target_native_import_paid_attempt_authority()` accepts a
catalog bundle without unrelated scene-reconstruction predecessors. It requires
fresh explicit user authority, caps the aggregate allowance at that attempt's
cap, permits one allocation and zero automatic retries, and retains the same
provider-zero, watchdog, TTL and closeout checks. A catalog bundle never grants
paid execution by itself.

## Observed 2026-10-23

Retained local evidence is under `output/simready_central_20261023/`.
`can-layout-preserved/intake.json` records 47 files, approximately 5.1 MB,
dataset revision `653d91e182011f357cf1644013cc94ddf194d349`, CC-BY-4.0 and complete
dependency closure. The default composed variant is PhysX; the SDF collider and
0.034704443 kg mass were read from the original package.

The published `simready-validate==2026.8.0` and
`simready-foundation-tier-core==2026.8.0` ran twice in an isolated environment
with networking disabled. The result is `passed_advisory`: 25 findings, zero
blocking failures and nine optional failures, with matching repeatability
fingerprints. The install report retains transitive package hashes. The
publisher's older failed grasp-and-lift test is separately preserved.

The exact published intake commit `c084532b7196e279abfed36411e980100191e987`
then passed the canonical paid native probe on RTX 4090 instance `57392781`.
The probe verified import, GPU collision cooking, SDF resolution 256, and
zero-gravity physics-step readback in the pinned Isaac Sim 6.0.1 image.
`native-import-executed/catalog-native-import-execution.json` records
`native_simulator_import_qualified: true` with execution digest
`sha256:54128418d60197aa666376f84e894a57fe2570e3cdb40d8faed0e4e31b0c707a`.
This is library asset import qualification; contact behavior, grasp success,
task registration, task suitability, and physical equivalence remain unqualified.

The official provider charge was **$0.082** ($0.074 compute and $0.008 disk).
The attempt had a $2 cap and zero retries. The initial root-account invocation
was refused before allocation by the authority-ledger ownership check; the
canonical service account then consumed the same unused authorization once.
`native-import-executed/catalog-import-provider-zero.json` confirms provider
absence, stopped watchdog, and removal of every staged object. The native probe,
package bytes, static receipt, billing source/charge, and teardown evidence are
retained together under `native-import-executed/`.
