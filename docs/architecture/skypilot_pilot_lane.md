# SkyPilot pilot lane (C4 step b) — isolated, one lane, guards keep money truth

Verdict encoded here (from the 2026-08-02 build-on-top audit and its review):
normalize provider transport now (`provider_transport.py`); evaluate SkyPilot
as an isolated, lane-specific provisioning backend; do **not** let it replace
the canonical allocator, warm-capacity lifecycle, readiness race, spend
controls, inventory reconciliation, or teardown authority until it proves
behavioral parity.

## Boundaries

- **Out of process.** Pinned `skypilot[vast]==0.13.0` lives in a separate
  venv (`scripts/setup_skypilot_venv.sh` → `.venvs/skypilot`), exposed only
  as a CLI via `BLUEPRINT_SKYPILOT_BIN`. It is never imported by or
  installed into the Blueprint runtime (a dry-run install would downgrade
  Click 8.4→8.1, Uvicorn 0.51→0.35, Pillow 12.3→12.2), and it is therefore
  absent from the runtime SBOM/license policy; it is reviewed in
  `docs/architecture/isolated-component-license-inventory.md`.
- **One lane.** `skypilot_provisioner.launch_disposable_vast_smoke` is the
  only mutation surface: cold, disposable, on-demand Vast smoke work. The
  task YAML must pin `cloud: vast` and a positive `max_hourly_cost`
  (fail-closed constraint check). No RunPod (SkyPilot's RunPod backend
  cannot stop pods and overrides entrypoints), no warm reuse, no retained
  sessions, no multi-node, no Windows.
- **Grant-gated.** Mutations require a `PaidResourceAdmissionGrant` for the
  new resource class `skypilot_vast_pilot` issued by
  `python -m blueprint_pipeline.paid_resource_allocator` — the adapter is
  registered as a grant-gated surface in
  `docs/architecture/paid-resource-mutation-surfaces.json` and verified by
  `scripts/verify_paid_resource_allocator.py`. It has no CLI launcher.
- **Teardown authority stays custom.** Every launch opens
  `pending_teardown.v1` first; `sky launch --down` and `sky down` are
  attempts, not proof. Teardown is proven only by provider-API inventory
  (`build_teardown_proof(..., status_source="provider_api")`, provider-zero
  for the cluster label). SkyPilot documents that provisioning/setup errors
  may intentionally leave clusters and that `--purge` merely forgets local
  state — so a nonzero `sky launch` is classified
  `allocation_outcome_ambiguous`, never "nothing was created", and an
  unproven teardown stays open for `paid_lane_guard reap-orphans` as
  explicit open billing risk. Orphan recovery must survive SkyPilot's own
  state database disappearing.

## What SkyPilot is *not* here

- Not the readiness race: `provider_race.py` launches contenders
  concurrently and accepts the first to produce the Blueprint bootstrap
  marker (`bootstrap.json` + launch session nonce). SkyPilot's optimizer is
  sequential-by-cost — a different policy, kept side by side.
- Not the spend ledger, preflight, or cumulative budget — it has none.
- Not a Windows lane (SkyPilot is Windows-free), so the Postshot worker can
  never ride this layer; that lane remains a licensed-tool exception.

## Promotion

`skypilot_promotion_gate.evaluate_skypilot_promotion` requires recorded
evidence for all 11 gates (image digest/launch constraints, hourly price
cap, readiness marker, interruption cleanup, ambiguous-create
reconciliation, exact provider-native instance identity, target-scoped and
global inventory, pending_teardown closure, provider-API teardown proof and
provider-zero, orphan recovery after SkyPilot state loss, warm-worker
latency non-regression). A full pass authorizes wider *disposable/batch*
ownership only; each further lane needs its own evidence, and the current
custom path is retained as rollback throughout.

Tests: `tests/test_skypilot_provisioner.py`, `tests/test_provider_transport.py`.
