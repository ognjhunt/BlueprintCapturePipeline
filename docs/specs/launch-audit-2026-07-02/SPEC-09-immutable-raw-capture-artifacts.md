# SPEC-09: Immutable raw capture artifacts (provenance)

- Status: Proposed
- Priority: **P1 — major**
- Area: `src/blueprint_pipeline/geometry_stage.py`, `frame_alignment_stage.py`
- Doctrine: "raw capture … and provenance are authoritative"; "downstream outputs must not rewrite capture truth"

## Problem

Two downstream stages mutate canonical capture files in place, destroying the original
values they claim are authoritative:

1. `_patch_descriptor_with_geometry` (`geometry_stage.py:871-986`) rewrites
   `context.descriptor_path` — the raw `capture_descriptor.json` — overwriting
   `geometry_source`, `geometry_ready`, `quality{}`, and `metadata{}` (write at `:986`).
2. `frame_alignment_stage.py:589-614` rewrites `site_reference_index.jsonl` in place
   (`:613`), overwriting/adding `site_frame_transform` / `T_site_camera` on raw records.

Manifests simultaneously assert `raw_capture_authoritative: True`
(`geometry_stage.py:384`). After a run, the pre-run descriptor/index state is
unrecoverable: re-derivation, audit, and dispute resolution against capture truth are
impossible, and a buggy stage can permanently corrupt the record of what was captured.

This compounds SPEC-01's finding that a provenance flag (`fallback_used`) is resettable
by a later stage (`geometry_stage.py:761`).

## Why this matters for launch

Provenance is a first-class sellable property (rights, licensing, disputes, trust). An
in-place-mutated descriptor means we cannot prove to a buyer — or ourselves — what the
device originally reported vs what the pipeline derived. This breaks the platform truth
hierarchy at its root and gets harder to fix after real buyer data exists.

## Proposed fix

1. **Raw intake artifacts become write-once.** After intake completes, the pipeline
   treats `capture_descriptor.json`, `frames.jsonl`, `poses.jsonl`,
   `site_reference_index.jsonl` (raw form), and device metadata as read-only. Enforce
   with a guard in the stage runner (hash the raw set at intake; verify at stage
   boundaries; a hash change fails the run).
2. **Derived state goes to derived files.** Geometry writes
   `derived/geometry_descriptor_patch.json` (or extends `geometry_summary.json`);
   alignment writes `derived/site_alignment.jsonl`. Consumers read raw + patch via an
   accessor that overlays them explicitly (`load_effective_descriptor()`), keeping both
   layers visible.
3. **Append-only provenance flags:** truth flags (`fallback_used`, `synthetic_*`,
   `estimated_*`) can only be added or strengthened; the accessor rejects true→false
   transitions (shared enforcement with SPEC-01 item 3).
4. Migration: a one-time script snapshots current mutated descriptors as
   `descriptor_as_found.json` so existing bundles keep a defensible baseline.

## Acceptance criteria

- [ ] A stage attempting to write any raw intake file fails the run with a provenance violation.
- [ ] Geometry/alignment outputs land in derived files; e2e passes using the overlay accessor.
- [ ] Intake hash manifest is recorded per bundle and verified at each stage boundary.
- [ ] Regression test: provenance flags cannot weaken across stages.
