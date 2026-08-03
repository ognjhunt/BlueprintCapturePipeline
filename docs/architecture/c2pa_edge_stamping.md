# C2PA edge stamping (sidecar-only) — c2pa_edge_stamping.v1

Scope: customer-facing media in the Post-Training Data Package export.
Implemented by `src/blueprint_pipeline/c2pa_stamping.py`, wired into
`post_training_data_package.py` immediately after the export manifest is
assembled and before the buyer readout, final checksums (`...checksums.v2`),
archive manifest, and package root signature are produced — so sidecars are
covered by every final integrity artifact.

## Authority model

- The internal Blueprint ledger is authoritative for rights, capture truth,
  evaluation meaning, and claim eligibility. C2PA is an interoperability
  wrapper proving what Blueprint asserted about the exported bytes; it never
  replaces or reorders internal evidence (`internal_ledger_authoritative:
  true` is pinned into every record and assertion).
- The custom assertion (`com.blueprint.ledger_ref`, payload schema
  `blueprint.c2pa_ledger_ref.v1`) carries digests and identifiers only:
  artifact sha256, consent-evidence digest, signed-chain manifest digest,
  holdout split digest, scene/capture ids, optional verification URI. Free
  text (e.g. consent scope wording) is refused at build time
  (`C2paAssertionContentError`) — the rights record itself never leaves the
  ledger.

## Why sidecars, not embedded manifests

The package video store is content-addressed: each clip lives at
`exports/video_bundle/objects/sha256/<aa>/<sha256><ext>` where the filename
is the digest of the bytes, with collision guards, and per-clip digests are
already recorded in `clips_manifest.json`, clip metadata sidecars, and the
lerobot/GR00T `video_files[].sha256` entries before stamping runs. Embedding
a manifest would change the bytes and invalidate all of those digests, so
asset bytes are never modified: the manifest is written as an adjacent
`<stem>.c2pa` sidecar (spec-conformant extension replacement), and the module
verifies byte-identity of the tool's output against the original before
installing anything (`c2pa_output_asset_bytes_changed` fails that file
closed).

## Fail-closed behavior

- Default off. `BLUEPRINT_PTDP_C2PA_STAMPING=1` enables it.
- Tooling is an external pinned binary (`BLUEPRINT_PTDP_C2PATOOL_BIN`) with
  server-side signing material (`BLUEPRINT_PTDP_C2PA_SIGN_CERT_FILE`,
  `BLUEPRINT_PTDP_C2PA_SIGN_KEY_FILE`). Any gap → status `unavailable` with
  explicit blockers; the export proceeds unstamped and the absence is
  recorded in `manifest["c2pa_edge_stamping"]`, the buyer readout
  (`sections.media_provenance.c2pa_edge_stamping`), and
  `exports/provenance/c2pa_stamping_record.json`.
- A file is claimed `stamped` only after (a) the tool exits 0, (b) the output
  asset is byte-identical to the original, and (c) a read-back verification
  reports the `com.blueprint.ledger_ref` assertion. Anything less is
  `failed`/`partial` — never a silent success.
- Formats without a C2PA binding (`.avi`, `.mkv`, `.webm`) are recorded as
  `unsupported_format`, not skipped silently.
- Stamping can never block an export: the packaging hook downgrades any
  exception to `status: failed` inside the manifest summary.

## 3D assets

C2PA has no format bindings for glTF/USD/PLY/SPZ. `stamp_3d_asset_sidecar`
is fail-closed (`blocked_experimental_not_qualified`) even when
`BLUEPRINT_PTDP_C2PA_3D_EXPERIMENTAL=1`, until a prototype proves: exact
byte-stream data-hash binding, tamper detection on renamed/modified/
substituted assets, and validator sidecar discovery — all against committed
fixtures. Blueprint's own checksum/provenance sidecars remain the 3D truth
regardless.

## Operational notes

- Signing material handling follows the package root-signature precedent
  (`BLUEPRINT_PTDP_SIGNING_KEY_FILE`): paths via env, never logged; the
  manifest definition (which references key paths) is written to a 0700
  temp dir and deleted after the run.
- Trust-listed certificates (SSL.com, DigiCert) make manifests validate as
  recognized; self-PKI validates with an "unrecognized signer" note. That is
  a procurement decision, not a code path.
- `c2patool` is not a Python dependency and is absent from the runtime
  license policy SBOM; the pinned binary and version are recorded in
  `docs/architecture/isolated-component-license-inventory.md` before first
  production execution, per the build-on-top adoption mechanics.

Tests: `tests/test_c2pa_stamping.py` (hermetic; fake runner, no c2patool
required).
