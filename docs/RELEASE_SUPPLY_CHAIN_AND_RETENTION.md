# Release Supply Chain And Retention

Status: machine-enforced contract. Live signatures, repository security
settings, immutable archive receipts, and deployed-digest readback remain
external evidence until their workflows actually pass for the release SHA.

Every release candidate must produce:

- a frozen `uv.lock`-derived CycloneDX 1.5 SBOM;
- an SPDX 2.3 projection with an exact-version, owner-reviewed license decision;
- an in-toto/SLSA provenance statement bound to distribution digests;
- GitHub keyless distribution-provenance and SPDX SBOM attestations on `main`;
- an image-digest signature and deployed-digest SBOM/signature readback from the
  deployment lane;
- a scope-complete evidence bundle retained in versioned S3 Object Lock
  `COMPLIANCE` mode for at least 2,555 days.

GitHub Actions artifacts are 90-day transport copies. They are not the durable
audit archive. `scripts/archive_release_evidence.py` uploads an exact-SHA
evidence bundle with Object Lock and requires version/checksum/retention
metadata plus a full version-bound GET digest readback before writing its
archive receipt and the `immutable_retention.json` envelope consumed by the
release evidence graph; the envelope preserves the receipt status and cannot
turn a blocked archive into accepted evidence. No workflow in this repository
runs the archiver.

CodeQL retention includes both exact-workflow run metadata and every CodeQL
SARIF analysis returned for the exact release commit. A green run ID without an
exact-commit analysis payload is insufficient.

Before upload, the archiver independently reopens the tarball, requires its
embedded manifest to equal the sidecar release binding, and re-hashes every
regular entry. Relabeling a bundle to another commit/image or changing an
internal entry therefore blocks before any provider call.

`scripts/validate_release_signature_evidence.py` checks an
`artifact_signature.json` envelope: the release-evidence envelope schema,
`verified` status, exact repository SHA/image digest, a fresh timezone-aware
validity interval, evidence URI and raw verification proof matching
`source_artifact_digest`. It does not claim a fresh live-registry readback.

The bucket must already have versioning and Object Lock enabled. The archiver
does not create or weaken bucket policy. A missing credential, version ID,
checksum, full-object restore readback, `COMPLIANCE` lock, or sufficiently long
retention date blocks.

For the legacy `PTDP` evidence-export scope, SC3, paid, or live scopes, assemble the additional groups declared in
`docs/release_evidence_retention_policy.json` with
`scripts/build_release_evidence_bundle.py`; absent provider, restore, Pub/Sub,
native LeRobot, or deployment evidence remains a scope-specific blocker.
SC3 retention includes the same sim-only gate required by the SC3 release
evidence graph; omitting it cannot yield a scope-complete archive.

Supply-chain generation is not a signature claim. An archive receipt is not a
release-correctness claim. Repository vulnerability alerts, Dependabot, CodeQL,
live image signing, and deploy readback must each be verified in GitHub or the
deployment environment before their external gaps can close.

The release evidence graph requires `artifact_signature` and
`immutable_retention` even for `BASE`; unsigned local provenance and an
unexecuted archive therefore cannot pass a release scope.
