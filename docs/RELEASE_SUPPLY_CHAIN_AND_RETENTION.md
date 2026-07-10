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
audit archive. `Release Evidence Retention / Immutable BASE release evidence
archive` downloads the exact-SHA CI/full-test artifacts, rejects red or
different-SHA CI, full-test, CodeQL, or signature-verification runs, requires a
release commit that is already an ancestor of `origin/main`, requires a
release-bound `artifact_signature.json`, creates a checksum-bearing
`git archive` of the exact commit, builds a checksum manifest, uploads
with Object Lock, and
requires version/checksum/retention metadata plus a full version-bound GET
digest readback before writing
`immutable-archive-receipt.json`. A second output,
`immutable_retention.json`, is the exact-SHA/image-digest envelope consumed by
the release evidence graph; it preserves the receipt status and cannot turn a
blocked archive into accepted evidence.

CodeQL retention includes both exact-workflow run metadata and every CodeQL
SARIF analysis returned for the exact release commit. A green run ID without an
exact-commit analysis payload is insufficient.

Before upload, the archiver independently reopens the tarball, requires its
embedded manifest to equal the sidecar release binding, and re-hashes every
regular entry. Relabeling a bundle to another commit/image or changing an
internal entry therefore blocks before any provider call.

Required repository secrets are deliberately absent from source:

- `BLUEPRINT_RELEASE_EVIDENCE_S3_URI`
- `BLUEPRINT_RELEASE_EVIDENCE_AWS_ACCESS_KEY_ID`
- `BLUEPRINT_RELEASE_EVIDENCE_AWS_SECRET_ACCESS_KEY`
- optional session token and region

The AWS archive credentials are scoped only to the Object Lock upload/readback
step. Checkout, artifact download, dependency install, evidence validation, and
bundle assembly do not receive them.

The manual archive run also requires exact-SHA CI, Full Test Lane, CodeQL, and
external signature-verification run IDs. The signature run must come from the
dedicated `Release Signature Verification` workflow and expose an artifact
containing `artifact_signature.json` with the release-evidence
envelope schema, `verified` status, and exact repository SHA/image digest. A
fresh timezone-aware validity interval, evidence URI, and raw verification
proof matching `source_artifact_digest` are also mandatory. A different
workflow, red run, wrong SHA, wrong digest, missing proof, or stale artifact
blocks before upload. `scripts/validate_release_signature_evidence.py` enforces
that contract without claiming a fresh live-registry readback.

The bucket must already have versioning and Object Lock enabled. The workflow
does not create or weaken bucket policy. A missing credential, version ID,
checksum, full-object restore readback, `COMPLIANCE` lock, or sufficiently long
retention date blocks.

For PTDP, SC3, paid, or live scopes, assemble the additional groups declared in
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
unexecuted archive workflow therefore cannot pass a release scope.
