# Release Evidence Graph

This is the authoritative release-evidence contract for
BlueprintCapturePipeline. The machine-readable requirements are in
[`release_evidence_requirements.json`](release_evidence_requirements.json).

## Trust model

An evidence envelope is routing metadata, not proof. A durable-looking URI,
accepted status string, or caller-supplied digest cannot satisfy a graph node.
Every accepted node requires all of the following:

1. A v2 envelope named `<node_id>.json`.
2. A relative `source_artifact_path` beneath `sources/` in the evidence root.
   Absolute paths, `..`, symlinks, missing files, non-JSON objects, and files
   larger than the evaluator limit are rejected.
3. An exact `source_artifact_digest` recomputed from the source bytes.
4. Native source validation: expected schema, node identity, evidence schema,
   source status, repository SHA, image digest, generated/expiry interval, and
   an empty blocker list.
5. Node-specific semantic validation. For example, dependency evidence must
   show a nonzero audited dependency set and zero known vulnerabilities;
   provider evidence must prove execution and teardown; retention evidence must
   prove Object Lock plus restore readback.
6. A trusted Ed25519 verifier attestation. The signature covers the node ID,
   source-byte digest, canonical parsed-claims digest, repository SHA, image
   digest, generation time, and expiry. The public-key fingerprint is pinned by
   authority ID in the requirements policy, outside the operator-controlled
   evidence directory.

This binding prevents URI-only proof, generic-source relabeling, cross-node
replay, source mutation, release rebinding, and self-signed authority injection.
Persisted graphs carry the signed normalized source claims and are
cryptographically revalidated by downstream launch surfaces.

## Envelope and attestation

The envelope schema is `blueprint.release_evidence.v2`:

```json
{
  "schema_version": "blueprint.release_evidence.v2",
  "evidence_id": "dependency_policy",
  "evidence_schema_version": "blueprint_dependency_security_gate.v1",
  "status": "passed",
  "repository_sha": "<40 or 64 lowercase hex>",
  "image_digest": "sha256:<64 lowercase hex>",
  "generated_at": "<timezone-aware timestamp copied from source>",
  "expires_at": "<timezone-aware timestamp copied from source>",
  "source_artifact_path": "sources/dependency_policy.json",
  "source_artifact_digest": "sha256:<digest recomputed from source bytes>",
  "evidence_uri": "gs://durable-release-evidence/dependency_policy.json",
  "source_verifier_attestation": {
    "schema_version": "blueprint.release_evidence_source_attestation.v1",
    "algorithm": "ed25519",
    "authority_id": "release_dependency_policy_v1",
    "public_key_base64": "<raw 32-byte Ed25519 public key>",
    "statement": {
      "schema_version": "blueprint.release_evidence_source_attestation.v1",
      "authority_id": "release_dependency_policy_v1",
      "node_id": "dependency_policy",
      "source_artifact_digest": "sha256:<source-byte digest>",
      "source_claims_digest": "sha256:<canonical parsed-source digest>",
      "repository_sha": "<release SHA>",
      "image_digest": "sha256:<release image digest>",
      "generated_at": "<normalized UTC timestamp>",
      "expires_at": "<normalized UTC timestamp>"
    },
    "signature_base64": "<Ed25519 signature over canonical statement JSON>"
  }
}
```

The envelope copies of status and release bindings must match the source, but
the evaluator always treats the source plus trusted signature as authoritative.

## Full CPU lane

`full_test_lane_ci` has additional requirements. The signed source must carry
canonical GitHub Actions provenance for
`.github/workflows/full-test-lane.yml`, `lane_id=cpu_full`, unrestricted
`pytest -m ''`, and identical nonzero planned, executed, and JUnit test counts
and test-ID digests. Failures, errors, and skips must all be zero. A reduced,
filtered, skipped, or self-described local run cannot qualify.

## Authority bootstrap

The checked-in authority IDs are intentionally fail-closed with
`public_key_sha256: null`. Operations must pin each verifier's real Ed25519
public-key fingerprint through a reviewed policy change after the corresponding
private key is provisioned only in the controlled verifier environment. Until a
node's key is pinned, it returns `authority_unconfigured`; no local test key or
operator-authored key is accepted as production proof.

## Evaluation

```bash
python scripts/build_release_evidence_graph.py \
  --scope PAID \
  --repository-sha "$(git rev-parse HEAD)" \
  --image-digest 'sha256:<64 lowercase hex characters>' \
  --evidence-dir output/release_evidence
```

`SIM` deliberately does not require physical-robot evidence. `SC3`, `PAID`,
and `LIVE` have stronger evidence sets; a green local sim graph must never be
reused as an external-fidelity, paid-launch, or live-runtime claim.
