# FABLE-007: Branch protection and signed release attestation plan (2026-07-11)

Status: implemented, with hosted same-SHA execution pending publication of the
patch. On 2026-07-11 the authenticated repository admin enabled classic branch
protection on `main`: strict required checks, admin enforcement, one approval,
stale-review dismissal, last-push approval, conversation resolution, linear
history, and force-push/deletion denial. The exact protection JSON was read back
from the GitHub API. The OIDC release-attestation workflow and retention-side
`gh attestation verify` path are now present in this patch.

Claim boundary: completing this plan proves repository governance and release
evidence integrity. It does not prove task success, simulator fidelity, or
physical readiness, and it must never be used to upgrade any
`local_contracts_advanced_live_end_to_end_task_success_not_proven` status.

## Part 1 — Branch protection / ruleset for `main`

The audit snapshot found no protection. The implemented control uses GitHub's
classic branch-protection API because it exposes the required enforcement on
this repository. A future migration to a ruleset is optional if it preserves
or strengthens every setting below.

### Required settings

| Setting | Value | Why |
|---|---|---|
| Target | branch ruleset, include `~DEFAULT_BRANCH` (`main`) | one authoritative rule |
| Enforcement | Active | evaluate-only mode proves nothing |
| Restrict deletions | on | no branch deletion |
| Block force pushes | on | append-only history |
| Require a pull request before merging | on | no direct push, including admins |
| Required approving reviews | 1 (this is a solo-owner repo; raise when a second maintainer exists) | independent eyes where possible |
| Dismiss stale approvals on new commits | on | approval binds to the reviewed SHA |
| Require conversation resolution before merging | on | review threads cannot be silently dropped |
| Require status checks to pass | on, strict (`Require branches to be up to date`) | red checks cannot merge; results bind to the merged tree, not a stale base |
| Required status checks (exact check names) | `test`, `Ruff and claim lint`, `Typed release contract core`, `Bandit high and reviewed-medium gate`, `Module and critical-lane governance`, `SBOM, license, and provenance contract`, `dependency-security`, `Container production contract`, `Full pytest lane on CPU runner`, `Python security analysis` (CodeQL), `Regenerate sim-only local gate artifact`, `Python 3.10 compatibility`, `Python 3.11 compatibility`, `Python 3.12 compatibility` | the CI job set that was green on `df030e45` plus the Bandit gate fixed by this change set |
| Require signed commits | on (admin must enroll a signing key first) | commit authorship evidence for the attestation chain |
| Bypass list | empty by default | see break-glass below |

Note on check names: GitHub matches required checks by check-run name (the job
`name:`, or the job id when no `name:` is set — `test` and
`dependency-security` have no display name in `.github/workflows/ci.yml`).
If job names change, the ruleset must be updated in the same PR.

### Time-bounded break-glass procedure

1. Break-glass is a deliberate, logged exception — never a standing bypass.
2. To invoke: the admin adds the `Repository admin` role to the ruleset bypass
   list, records in a tracked issue BEFORE merging: the reason, the exact SHA,
   the checks being bypassed, the independent approver (a second person, or for
   a solo repo a written justification posted publicly in the issue), and an
   expiry timestamp at most 24 hours ahead.
3. The bypass MUST be reverted within that window; the reverting settings
   change is itself audit-logged by GitHub.
4. Post-hoc: the skipped checks must be re-run on the merged SHA and their
   results attached to the issue; the quality ledger row for the affected
   release is `open` until they pass.
5. Negative tests (acceptance): with the ruleset active, prove that
   (a) a direct `git push origin main` is rejected,
   (b) a force push is rejected,
   (c) a PR with a failing required check exposes no enabled merge button and
   `gh pr merge` fails,
   (d) an unresolved review thread blocks merge.
   Record the four rejection outputs as evidence artifacts.

### Historical ruleset alternative (not the active control)

```bash
# 1. Create the ruleset (GitHub REST: rulesets API).
gh api repos/ognjhunt/BlueprintCapturePipeline/rulesets \
  --method POST \
  --input - <<'JSON'
{
  "name": "main-release-integrity",
  "target": "branch",
  "enforcement": "active",
  "conditions": {"ref_name": {"include": ["~DEFAULT_BRANCH"], "exclude": []}},
  "rules": [
    {"type": "deletion"},
    {"type": "non_fast_forward"},
    {"type": "required_signatures"},
    {
      "type": "pull_request",
      "parameters": {
        "required_approving_review_count": 1,
        "dismiss_stale_reviews_on_push": true,
        "require_code_owner_review": false,
        "require_last_push_approval": false,
        "required_review_thread_resolution": true,
        "automatic_copilot_code_review_enabled": false,
        "allowed_merge_methods": ["squash", "merge"]
      }
    },
    {
      "type": "required_status_checks",
      "parameters": {
        "strict_required_status_checks_policy": true,
        "do_not_enforce_on_create": false,
        "required_status_checks": [
          {"context": "test"},
          {"context": "Ruff and claim lint"},
          {"context": "Typed release contract core"},
          {"context": "Bandit high and reviewed-medium gate"},
          {"context": "Module and critical-lane governance"},
          {"context": "SBOM, license, and provenance contract"},
          {"context": "dependency-security"},
          {"context": "Container production contract"},
          {"context": "Full pytest lane on CPU runner"},
          {"context": "Python security analysis"},
          {"context": "Regenerate sim-only local gate artifact"},
          {"context": "Python 3.10 compatibility"},
          {"context": "Python 3.11 compatibility"},
          {"context": "Python 3.12 compatibility"}
        ]
      }
    }
  ],
  "bypass_actors": []
}
JSON

# 2. Verify the ruleset is active and applies to main.
gh api repos/ognjhunt/BlueprintCapturePipeline/rulesets --jq '.[] | {id, name, enforcement}'
gh api 'repos/ognjhunt/BlueprintCapturePipeline/rules/branches/main' \
  --jq '.[].type'

# 3. Negative test: direct push must be rejected.
git push origin HEAD:main   # expected: rejected by ruleset

# 4. Break-glass (time-bounded; see procedure above), then revert:
gh api repos/ognjhunt/BlueprintCapturePipeline/rulesets/<RULESET_ID> \
  --method PUT --input updated-ruleset-with-bypass.json
# ... merge, then restore bypass_actors: [] within 24h with the same PUT.
```

## Part 2 — External signed command/release attestation

### Problem

The 107-row quality ledger
(`docs/public_launch_sc3_quality_gap_ledger_2026-07-09.json`) is repository
content: any claim it makes about "CI passed" or "artifact hash X" can be
edited in the same commit that it purports to attest. A trustworthy design
must anchor evidence OUTSIDE the commit it describes (non-circular).

### Design

1. Evidence producer: `.github/workflows/release-signature-verification.yml`,
   dispatched only with successful same-SHA CI, Full Test Lane, and CodeQL run
   IDs plus an immutable image digest and release ID.
2. Attestation subject (what gets signed), one JSON document per release
   candidate:
   - `attested_commit`: `${GITHUB_SHA}` (the merged main SHA),
   - `repository` and `ref`,
   - `workflow_run_ids`: the run IDs and run attempt numbers of CI, Full Test
     Lane, CodeQL, Sim-Only Local Gate, and Python Compatibility runs on that
     exact SHA (queried via the Checks API inside the job, not passed in),
   - `artifact_hashes`: SHA-256 of every retained evidence artifact
     (`bandit.json`, `bandit-policy-gate.json`, `cpu_full.json`, junit XMLs,
     SBOM, container evidence, signature evidence) as downloaded from the
     workflow-run artifact store,
   - `release_id`: the GitHub release/tag being cut (when releasing),
   - `ledger_digest`: SHA-256 of the committed ledger file at that SHA.
3. Signing: `actions/attest-build-provenance` pinned to
   `977bb373ede98d70efdf65b84cb5f73e068dcc2a` uses GitHub OIDC/Sigstore. The signature binds the payload to the
   workflow identity (`repo`, `workflow_ref`, `run_id`) so it cannot be
   reproduced by a laptop or by a different repository.
4. Storage: the signed attestation goes to (a) the GitHub attestation store
   (`gh attestation verify` retrievable) and (b) the immutable evidence bucket
   already used by `scripts/archive_release_evidence.py` (object-lock
   retention). It is NEVER committed back into the attested commit; a LATER
   commit may record its pointer (attestation ID + digest), which is a
   reference, not the evidence.
5. Ledger recompute remains deliberately fail-closed: local digest rebinding
   does not close rows. A future `recompute_ledger_from_attestations.py` may
   - fetches attestations for the current SHA,
   - verifies signatures and identity claims (`gh attestation verify
     --repo ognjhunt/BlueprintCapturePipeline`),
   - marks a ledger row `closed` ONLY when its required evidence group appears
     in a verified attestation bound to the current release,
   - fails closed with `ledger_row_evidence_unattested:<row_id>` otherwise.
   Local test results can move a row to `partial` at most; `closed` requires
   the external verified attestation. This preserves the audit's rule: rows do
   not close merely because local tests pass.

### Non-circularity argument

- The attestation payload is produced from hosted-run outputs and signed with
  an identity the repository contents cannot forge.
- The attested commit cannot contain its own attestation (it is created after
  the commit exists and stored externally).
- Verification therefore needs: the commit, the external attestation store,
  and the Sigstore trust root — tampering with the repository alone cannot
  fabricate a passing release.

### Remaining external execution

1. Publish this patch through a PR and obtain every newly required same-head
   check.
2. Dispatch `Release Signature Verification` for that SHA/release and verify
   the generated bundle.
3. Provision the immutable evidence bucket credentials as an environment-scoped
   secret (never a repo-wide secret).
4. Decide the release tag convention (`release/vYYYY.MM.DD-N`) so `release_id`
   is unambiguous.

### Acceptance (from the audit)

- Negative branch tests prove red checks cannot merge and direct/force push is
  denied.
- The ledger command/release attestation verifies cryptographically
  (`gh attestation verify`) and is bound to the current commit and release
  artifacts.
- No ledger row is `closed` without a verified attestation covering its
  evidence group.
