# Runtime dependency security repair — 2026-10-05

The dependency gate for main commit
`f4f34676d1377cb64c60787dd933c7ef5dd8843e` in
[Actions run 37360571840](https://github.com/ognjhunt/BlueprintCapturePipeline/actions/runs/37360571840)
reported 19 known advisories across three runtime packages. The supply-chain
job also refused two unreviewed package versions and one obsolete review.
No matching active dependency repair PR was found when this isolated repair
started.

This is an ADP-009D day-14 rehearsal delivery-safety dependency, not a new
program or proof claim. The existing frozen graph could not pass the security
gate. The smallest change is three targeted lock updates, their two canonical
exports, and measured continuity evidence for previously reviewed license terms.
No runtime configuration, dependency constraints, gate implementation, advisory
waiver, model, provider, credential, paid resource, or deployment was changed.

## Exact changes

| Package | Previous | Replacement | Reason |
| --- | --- | --- | --- |
| anyio | 4.14.1 | 4.14.2 | Fixes the three reported AnyIO advisories. |
| urllib3 | 2.7.0 | 2.8.0 | Fixes the three reported urllib3 advisories. |
| PyJWT | 2.13.0 | 2.15.0 | 2.14.0 fixes most reported JWT issues; 2.15.0 is required by PYSEC-2026-4141. |

`uv==0.10.7 lock --upgrade-package anyio==4.14.2 --upgrade-package
urllib3==2.8.0 --upgrade-package pyjwt==2.15.0` changed only these package
versions. `scripts/verify_dependency_exports.py --write` regenerated both
`requirements.txt` and `requirements-geometry.txt` without other graph changes.

The old PyJWT advisory `PYSEC-2026-4146` supplies no `fix_versions`. This was
not waived or assumed fixed from its empty annotation. Version-specific PyPI
metadata and the actual audit report no known advisories for 2.15.0. A direct
regression reproduces caller-options mutation on installed 2.13.0 and passes
on 2.15.0. A second regression reproduces silently accepted malformed signature
bytes on 2.13.0 and verifies rejection on 2.15.0.

## License continuity and remaining release blockers

The previous and current published wheel bytes were downloaded from the exact
PyPI metadata URLs and checked against the published SHA-256 digests. Their
embedded license files are byte-identical for all three updated packages and
for `google-auth` 2.55.2 → the already locked 2.58.0. The existing MIT/Apache
license policy is unchanged from the base commit: identical license bytes do
not transfer an owner approval to a new exact name/version. No new human review,
inherited approval, or legal acceptance is asserted. `license-continuity.json` records both exact
wheel identities and the comparison; copies of the measured license bytes are
retained here.

**The supply-chain gate remains blocked for five missing exact-version reviews:**
`anyio==4.14.2`, `urllib3==2.8.0`, `pyjwt==2.15.0`, the already locked
`google-auth==2.58.0`, and `meta-sam-parser==0.0.5`. The policy also retains
three obsolete exact-version records (`anyio==4.14.1`, `pyjwt==2.13.0`, and
`google-auth==2.55.2`), which the gate reports as orphaned. Those historical
records were not rewritten to manufacture current approval. The unchanged
permissive-license bytes are evidence for the required owner review, not a
replacement for it.

The `meta-sam-parser==0.0.5`
embedded custom SAM License has SHA-256
`4dea99bfaa016e21bc860d73f344236bd1e5c4977d1a9a8fd32f822b500ae1be`, matching
the terms documented in
`docs/research/semantic_splat_object_index_decision_2026-07-30.md`. That document
explicitly requires an exact use authorization. This repair found no package
approval or authorization that permits marking this runtime component reviewed;
it adds neither an approval nor an exception. Authorized acceptance for this
exact use, or an independently reviewed implementation/dependency change, is
still required before the supply-chain gate can pass.

## Verification and evidence

- Unmodified `scripts/run_dependency_security_gate.py`, using pinned
  `pip-audit==2.10.1`: **passed, 86 runtime dependencies audited, zero known
  advisories**, compared with the retained CI result of 19.
- Both frozen compatibility exports and `uv lock --check`: passed.
- Focused auth/transport/dependency-license tests: **19 passed**. These exercise
  JWT verification behavior, bounded urllib3 streaming and truncation,
  AnyIO cancellation, existing Cloud Run IAM/header behavior, provider retry
  contracts, and exact-version supply-chain approval enforcement.
- Regression control using the original installed packages: the two new JWT
  regressions fail as expected; the streaming/cancellation checks pass.
- Wheel and source distribution built; distribution metadata verification passed.
- Actual CycloneDX/SPDX/provenance builder: 95 components, blocked by the five
  missing exact-version reviews and three orphaned reviews listed above.
  No signature or deployed-image claim is made.
- Changed-file Ruff and `git diff --check`: passed.

The retained security report was generated from the changed worktree based on
the main SHA above, before committing; its `repository_sha` is that base SHA.
The supply-chain report was regenerated while correcting the approval records
on candidate `310c0373042f41c3a7e4da70a7da54fe79286ea6`; that is its recorded
source SHA, not a claim that the pre-correction candidate preserved authority.
The candidate lock is bound by SHA-256
`7d177694b52ec7a3d3567d3fd2d7fdca524f967c41bed10b820c535796567eb4`.
They are candidate evidence, not proof that the original main commit was green.

The local gate used an isolated tool environment with `UV_NO_SYNC=1` and
`UV_PROJECT_ENVIRONMENT` pointing to that environment, containing the exact
required audit tool. It audited the unmodified frozen runtime export, not the
tool environment. This avoids a full runtime installation for a metadata audit.
Compatibility tests used the three patched packages ahead of the existing
checkout's installed test dependencies through a local `.pth` entry; no shared
environment was modified. Cache paths stayed inside the isolated worktree.
No provider or production action was performed.
