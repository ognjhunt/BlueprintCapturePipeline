# CI Required Checks (BlueprintCapturePipeline)

Remediation for audit finding **R044** — the slow/integration/GPU test lane
never gated a merge, so robot-eval / provider / runtime paths could regress
unguarded.

## Lanes

| Lane | Workflow | Trigger | Selection | Purpose |
| --- | --- | --- | --- | --- |
| Fast lane | `.github/workflows/ci.yml` (`test`) | push + `pull_request` | `pytest -q` → `-m "not slow and not gpu"` (pyproject addopts) | Hermetic pre-merge gate for the fast tests. |
| Slow lane gate | `.github/workflows/full-test-lane.yml` (`pr-slow-lane`) | `pull_request` → `main` | `pytest -m "slow or gpu"` | **New (R044).** Runs the exact complement of the fast lane so the heavy subprocess/Isaac/render/module-entrypoint + robot-eval/provider/runtime tests gate PRs to `main`. Bounded to 75 min. |
| Full sweep | `.github/workflows/full-test-lane.yml` (`full-pytest`) | `schedule` (weekly) + `workflow_dispatch` | `scripts/pytest_full.sh` (`-m ""`) | Unbounded whole-suite sweep; not a per-PR gate. |
| Sim-only local gate | `.github/workflows/sim-only-local-gate.yml` | `pull_request` (path-filtered) + push | sim-only local gate script | Cross-repo sim-only regression gate. |

`slow or gpu` is the precise complement of the fast-lane deselection
`not slow and not gpu`, so the two PR lanes together run every collected test
without overlap. The suite currently has 69 files tagged `@pytest.mark.slow`
and no `@pytest.mark.gpu` tests, so the slow lane runs entirely on the hosted
CPU runner today. If GPU-only tests are added later, they need a self-hosted
GPU runner (a `runs-on` change) — the weekly `full-pytest` sweep and local
`scripts/pytest_full.sh` runs continue to collect them.

## What config now enforces

- `pr-slow-lane` runs automatically on every PR targeting `main`. It is **not**
  path-filtered, so any source change is covered.
- A red slow lane produces a failing check on the PR.

## Required human/dashboard step (branch protection)

GitHub branch protection is a repository setting, not a file in the repo, so it
must be set once in the GitHub UI (or via `gh`/API by an admin):

1. GitHub → repo **Settings → Branches → Branch protection rules → `main`**.
2. Enable **Require status checks to pass before merging**.
3. Add these checks as required:
   - `Slow lane gate (PR to main)` (job `pr-slow-lane`)
   - the existing fast-lane `test` job from `CI`
4. (Recommended) Enable **Require branches to be up to date before merging**.

Equivalent `gh` call (admin token required):

```bash
gh api -X PUT repos/ognjhunt/BlueprintCapturePipeline/branches/main/protection \
  -H "Accept: application/vnd.github+json" \
  -f 'required_status_checks[strict]=true' \
  -f 'required_status_checks[contexts][]=Slow lane gate (PR to main)' \
  -f 'required_status_checks[contexts][]=test' \
  -F 'enforce_admins=true' \
  -F 'required_pull_request_reviews[required_approving_review_count]=1' \
  -F 'restrictions=null'
```

Until that setting is applied, the slow lane **runs** on every PR but GitHub
will not **block** merge on its failure.
