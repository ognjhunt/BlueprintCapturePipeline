# Agent Guide For `scripts/`

Scripts are operational entrypoints. Classify a script before running it:
read-only check, local artifact writer, external provider/API caller, GPU runner,
cross-repo gate, or live deploy.

Two programs are active: the free beta's evaluated, recommended pilot and Arm
Decision Proof v1. Do not create or extend a script unless it removes a named
beta journey blocker or ADP gate blocker. Legacy scripts remain
compatibility entrypoints and do not authorize their former lanes.

Safe checks include focused gate/test commands. Treat `setup_environment.py --check`
as a legacy GPU/runtime probe, not a current pipeline readiness check.
Do not run live provider jobs, GPU deployments, Stripe, WebApp mutation, external
sync, Terraform, or deploy scripts without explicit approval.

After any script that writes `output/` or staged capture artifacts, inspect
`git status --short --branch` and `git diff --stat`.
