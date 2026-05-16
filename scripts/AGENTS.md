# Agent Guide For `scripts/`

Scripts are operational entrypoints. Classify a script before running it:
read-only check, local artifact writer, external provider/API caller, GPU runner,
cross-repo gate, or live deploy.

Safe checks include `setup_environment.py --check` and focused gate/test commands.
Do not run live provider jobs, GPU deployments, Stripe, WebApp mutation, external
sync, Terraform, or deploy scripts without explicit approval.

After any script that writes `output/` or staged capture artifacts, inspect
`git status --short --branch` and `git diff --stat`.
