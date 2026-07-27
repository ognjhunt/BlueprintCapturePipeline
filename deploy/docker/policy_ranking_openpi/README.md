# OpenPI policy-ranking worker

This is a one-shot, simulator-only worker for the frozen four-policy OpenPI
DROID joint-position cohort. The image pins OpenPI and MuJoCo Menagerie source,
but deliberately excludes all policy checkpoints and InteriorGS-derived pixels.
Those inputs must be supplied privately at runtime and verified against the
frozen inventory before inference.

The worker produces learned-policy MuJoCo episode artifacts and either a
prospective captured-site ordering or an abstention. It does not expose a robot
endpoint, run a WAM, operate hardware, or establish site-specific physical
success.
