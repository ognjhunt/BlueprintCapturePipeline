Minimal warehouse fixture for industrial-first launch gates.

This fixture is intentionally small and local. It proves the contract path can
read warehouse capture truth, select an industrial target, configure the
industrial containment proxy, and preserve proof boundaries. It is not a
simulator run, WAM success proof, physical contact proof, or deployment approval.

The `pipeline/` subtree is source fixture input, not generated run state. It is
explicitly unignored in the repository so clean checkouts and `git archive`
contain the task-anchor and camera-calibration contracts used by the test lane.
