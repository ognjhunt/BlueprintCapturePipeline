# Optuna native-construction search dependency

- Package: `optuna`
- Version: `4.9.0` (exactly pinned in `pyproject.toml` and `uv.lock`)
- Upstream: <https://github.com/optuna/optuna/tree/v4.9.0>
- Distribution: <https://pypi.org/project/optuna/4.9.0/>
- License: MIT
- Storage API: `JournalStorage(JournalFileBackend)`

Blueprint uses Optuna only as a durable ask/tell experiment ledger for bounded
native-construction recovery. It does not author robot poses, trajectories,
resets, camera configurations, acceptance gates, thresholds, or verdicts. Each
trial is fixed to one exact member of a deterministic, self-digested candidate
inventory. Native Isaac evidence remains the construction grader.

The file journal is mutable operational state. Immutable Blueprint inventory and
attempt receipts self-digest the exact study attribute or frozen trial snapshot
and can be reopened against the journal after process restart. The deterministic
sampler seed is derived from the run ID unless an explicit seed is supplied and
sealed into the study and every receipt.
