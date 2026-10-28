# Intake nonce retention

AUT-05 keeps the existing per-client, immutable nonce claim files and adds one
process-shared lock for both claims and reclamation. Fresh/replayed claims keep
their existing rejection semantics across processes and restart. Expiry
reclamation requires the recorded expiry plus a 60-second skew/race grace.

Unknown, malformed, nonfinite, symlinked, nonregular, multiply linked, unsafe-mode
or unreadable claim state is retained. It is counted as unproven and cannot
become a new accepted claim. A failed write/fsync fails authenticated intake
closed with a sanitized 503; it never reports success or silently drops partial
anti-replay state. A persistent clock high-water mark prevents a forward jump
followed by substantial clock rollback from reopening reclaimed nonces.

Normal signed intake invokes a bounded maintenance observation at most once per
minute. Its default mode is `report_only`. The optional
`BLUEPRINT_INTAKE_NONCE_CLEANUP_ENABLED` policy enables actual expiry
reclamation; this change does not enable it or perform production deletion.
Existing per-key expiry reuse also follows the common lock and grace.
All nonce claim and reclamation processes must be upgraded to the common-lock
implementation before cleanup is enabled; an older uncoordinated expired-key
writer cannot participate in the new race guarantee.

Each pass inspects at most 64 canonical claim records in the intake path (the
standalone helper supports a cap up to 1,000), with a persistent round-robin
cursor. Filename discovery still enumerates the existing flat directory. The
inspection budget bounds content reads, validation and deletion, rather than
the total cost of enumerating an arbitrarily large historical directory. That
remaining scaling boundary must be measured before claiming an unbounded-volume
maintenance guarantee; partitioning/indexing is appropriate if actual volume
requires it.

The private `.nonce_retention_report` records the inspected byte/record window,
fresh/expired/unproven counts, observed free bytes/inodes, truncation, applied
reclamation and policy mode. It is not a claim that all root bytes or all nonce
records were measured. Existing `/health` exposes only these safe bounded
counts; no nonce, client identity, path or secret value is returned.

Local acceptance covers two process claimants racing a sweeper, restart and
clock rollback, malformed/symlink/permission state, injected full-disk fsync
failure, incremental expiry, and a bounded-rate traffic soak stabilizing within
the TTL, grace and inspection-lag envelope. Production starts with observation
only and still requires measured-volume retention evidence before cleanup is
enabled by the existing policy owner.
