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
Clock state uses the same nofollow, nonblocking, regular-file, private-mode and
4,096-byte read bounds as nonce records. A FIFO or malformed clock file fails
closed without holding the shared nonce lock indefinitely.

Normal signed intake invokes a bounded maintenance observation at most once per
minute. Its default mode is `report_only`. The optional
`BLUEPRINT_INTAKE_NONCE_CLEANUP_ENABLED` policy enables actual expiry
reclamation; this change does not enable it or perform production deletion.
Existing per-key expiry reuse also follows the common lock and grace.
All nonce claim and reclamation processes must be upgraded to the common-lock
implementation before cleanup is enabled; an older uncoordinated expired-key
writer cannot participate in the new race guarantee.

Each intake pass examines at most 136 directory entries and reads at most 64
canonical records, each limited to 4,096 bytes. Unknown filenames consume the
same directory-entry budget. The standalone helper caps inspection at 1,000
records and discovery at 2,048 entries. A 50 ms monotonic scan deadline stops the
pass before starting another entry; one entry's bounded validation/reclaim work
already in progress can finish after that deadline. Lock acquisition,
constant-size metadata writes and filesystem latency are not a hard wall-clock
promise for the entire HTTP request. There is no sorted listing or
full-directory enumeration before those limits apply.

An opened nofollow directory FD and its device/inode identity anchor a retained
`scandir` stream. Healthy ticks resume that stream despite new claims. A replaced
root refuses the old stream; record reads and expiry unlinks use its anchored FD
and fresh identity checks. At most four streams are cached per process, and both
stream and FD close on completion, error, eviction or fork. This adds no queue
or alternate nonce authority; existing flat files remain canonical.

The claim/reclaim lock itself is opened relative to a nofollow directory FD held
through the whole operation. Clock reads, metadata temporary-file creation,
atomic replacement, fsync, nonce creation and expiry unlink all use that same
locked directory identity. A root swap before the final state/report commit
cannot redirect writes into the replacement store or lower its clock high-water.
The operation refuses its result if the configured pathname changed during the
commit; it preserves any claim already retained in the original store.

Portable directory streams cannot be durably resumed after process restart.
Restart or cache eviction begins another bounded survey, so frequent restarts
can delay cleanup progress. Canonical nonce claims and the clock high-water
remain durable and preserve replay rejection. Unvisited legacy entries stay
unknown. Neither a partial window nor reaching the end of a changing directory
establishes a complete census.

The private `.nonce_retention_report` records the inspected byte/record window,
fresh/expired/unproven counts, observed free bytes/inodes, truncation, applied
reclamation and policy mode. Discovery counters and the monotonic deadline
describe this window, with `discovery_continuation_scope: process_lifetime` and
`inventory_complete: false`. `canonical_filenames_discovered` counts only this
window. It is not a claim that all root bytes or all nonce records were measured.
Existing `/health` exposes only these safe bounded counts; no nonce, client
identity, path or secret value is returned.

Local acceptance covers two process claimants racing a sweeper, restart and
clock rollback, malformed/symlink/permission state, injected full-disk fsync
failure, incremental expiry, a 10,000-entry legacy inventory under continuing
claims, discovery deadline/entry limits, fair continuation, root replacement,
cursor closure/restart, and a bounded-rate traffic soak stabilizing within the
TTL, grace and inspection-lag envelope. Production starts with observation only
and still requires measured-volume retention evidence before cleanup is enabled
by the existing policy owner.
