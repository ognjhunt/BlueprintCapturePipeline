# Website native startup ownership — 2026-09-21

ADP-009D/day-21 blocker: the controller-origin development construction for
`team-eval-5aea93d1-ab8a-4fe5-995a-48a0a22f90bb` never reached Isaac. The spend
guard journal at 18:17:04 UTC records deleting Vast instance `51947666` after
8m31s as `unbooted_dud_past_boot_ttl`, although its independent watchdog was
live and armed through 18:40 UTC. Construction used `blueprint-native-task-arena-`;
the exact-watchdog owner scanner admits `blueprint-task-evaluation-` jobs.

Construction now uses the registered
`blueprint-task-evaluation-native-arena-construction-` namespace. The owner
scanner and orphan TTL are unchanged. Exact process, directory, provider,
deadline and instance-id checks remain required; dead, cancelled, expired or
mismatched watchdogs confer no protection.

The producer-to-owner regression failed on the old construction prefix. It now
covers construction and runtime preflight, including real watchdog processes
that arm and cancel without any provider allocation. This was a pre-spend
coverage gap: the earlier test covered the preflight producer only, not the
construction producer used by this website run.

The failed run retained its prior assets and placement. Provider absence was
confirmed, watchdog mutations were zero, and estimated runtime cost was
$0.086267 (not an invoice). No robot episode ran. This change does not claim
native completion or automatic replacement after unrelated startup failures;
bounded controller recovery for those failures remains separate work.
