# Retained scene accounting after evidence offload

ADP-009D, day-28 public-scene rehearsal. InteriorGS 840938's source preparation
finished, but the coordinator and spend refresher refused
`unstarted_controls_evidence_unsafe`: cold-evidence GC removed launch directories
whose `launch_receipt.json` was still required by sealed attempt settlements.
The failed handoff was an orchestration failure; CAD had not begun.

The smallest repair preserves the exact launch receipt bytes in the sealed
archive pointer. The accounting reader uses those bytes only when the original
file is absent, checks the pointer identity and seal and the receipt's original
archive member hash/size, and retains the original path in its reference.
Existing settlement seals, launch identity/status checks, spend holds and
attempt counts remain unchanged. A present corrupt file is never hidden by
archived evidence. Reads make no network or provider calls. Receipts above
64 KiB fail closed and keep the original run, bounding pointer growth. Legacy
pointers lacking the embedded receipt still require explicit digest-verified
restoration; no receipt is synthesized from a status or timestamp.

Restoration adopts the destination parent's UID/GID for every extracted file
and directory before publishing the tree. Ownership failure keeps the target
unpublished. Symlink trees are refused.

Verification: focused evidence-offload and real settlement fixture tests prove
all holds remain identical after offload, including $21.26 of unreconciled
source/configuration caps. Corrupt seals, altered bytes, missing members,
foreign run identity, symlinks, and ownership failures refuse. Scene-spend and
unstarted-controls tests cover compatibility with the consuming accounting
paths. This is a storage/accounting repair; it does not prove CAD execution,
policy success, billing closure, deployment, or physical qualification.

A temporary canonical storage pin protects 21 exact launch roots referenced by
this scene's records: `preparation/scene840938-accounting-references-20260915`.
It was created using the actual clock for seven days while the live run
continued, without changing pipeline timers. Release it only after the deployed
reader can validate every archived reference or an appropriate terminal closure.
