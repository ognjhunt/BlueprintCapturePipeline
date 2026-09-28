# Scene 841757 disk and validation repair

Scope: ADP-009B construction and ADP-009D's public-scene rehearsal precursor to
the day-14 protocol freeze. Completion evidence is a retained-input replay,
archive/readback/restore receipts, and resumed production preparation. These
are development-only operational results, not physical or policy qualification.

The R10 preparation refused at 12:19:17 UTC with 7,972,315,136 free bytes against
the unchanged 8 GiB floor. The periodic evidence offloader was running at the
same time and finished at 12:23:16. It had built an approximately 5.2 GB archive
of a September 1 interpretation backfill, then rejected it as changed. Free
space subsequently returned to approximately 13 GB. Archive staging did not
participate in the shared reservation ledger.

That backfill contains 5,020 repeated inode references. Tar encodes these as
hardlink headers with size zero; the old evidence manifest incorrectly recorded
the header's size and empty digest rather than the restored file's bytes. The
independent pre-eviction readback correctly rejected that manifest, preserving
the source but causing repeated archive work without reclamation.

The repair reserves an inode-aware archive working set before creating its
temporary file, preserves full file sizes/digests for hardlinks, and releases
the reservation after temporary cleanup on every exit. The systemd unit grants
access to the shared ledger. Canonical child and parent CPU replays also reserve
their diagnostic working set before creating scratch. These estimates are
admission budgets, not filesystem quotas; unusually large diagnostics still
need a correspondingly sized role footprint. Existing replay outputs remain
retained and must not be blindly reaped: they can contain linked publisher
inputs. Preserve or verify an admitted archive before retiring those copies.

A production-input profile of one completed-prefix validation took 207.50
seconds: 74.24 seconds in the common file hasher and 76.18 seconds in repeated
camera measurements, among other work. The factory repeats this validation
while selecting and sealing the adopted prefix.

The same saved production adoption passed with candidate `b05c66477` in
125.69 seconds, versus 207.50 seconds before: a 39 percent wall-time reduction
for one complete validation. Camera-measurement time fell from 76.18 to 11.85
seconds. Both runs used the same host, network isolation and retained inputs;
host contention can affect wall time. This is not yet a timing claim for a
whole factory submission.

The cache is confined to one outer synchronous operation and starts empty on
the next attempt. It hashes all bytes on first use and only reuses hashes of
read-only regular files of at least 1 MiB after checking inode, size, owner,
mode, link count and nanosecond modification/change times through an opened
descriptor. Mutable and small authority documents are hashed again. Pure
camera measurements are keyed by current frame digests, camera ID and every
gate parameter; returned values are copied. No authority verdict, queue state,
release check, provider-zero observation or scientific acceptance is cached.
There is no persistent cache to survive validator or dependency changes.

Verification covers changed bytes with restored mtime/mode, inode replacement,
symlink substitution, scope reset, mutable documents, parameter changes and
caller mutation; hardlinked archive publication/readback/restore; disk refusal
before archive or replay staging; prefix adoption and factory contracts; and
the real camera packet/worker/host fixture. The policy lifecycle rehearsal and
provider import closure protect the downstream paid boundary. No broad test
lane is used as a substitute for these changed-risk checks.
