# Resume the original runtime installation before refreshing it

ADP-081 day-42 prerequisite: release the immutable usage recovery fix (#2597).
This is protected installation recovery, not completion of the economics gate.

Observed: after dependency extraction completed, initial runtime copying began
and a bounded deployment refused. Its exact retry retained the same partial
source tree. `prepare_deployment` selects refresh whenever installation.json
exists, although `prepare` writes that file as an immutable copy intent before
copying. Refresh requires the not-yet-installed dependencies and bootstrap.

Contract before editing:

- An installation intent without CURRENT or bootstrap is pending initial copy,
  not a completed baseline. Resume its exact historical source/SDK/bootstrap
  bytes using the newly authenticated helper; historical helpers are only data.
- Discover matching inputs only in existing protected input caches. Verify the
  complete source, SDK and bootstrap map reproduces the original canonical intent
  before any runtime copy. Reuse prepare's locked intent/prefix validation.
- Complete the original intent first, reread the selection, then use normal
  refresh to the new authenticated target. Never edit/delete the old intent or
  unknown partial state, issue authority, enable cleanup, or broaden access.
- Preserve the shared deadline, including interruption between completion and
  refresh. Verify retained source cache bytes against authenticated Git blob IDs
  directly on retry rather than reacquiring thousands of already-retained blobs.
  Hash, protected ancestry, size and immutable-byte requirements remain.
- Reproduce interrupted connected deployment, changed-commit recovery, tampered
  source/SDK/partial state, missing retained inputs, and interruption between
  preparation and refresh. Independent exact-head review and required CI precede
  release; installed hashes and normal-workflow outcome remain separate proofs.

An independent design reviewer found no existing supported route that completes
the changed-commit case while preserving the immutable original intent. This
isolated repair continues the same release dependency; it creates no new lane.

The protected filesystem test fixture now holds one shared file lock across
creation, testing and cleanup. Hosted four-worker runs exposed ancestry metadata
refusals in two unchanged initial-copy tests; each passed in isolation. Concurrent
fixture directory creation under the same protected home is a source of ancestor
mtime changes. Serialization removes that fixture interference without relaxing
production checks or repeatedly rerunning failures.

## Installation time allowance

Protected installation now has one fixed 900-second (15-minute) maximum. The
normal deploy wrapper and initial install wrapper establish the same ceiling;
the installer honors any earlier caller deadline, rejects nonfinite deadlines,
and passes one absolute deadline through source acquisition, SDK validation,
initial-intent recovery, refresh and installer publication. Phases cannot renew
that allowance. Expiry still refuses activation and retains the exact partial
intent for supported recovery.

The original 300-second ceiling was introduced for copying already-prepared
source and SDK inputs. Connected installation now also authenticates source,
builds/verifies the locked CPU SDK and may finish a historical intent before
refreshing. A measured host attempt spent 58 seconds staging source and another
215 seconds before its first newly copied SDK leaf. It then copied about 1,835
small leaves in 26 seconds before expiry; approximately 11,455 leaves and 593 MB
remained. Extrapolating that small-file rate gives about 164 seconds of remaining
copy work, before final verification and refresh. These measurements support a
finite 900-second installation allowance with margin; they do not guarantee
completion on every host. A local synthetic connected rehearsal completed in
76 seconds and does not establish production storage performance.

The tradeoff is that normal deployment can retain controller pauses and release
locks for up to ten minutes longer while installation proves its result. The
normal failure restoration and activation checks remain. This changes no scene
retirement/deletion allowance, consent expiry, rights or spend authority,
five-second native state query, intake timeout, or outer Door watchdog. The
30-second Git invocation limit, 32,768-leaf and 4-GiB inventory caps, protected
ownership/no-follow checks, exact hashes, immutable intents, file/directory
fsync and required native CI proof remain unchanged.
