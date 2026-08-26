# Task Evaluation Result delivery

Task Evaluation Results are delivered by a five-stage, fail-closed path. The
Pipeline remains the scientific and byte-integrity authority; the WebApp is a
tenant-scoped projection and authenticated transport.

1. **Validate** — verify the episode evidence index digest and independently
   re-hash every indexed receipt, multicamera manifest, exact lossless policy
   input, lossless camera frame, and external/wrist/overview review video.
2. **Seal** — bind the terminal Decision Envelope, evidence-index digest, run
   identity, and every customer-visible artifact into one delivery digest.
3. **Project** — produce a small secret-clean result projection for Firestore.
   Large media and evidence bytes remain Pipeline-owned.
4. **Package** — stream a deterministic ZIP64 review pack and full-evidence
   pack. The review pack contains human-review media and receipts; the full pack
   also contains the exact lossless policy inputs and camera frames. Packaging
   checks free disk first and never reconstructs missing evidence.
5. **Publish** — send the signed `task_evaluation_run_publication.v2` projection
   to the WebApp. The WebApp derives access from the authoritative capture owner
   and verified Firebase tenant; Pipeline cannot choose its audience.

## Customer and operations views

Each owner or verified organization sees only its own result cards, bounded
decision, five delivery stages, episode outcomes, external/wrist videos,
review-only overview, exact receipts/manifests, and review/full ZIP downloads.
Blueprint operations may inspect all tenant records for delivery health. There
is no public or cross-team leaderboard unless a separately authorized campaign
proves identical task, testbed, robot, candidates, seeds, scoring, and disclosure
rights.

## Storage and transport

Firestore stores the small immutable projection and access index. Evidence bytes
remain under the Pipeline run root and are served only by exact artifact ID from
the sealed registry. Each request is re-hashed by Pipeline, then streamed through
an authenticated WebApp proxy. Browser playback and downloads use short-lived,
artifact-bound tickets so phones can stream large videos and ZIPs without putting
Firebase bearer tokens in URLs or buffering the complete artifact in WebApp
memory. Email, Google Drive, and ad hoc shared links are not systems of record.

The current ADP rehearsal is `development_only`. Its videos are derived review
evidence, simulator results are not physical success, and successful execution
does not establish policy superiority, deployment approval, or safety.
