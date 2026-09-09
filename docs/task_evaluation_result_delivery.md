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
an authenticated WebApp proxy. Email, Google Drive, and ad hoc shared links are
not systems of record.

The current ADP rehearsal is `development_only`. Its videos are derived review
evidence, simulator results are not physical success, and successful execution
does not establish policy superiority, deployment approval, or safety.
# Operator diagnostic run delivery

ADP-009D, day-28 public-scene rehearsal: an operator-authorized run may have a
new task contract while retaining an earlier offering only as scene provenance.
The Website registers its immutable run, owner, task, policies, and input
digests without forwarding another launch. Its v4 publication carries both
`plan_digest` and `operator_registration_digest`; either without the other is
refused. The Website validates those values against the retained registration.

For an explicitly authorized omission of controls, the delivery materializer
accepts the original `control_omission_authority`, validates its task binding,
and retains it as a downloadable artifact. Delivery and projection carry
`controls_omitted_by_user`, zero planned/completed control rollouts, and the
omission digest. Required-controls contracts cannot use this path. Results stay
`diagnostic_policy_execution`; omission never authorizes policy ranking or
physical-success claims. Source episode results remain unchanged.
