# Policy canary inference and cell deadlines

This is an ADP-009D development-only precursor to the day-21 simulator rehearsal.
The completion artifact is a bounded two-candidate run with retained exact policy
inputs and terminal evidence, or an explicit interruption record. Controls were
omitted under the existing operator authority, so this work cannot qualify a
ranking, a winner, or a physical outcome.

## Observed failure

Scene 841757 run `operator-policy-841757-44d6dad93548-droid-camera-action-budget`
passed the actual rendered observation gate. OpenPI then failed its first request
with websocket code 1011 and `keepalive ping timeout`. The pinned upstream server
runs synchronous inference on its asynchronous server loop; its delayed pong can
trip the client's heartbeat deadline while inference is still pending. A local
reproduction using the pinned client/server and a delayed inference produced the
same error. The corrected transport returned one action response without retrying
the request.

The first isolated cell subsequently hit the old 900-second parent deadline.
GR00T had retained 73 request records and their input frames, but no terminal
score or action-delivery receipt. The GPU was destroyed, its posted charge was
$0.417, and provider-zero was verified. The interrupted result and 663 artifacts
were published to the Website. All 663 artifacts resolved and passed digest
verification as the control-plane service user.

## Encoded changes

- Retain the pinned OpenPI codec and inference implementation. Bound connection
  plus metadata receipt to 30 seconds, inference response receipt to 300 seconds
  measured from send start, and connection close to 5 seconds. Keep ping traffic
  but disable the pong deadline that conflicts with synchronous cold inference.
  Do not add connection or inference retries.
- Give an isolated cell 2700 seconds for native startup and both bounded policy
  episodes. The separately admitted allocation watchdog remains the total-run
  hard stop.
- Once a timed-out child is stopped, verify the frozen input/task/candidate
  bindings and retained file hashes, then add a blocked interruption record.
  Preserve source files, exact request bytes, lossless images, and existing
  failure receipts. Derive chronological review videos only from saved frames;
  label playback timing separately when native timestamps were not retained.
  Do not infer action delivery, a score, or a completed episode from requests.
- Allow the partial-result consumer to retain zero, one, or two bound episode
  rows from a blocked child. Reject duplicates, foreign candidates/cells/seeds,
  and mismatched task or execution identities.
- Include recovery and its pure worker validators in the actual shipped package.
  The isolated bundle test caught the missing worker package import before any
  paid attempt using these changes.

Validation uses the canary lifecycle rehearsal for real orchestration/close and
rebuild semantics; provider import closure for the actual sealed package; local
websocket tests for delayed inference, dead peers, startup/response deadlines,
and no retry; and interruption/dispatcher tests for immutable evidence, tampering,
identity pairing, and typed missingness. These checks do not prove GPU completion.
