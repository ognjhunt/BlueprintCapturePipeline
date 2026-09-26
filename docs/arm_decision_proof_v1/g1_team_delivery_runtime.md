# G1 team policy delivery runtime

ADP-050, Day 28 development lane. A team's policy profile binds its owner,
retained site/task setup, G1 preset, observation schema, action schema, and one
delivery mode. Registration alone remains `planning_only`.

| Delivery mode | Team supplies | Controller's first executable check |
| --- | --- | --- |
| Authenticated endpoint | HTTPS URL and secret reference | Bound synthetic G1 request through the HTTPS client after origin and secret review |
| Container | Digest-pinned OCI image | Locally verified image in a networkless, read-only Docker process; JSONL synthetic probe and removal receipt |
| Noncontainer artifact | HTTPS archive URI, SHA-256, relative entrypoint | Operator-staged archive checked against the approved digest; regular files extracted and JSONL entrypoint probed in a networkless Linux namespace |

For a noncontainer artifact, an operator must fetch and stage the archive through
the governed input path first. `native_g1_team_artifact_runtime` does not fetch
team URLs or pass site observations during qualification. The archive may contain
regular files and directories only; links, special files, traversal, duplicates,
oversized expansion, and extraction that would breach the host's 8 GiB free-space
floor are rejected. The probe requires a privileged launcher
to create a Linux user namespace, then drops the policy to UID/GID 65534 with
all capabilities removed. It exposes the archive read-only, standard runtime
libraries, a private temporary filesystem, minimal environment variables, and
no network. Raw stderr stays quarantined. Its receipt proves one synthetic G1
wire exchange and process teardown; it does not prove model rights, paid-resource
admission, site-task scoring, or public redistribution.

`native_g1_team_runtime_session.open_g1_team_runtime_session` gives a worker
one mode-independent client after the matching synthetic wire probe. The
reviewed HTTPS origin and secret reference, pinned local OCI image, or
SHA-verified staged artifact remain mode-specific operator bindings. Closing
the session retains the child teardown receipt for process modes; the endpoint
credential is never written to a receipt. The same returned client can enter
`run_g1_team_scored_scene_episode` for the retained scene and independent
scorer. This is a worker seam, not an intake-to-paid-run controller.

The existing G1 shared-scene episode seam accepts a qualified client for either
the book objective or movement objective and writes development-only scores and
traces. A future controller must bind the selected registered policy, its
runtime identity, rights, site-observation authorization, bounded provider
allocation, scorer, media, billing, and teardown before this becomes a team-run
result. The currently staged Scene 841757 four-policy run uses Blueprint's
approved built-in candidates and remains a separate development evaluation.
