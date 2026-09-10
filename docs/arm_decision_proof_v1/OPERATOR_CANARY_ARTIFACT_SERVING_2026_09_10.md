# Serve the registered operator canary result store

ADP-009 development-only paired rehearsal, day-21 delivery gate: the V25 Website
publication succeeded, but its videos and deterministic score files returned404.
The live download resolver searched only the queued-run directory suffixed with
`-activation`; the operator publisher retained its sealed registry and evidence
under the exact run ID. The retained video reproduced
`result_delivery_registry_missing` as the service user before this fix.

The resolver now admits that exact direct-run directory only with a matching,
digest-valid internal diagnostic operator registration. Legacy-run precedence
and the queued activation mapping remain. Conflicting queued/direct stores,
cross-run registrations, symlinks, malformed records, paths outside the run, and
changed artifact bytes fail closed. Authentication remains on the HTTP endpoint;
artifact IDs still resolve only through the sealed registry.

The fifteen artifact API tests exercise authenticated full and range reads plus
those refusal boundaries. Live verification must then read the retained V25
video through the deployed authenticated endpoint and compare the complete
returned bytes to its sealed SHA-256. Publication acceptance by itself does not
prove a playable video or readable score receipt.
