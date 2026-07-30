# Protocol amendment 14: request-bound Ctrl-World paid profile freeze

Frozen prospectively before any current-reference Ctrl-World output existed.

## Admission correction

The generic successor paid lane already inspected Blueprint Ctrl-World
current-reference bundle contents, but its receipt-to-profile selector did not
yet have a safe way to bind a request-specific bundle digest. That digest cannot
exist until the preceding real-policy canary has produced and hashed the native
action used to build the WAM request.

Before a current-reference WAM allocation, the experiment must therefore commit
one `policy_ranking_ctrl_world_current_reference_gpu_profile_freeze.v1` file to
the immutable launch SHA. The canonical allocator requires that file to be:

- a regular, non-symlink file inside the launch repository;
- tracked by Git and byte-identical to the file at `HEAD`;
- bound to the exact launch commit;
- self-hashed with Blueprint canonical JSON;
- bound to the exact bundle SHA-256, size, embedded input hashes, source,
  checkpoint, runtime image, request count, provider, authorization identity,
  TTL, hourly-rate ceiling, and spend ceiling.

An absent, external, untracked, dirty, staged-only, digest-invalid, or
semantically incompatible profile fails before authorization consumption or any
provider mutation. The profile is passed only through
`python -m blueprint_pipeline.paid_resource_allocator gpu-canary ...`; it is not
a new launcher.

## Frozen limits

The first current-reference profile must retain one Vast allocation, one WAM
qualification request, zero scientific-matrix requests, and one total initial
generation. Its projected worst-case hourly-rate times TTL must fit its target
spend, and that target must fit its per-allocation and campaign ceilings.

The final profile values and exact bundle bindings will be frozen only after the
real-policy output exists and before the WAM provider is called. This ordering
does not permit changing the policy output, WAM request, causal controls,
reliability thresholds, scientific thresholds, or claim ceiling after seeing a
WAM output.

## Claim boundary

An admitted profile proves only immutable paid-request binding and fail-closed
resource authority. It is not WAM execution, valid generated media, policy
re-query, causal qualification, ranking evidence, physical confirmation,
captured-site transfer, or economics evidence.
