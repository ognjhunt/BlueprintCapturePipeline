# Protocol Amendment 2: Vast Stale-Offer and Private-Image Admission

Frozen prospectively: 2026-07-30T10:57:08-0500

## Triggering evidence

The first non-dry policy-identity launch reached Vast create for ask `39007828`
and received HTTP 410 with provider reason `no_such_ask`. The allocator then
confirmed that no resource with the successor prefix existed. No successor GPU
was allocated and attributable provider spend was USD 0. The preserved
watchdog, lease, and reservation are conservative bookkeeping, not evidence of
a live successor instance.

An independent registry audit also established that the pinned Docker Hub image
is private: anonymous manifest access returned HTTP 401. The generic Vast render
provider did not yet forward its existing Docker Hub login contract.

An unrelated writer's Vast instance remains outside this experiment. This
experiment may inspect it for the global one-GPU admission gate but may not stop,
alter, or claim it.

## Generic repair

The Vast provider must now:

- treat create HTTP 410, like 400, 404, 409, and 422, as a definitive stale or
  unavailable ask response and try the next already qualified offer;
- continue treating timeouts, HTTP 5xx responses, and success responses without
  a valid instance identifier as ambiguous and stop without a second create;
- resolve the existing Docker Hub username/PAT credential contract for images
  in the credentialed user's namespace;
- retain only the redacted image-login posture in the portable launch request;
- inject the raw Docker login only into the in-memory create payload at the Vast
  API boundary;
- block before create when a private-namespace image is selected but its PAT is
  unavailable; and
- never return, print, serialize, upload, or commit the raw login value.

Focused regressions must cover HTTP 400 and 410 fallback, private-image login
injection, secret non-retention, missing-PAT fail-closed behavior, and unchanged
ambiguity handling.

## Retry identity and admission

This repair changes runtime source. Any retry therefore requires:

1. focused tests and static checks;
2. a new clean, pushed immutable experimental commit;
3. a newly downloaded exact-commit source archive and digest;
4. a versioned input bundle and receipt binding that new source identity;
5. fresh unique object-store input and output keys;
6. fresh provider and writer inventory;
7. global one-GPU admission;
8. a newly armed watchdog, TTL, and reservation through the canonical allocator.

The failed input, output directory, provider response, and bookkeeping artifacts
remain immutable. They may not be overwritten or silently reused.

## Claim boundary

Passing the repaired launch path would prove only that the exact private image
and source overlay can reach the previously registered three-policy identity
canary. It does not prove learned-policy inference until all three native outputs
are retrieved and validated, and it proves nothing about repeated re-query, WAM
causality, episode coherence, ranking, abstention, physical outcomes, transfer,
economics, or the thesis.
