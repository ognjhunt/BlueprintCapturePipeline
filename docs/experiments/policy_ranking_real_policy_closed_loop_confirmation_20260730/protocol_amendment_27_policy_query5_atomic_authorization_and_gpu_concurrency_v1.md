# Protocol Amendment 27: policy-query-5 atomic authorization and GPU concurrency

Status: frozen before policy query 5 provider execution

Date: 2026-08-01

## Reason for amendment

The previously frozen query-5 request bound the canonical allocator's provider
launch request to a policy-query runtime that validated the request as an input
but did not pass it into the OpenPI campaign or consume it. That would have made
the single-use authorization decorative rather than fail-closed. No query-5
provider allocation or policy request used that runtime or authorization.

The generic correction is frozen at immutable pushed runtime SHA
`4f6ed4f9654298b474f89a4e1370eb9ca8623d05`. The runtime now validates the
authorization's experiment, ID, exact runtime SHA, exact input SHA, policy,
query index, resource limits, provider-zero boundary, and excluded physical and
evaluator scopes. Immediately before provider mutation it atomically writes a
private mode-0600 consumption record under a mode-0700 local authority root.
Reuse or inability to persist the record blocks with zero provider mutations.

The focused policy-launch, paid-allocator, and production-campaign-budget gate
passes: 72 tests passed, lint passed, formatting passed, and the published
branch is clean with `0 0` divergence from its upstream.

## Superseded query-5 artifacts

`compute_authorization_openpi_policy_query_5.json` and
`policy_query_5_input_529bc396_v1.zip` remain preserved as historical evidence,
but are prospectively superseded and may not authorize provider mutation. They
were not consumed and incurred no query-5 provider spend.

The replacement runtime source archive SHA-256 is
`0ddd18dd8feec1b9e216f8a0d5bc9ece45326332857e0cc94aed6c20676a017e`;
its audit receipt SHA-256 is
`5cb5c005ca6664e206689bc0393eaf034f3e3bbc94c637a62b7b1f83e00af552`.
The replacement policy-query-5 input archive SHA-256 is
`b9ac62359b7f60a39521b8f2556f01ae64adf9ca79cc92484aa3ea78ee2affa7`;
its input-receipt file SHA-256 is
`11631fb8481db608d09ac3bf8ee3ce645db00598f6b9540a1b4e770ac0f08829`;
and its independent safe-extraction receipt SHA-256 is
`7590259b3a5d42a3d1c9fc4a5e3bdb3a1595560edc0e21307151361958b2a76c`.
The observation, checkpoint, image, WAM4 output, policy identity, query index,
and no-future-physical-data boundaries are unchanged.

## Prospective GPU-concurrency amendment

At `2026-08-01T08:45:31-0500`, the user explicitly amended the campaign maximum
from one to two concurrent GPUs. For paid stages beginning after this amendment,
the global concurrency ceiling is therefore two. Historical one-GPU admission
and result records remain unchanged.

This is a ceiling, not a requirement to allocate two GPUs. Each provider launch
must still have its own immutable input, single-use authorization, cumulative
budget reservation, stage-local cap, watchdog, TTL, teardown, and provider-zero
evidence. Policy query 5 remains authorized for exactly one provider allocation
and one policy request. It may overlap another authorized allocation only when
the global active total would remain at or below two and ownership is explicit;
unowned resources must never be modified or closed.

## Next finite gate

Stage the replacement bundle under a new immutable object key, prove the output
key absent, refresh provider inventory and provider preflight, and pass a new
canonical allocator dry run using the replacement authorization. Only then may
one live query-5 allocation execute. WAM5 remains neither authorized nor
running. Judges remain forbidden until the complete 12-interaction episode and
the registered causal-control matrix pass.
