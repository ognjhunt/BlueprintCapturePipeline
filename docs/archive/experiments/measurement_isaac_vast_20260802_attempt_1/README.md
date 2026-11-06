# Isaac Sim 6.0.1 Vast measurement canary: attempt 1

Date: 2026-08-02

## Outcome

The first paid Vast attempt allocated instance `46606301` and then failed in
the local result poller before the remote worker could publish a runtime
result. The initial, expected object-store `404 Not Found` was propagated as an
exception instead of being treated as "not ready yet." This attempt therefore
provides provider lifecycle and failure evidence only. It does not provide an
Isaac/PhysX execution result, qualification evidence, R5 evidence, an R6
decision, R7 admission, policy-ranking evidence, or physical-success evidence.

No automatic paid retry was performed.

## Immutable inputs and admission

- Attempt source commit: `deb3ffaa29213c9a2f862f9b835e560da295422b`
- Pushed experiment branch: `codex/measurement-isaac-vast-20260802`
- Request digest: `sha256:39e67506d68a80af7dc103a1a83d9f8f4739617a2ada03e4e19c88b92261356a`
- Bound request digest: `sha256:960917a51c4e9c581a425e057fa2338cb8719c37f627450121105a0184480a81`
- Admission digest: `sha256:ffdf2f9c7e0e03dc1ba4898b1f6eef0648da58f357dffe1243d1956d35f10c0f`
- Preflight digest: `sha256:2d476a11096c1ae255f0bcbd7f88f1b5cac478bdd2bc8a379df41b9429557082`
- Input bundle digest: `sha256:1de8e71b152ae60598bee3af5098acdc7be53a5c763fd85f6048323902600817`
- Bundle manifest digest: `sha256:6a646c30db86749a53a3c43efde3248fe68f6e5eebd6ed8c2bada35c70dd353c`
- Runtime release digest: `sha256:0c3e07ac61045e8e27707f6bc5373e078961e2041555556588e9fd52dc500623`
- Runtime image: `nvcr.io/nvidia/isaac-sim:6.0.1@sha256:783444c706538aa76cf5126e911ddc5e618779e6105305ad4af4260362a30aa9`
- Selected capacity: one Vast L40, 46,068 MB reported GPU RAM, approximately
  `$0.457037037/hour`
- Hard TTL: 1,800 seconds
- Spend cap: `$1.00`
- Retry cap: `0`

## Teardown and cost boundary

- Pending teardown opened: `2026-08-02T13:08:51.134381+00:00`
- Pending teardown closed: `2026-08-02T13:08:53.174650+00:00`
- Observed allocation window: approximately `2.040269` seconds
- Computed upper-bound usage at the selected hourly rate: approximately
  `$0.000259021`; this is a calculation, not a provider invoice
- Owner teardown receipt: `PASS`
- Teardown receipt digest: `sha256:ba7caf173e7f43514885d5180f9bea54527922fde5d45a72c068bc3892094b8e`
- Scoped and global Vast provider-zero receipt: `PASS`
- Provider-zero digest: `sha256:9354f295ef18bd54b545f5ad0535ccaf27e3f722756558a3fd16cb2d0d1556de`
- Independent watchdog status: `provider_terminal`
- Independent watchdog exact-id checks: instance `46606301` inspected twice and
  confirmed absent; scoped and global inventories were each confirmed zero
  twice
- Object-store cleanup: both attempt objects confirmed absent; signed URL files
  removed

## Encoded remediation

- `b258ed46abcb7bfb79e13529339a04200bbc63d5` maps output-object 404 responses
  to the existing bounded polling path.
- `4fcc96f05b41637845fd8108d0538739fc4b14ac` ensures unexpected output-fetch
  failures still return a durable failed execution record after teardown.
- The focused paid-lifecycle, admission, watchdog, and Isaac canary suite passes
  after remediation (`90 passed`).

A second paid execution requires fresh explicit authorization and a new
immutable bundle because the remediation changes the source commit.
