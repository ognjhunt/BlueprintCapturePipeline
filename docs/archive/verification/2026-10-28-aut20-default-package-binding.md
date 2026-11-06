# AUT-20 default package binding

Reproduced [PR 3027's default-path reporting defect](https://github.com/ognjhunt/BlueprintCapturePipeline/pull/3027#discussion_r4404205505)
on merged `d9bd91fcb78eda61938a9a2ad507930f076b61ea`. With a capture root and
no package override, setup finds a passed canonical audit under
`<capture>/pipeline/arena_eval_package`, but the generated input packet had
`configured_paths.package_dir: null`. All eight local capability rows therefore
correctly refused the incomplete source binding. The missing propagation was
the producer defect; the proof audit's evidence checks were retained.

The control plane now carries setup's resolved default into the packet when
there is no explicit or environment package override. Setup continues to own
the default selection. Explicit CLI/argument selection retains precedence over
environment configuration, and a missing explicitly selected package cannot
fall back to an unrelated valid default.

Regression fixtures use the actual canonical package auditor and the real
setup, control-plane packet and proof-audit builders. Default, argument and
environment selections preserve only `local_package_contract` readiness with
`execution_proven: false`. Absent audit and mismatched capture/package source
bindings remain unproven, as do stale or missing artifact dependencies. No
readiness validator or execution boundary was relaxed.

On the merged base, the new default-path positive failed while argument and
environment positives passed. After the propagation fix, all **67 focused
control-plane/setup/proof/readiness tests passed in 8.24 seconds**. Changed-file
Ruff and whitespace checks passed. Red evidence and JUnit records are retained
at `/tmp/aut20-default-package-binding-20261028/`.

These are development-only local fixture results. Independent review, hosted
CI and normal root-owned publication/release remain separate milestones. No
provider/network calls, live operations, spending, hold release or production
readiness acceptance was performed.
