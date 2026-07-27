# Evidence claim matrix

`yes` means the artifact supplies evidence for that class; `bounded` means it supports only the limitation stated; `no` means it must not be used for that claim.

| Artifact or group | Implementation | Runtime | Generated media | Simulator outcome | Evaluator validity | Ranking fidelity | Captured-site portability | Cost comparison | Physical performance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `environment_and_source_manifest.json`, source modules, tests | yes | no | no | no | no | no | no | no | no |
| `provider_evidence/heldout/`, `provider_matrix_closure.json` | bounded | yes | no | no | no | no | no | bounded | no |
| `heldout_causal_conditioning_report.json`, development causal reports | no | bounded | bounded | no | yes, falsification only | no | no | no | no |
| `wam_action_semantics_report.json`, `action_following_validation_report.json` | no | no | bounded | no | yes, semantics limits | no | no | no | no |
| `frozen_benchmark_calibration_report.json` | no | bounded | bounded | no | yes | bounded historical calibration only | no | no | no |
| `risk_coverage_curves.json`, `policy_rankings_with_confidence_intervals.json` | no | no | no | no | no | no; explicit unavailable records | no | no | no |
| `warehouse_controlled_hybrid_report.json` | no | bounded historical | no | bounded historical | no | no | no | no | no |
| `interiorgs_captured_transfer_report.json` | no | bounded historical | no | bounded historical | no | no | yes, negative/limited result | no | no |
| `economics_and_time_report.json`, spend/provider-zero records | no | bounded | no | no | no | no | no | bounded; physical comparator incomplete | no |
| `final_verdict.json`, `final_report.md` | no | no | no | no | synthesis | synthesis | synthesis | synthesis | no |
| rights, label-access, contradictions, and limitations records | no | no | no | no | claim constraints | claim constraints | claim constraints | claim constraints | claim constraints |
| `successor_model_decision.json`, `gemini_discovery_ledger.json` | no | model-discovery only | no | no | no | no | no | projected only | no |
| `successor_wam_decision.json` | no | model-discovery only | no | no | no | no | no | projected only | no |
| `statistical_correction_amendment_002.json` | no | no | no | no | uncertainty-method correction only | no | no | no | no |

No Experiment-2 artifact proves physical task performance. Generated media and simulator execution remain support evidence and are never promoted to physical truth.
