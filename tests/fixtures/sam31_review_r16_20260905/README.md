# Golden production documents: SAM review, scene 841757, attempt R16 (2026-09-05 21:54Z)

Retained, unmodified documents from child `sam31-8359e690c3369b20…` on the production
control plane. The Agents SDK reviewer **accepted** the selection (16 frames, 9 with
masks), yet the review receipt seal refused the execution with
`sam31_review_execution_receipt_invalid`: the inference-reservation ledger (hardened
2026-08-31, `39dd7b970`) keeps a completed call's reconciled cost in
`reserved_max_cost_usd` until official billing posts, while the seal (2026-08-15) still
demanded `0.0`. No SAM review ran between those dates; a paid run was the first time the
two documents met. Tests read these files so that seam is exercised in CI with the real
producer output. No secrets: `raw_secret_values_recorded: false`; paths are host paths.
