# Arm Decision Proof v1 Replay

From the repository root, the one-command reconstruction is:

```bash
PYTHONPATH=src .venv/bin/python -m blueprint_pipeline.arm_decision_proof
```

It consumes the admitted manifest, cached immutable execution directory, and
separate published physical-outcome artifact at their canonical paths. If the
execution cache is absent, the command fails closed and prints this exact
canonical acquisition command:

```bash
BLUEPRINT_LAUNCH_DETACHED_GPU_CANARY_SUPERVISOR_DIR=output/arm_decision_proof_v1/supervisor PYTHONPATH=src .venv/bin/python -m blueprint_pipeline.paid_resource_allocator gpu-canary --probe-kind adp-simpler-public-reference --provider vast --provider-launch-request output/arm_decision_proof_v1/unused_request.json --release-evidence output/arm_decision_proof_v1/unused_release.json --model-cache-evidence output/arm_decision_proof_v1/unused_model.json --preflight-bundle output/arm_decision_proof_v1/unused_preflight.json --admission-out output/arm_decision_proof_v1/paid_admission.json --bound-request-out output/arm_decision_proof_v1/unused_bound.json --adapter-output output/arm_decision_proof_v1/allocator_result.json --pod-name adp-simpler --adp-public-reference-manifest docs/arm_decision_proof_v1/manifests/simpler_google_robot_pick_coke_can.v1.json --adp-job-dir output/arm_decision_proof_v1 --adp-machine-avoidlist docs/arm_decision_proof_v1/manifests/simpler_vast_machine_avoidlist.v1.json --adp-max-hourly-rate-usd 0.80 --adp-max-spend-usd 2.00 --adp-hard-ttl-seconds 7200 --execute
```

The acquisition transfers public digest-bound runtime inputs only. It does not
include the separate physical-outcome values. It is capped at one Vast instance,
`$0.80/hour`, `$2.00`, 7,200 seconds, and zero internal retries, with teardown,
provider-zero, and staged-object absence receipts.

The result can qualify only Blueprint's bounded retrospective external-reference
harness. It remains `development_only`; it is not prospective validation,
deployment readiness, safety evidence, customer value, a digital twin, general
sim-to-real fidelity, general policy ranking, or rank correlation.
