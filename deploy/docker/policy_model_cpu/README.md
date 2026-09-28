# Approved CPU model service

ADP-011/day 7 and ADP-050/day 28. This is the HTTP service for the explicit
`onnx_state_mlp_cpu_v1` profile, executed only inside the dedicated runsc policy
sandbox. No caller-selected model path or Python import is accepted over HTTP.
The image contains immutable model bytes and a secret-clean artifact manifest.

The trusted builder supplies `policy-model-build-input/policy.onnx` and
`policy-model-build-input/artifact.json` in a temporary build context, verifies
the original uploaded artifact's SHA-256 and size, and verifies its interface
against the frozen task. Never include storage credentials, private object
URIs, scene files or scoring code in that context. Do not start this image on
the shared control plane, mount a workspace, or give it external networking.

Use the existing company-policy v2 plan, credential lease, signed dedicated
worker boot receipt, measured denial probes, proxy, timeout and cleanup path.
Its fixed command is `python -m blueprint_pipeline.policy_model_server`,
listening on loopback port 8600 with HTTP access logs disabled. Loading the
graph occurs inside this worker, after digest verification.

`policy_model_packaging` builds a secret-clean context from the uploaded
generation and a frozen task contract. The model service refuses model path
carriers, non-finite state, wrong content type, extra routes and actions outside
the declared limits. Packaging is distinct from execution qualification.
