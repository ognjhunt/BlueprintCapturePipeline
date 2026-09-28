# Blueprint synthetic company-policy rehearsal

ADP-011, Day 7 partner admission; no-spend precursor to ADP-050, Day 28 company-policy execution. This fixture exercises protocol and confidentiality boundaries with zero pixels and zero state only. It refuses every non-synthetic observation. It is not partner, robot performance, physical outcome, paid launch, or production fulfillment proof.

Build from the repository root using the Dockerfile in this directory. The 2026-09-28 proof image is private at `us-central1-docker.pkg.dev/blueprint-8c1ca/pipeline-jobs/company-policy-synthetic@sha256:909e782a917bf2a959bf9eb0ee0bf6da7463d156c27cb589a60626a18495ded4` (Linux ARM64).

Run only in a newly created dedicated Linux worker without workspace, customer, capture or scene mounts. The measured worker used Ubuntu 24.04, Docker and runsc 20260921.0. Install the supplied profiles under `/etc/blueprint/company-policy/`; load the AppArmor source with `apparmor_parser -r` and bind their exact SHA-256 values into the plan.

For this Docker sidecar layout, configure the dedicated worker runtime as:

```json
{"runtimes":{"runsc":{"path":"/usr/bin/runsc","runtimeArgs":["--network=host","--host-uds=create"]}}}
```

Docker must still start the proxy with `--network=none` and the policy with `--network=container:<proxy>`. The runsc networking flag uses that isolated Docker namespace, never the worker host namespace. Ordinary gVisor netstack keeps loopback private to each sandbox, so Docker namespace sharing alone does not connect the sidecar. The Unix-socket flag permits creation only in the proxy's explicit IPC mount; the policy has no host mounts. Never replace Docker's network arguments with `--network=host`. Qualification measures denial and reachability; runtime inventory alone is insufficient. See the upstream [gVisor networking documentation](https://gvisor.dev/docs/user_guide/networking/).

The vendored seccomp profile is the upstream Moby profile at commit `85e237f1fe229a0c61c9c7d8e743fa780d3b97ca`, `seccomp/default.json`, Apache-2.0. This is a synthetic worker fixture, not an approved GPU/customer production security profile.

The WebApp companion `scripts/proof-company-policy-synthetic.ts` requires an owner-only config and explicit `synthetic-local-emulator-only` acknowledgement. It pins Firestore to `demo-blueprint-policy-proof` at loopback port 8788, uses the existing development route-proof identity, local envelope encryption, the real candidate/credential routes, and the real outbox worker. `admission_server.py` runs the actual Pipeline HTTP app on loopback port 8801 with a dedicated temporary work root. The broker binds on Mac port 8802; reverse-forward guest port 8803 to it.

`proof.py <protected-input.json> synthetic-dedicated-vm-only` executes the actual sandbox executor without replacing its command runner, broker, socket readiness, proxy request, or security-profile checks. Input contains the exact admitted contract, admission receipt, private image, registry host, and tested Pipeline source commit. The broker token stays in owner-only `/tmp/blueprint-proof-broker-token`. The worker signing key is ephemeral and never included in retained artifacts. Persist the signed result, plan and boot receipt before stopping the worker. Delete temporary host registry authentication and broker/input files when done.

This rehearsal covers the synthetic pre-observation chain. Production Firebase identity, KMS, deployed outbox configuration, dedicated worker allocation, real-scene observation routing, episode execution, aggregate result delivery and financial settlement require separate proof.
