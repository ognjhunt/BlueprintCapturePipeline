# Provision SAM preparation services

ADP-009D/day-28 production source preparation uses two services: the launch
preparation worker validates the scene-bound SAM profile before queueing child
phases; the SAM execution worker reopens that same profile while running them.
A profile configured only on the execution service cannot start a fresh scene.

After the canonical `task_evaluation_sam31_preparation_profile` materializer
produces the exact deployed-commit profile, install both service bindings with:

```bash
python -m blueprint_pipeline.task_evaluation_sam31_service_provisioning \
  --profile "$SAM_PROFILE" \
  --expected-source-commit "$DEPLOYED_COMMIT" \
  --openai-api-key-file /etc/blueprint/provider-secrets/openai_api_key_sam31_visual_review \
  --openai-api-key-id "$SAM_REVIEW_KEY_ID" \
  --allow-live-agents-sdk \
  --reload-systemd \
  --receipt-out "$PROVISIONING_RECEIPT"
```

Run installation with authority to write `/etc/blueprint` and systemd drop-ins.
The supplied inference key file must be the operator-provisioned dedicated SAM
key for the profile's cost-attribution key ID; the provisioner checks that ID,
secret-file permissions, and the profile's existing evidence. It never reads
secret values or claims to verify the key against the provider API. Provisioning
records must retain the independent operator/API identity check.

`--allow-live-agents-sdk` represents explicit operator authorization for live SDK
execution. Without it, the SAM service receives an explicit disabled setting.
Only that service receives the inference-key path; the launch preparation
service receives the profile path alone. Existing generic credential files stay
untouched. Both drop-ins load a final `EnvironmentFile`, because a systemd
`Environment=` override cannot supersede values from an `EnvironmentFile`.

Root validation applies process-scoped Git trust only to the validated profile
checkout, pinned FlashSplat root, and its fixed submodule paths. It restores the
previous environment after validation, including on failure. Global Git
configuration is unchanged and wildcard trust is never introduced.

The command retains immutable environment snapshots for each binding, verifies
written bytes, and reloads definitions only when `--reload-systemd` is present.
It never starts or restarts a service, submits a scene, allocates a provider, or
performs inference. Coordinate updates with active scene ownership, and run this
command again with the freshly generated profile after any execution-commit or
task change. Retain earlier profiles, environment snapshots, and receipts.
