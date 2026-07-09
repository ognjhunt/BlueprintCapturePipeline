# Runbook: GPU provider API key rotation

This runbook covers metadata and evidence for rotating Blueprint GPU provider
API keys. It does not rotate secrets by itself and must not record raw secret
values in artifacts.

## Scope

Covered provider credentials:

| Provider | Default local key file | File env override | Inline env accepted by adapters |
| --- | --- | --- | --- |
| RunPod | `~/.blueprint-secrets/runpod_api_key` | `RUNPOD_API_KEY_FILE` | `RUNPOD_API_KEY` |
| Vast.ai | `~/.blueprint-secrets/vast_api_key` | `VAST_API_KEY_FILE` | `VAST_API_KEY` |
| Lambda Cloud | `~/.blueprint-secrets/lambda_api_key` | `LAMBDA_API_KEY_FILE` | `LAMBDA_API_KEY` |
| DigitalOcean | `~/.blueprint-secrets/digitalocean_api_token` | `DIGITALOCEAN_TOKEN_FILE`, `DIGITALOCEAN_API_TOKEN_FILE` | `DIGITALOCEAN_ACCESS_TOKEN`, `DIGITALOCEAN_API_TOKEN` |

The manifest helper checks local configuration and the rotation ledger only. A
passing manifest proves that local key material is present and has fresh
rotation metadata; it does not prove a live provider call, paid launch,
teardown, simulator execution, or task success.

## Rotation procedure

1. Create a metadata-only baseline manifest:

```bash
python -m blueprint_pipeline.gpu_provider_key_rotation \
  --owner platform-security \
  --output output/provider-security/gpu_provider_key_rotation_manifest.json
```

2. Rotate each key in the provider console or approved secret manager. Capture
   the provider-side evidence as a durable record URI, such as a secret-manager
   version, ticket, or access-controlled audit log URL.

3. Update the local file-based secret. Use an editor, password manager CLI, or
   secret manager sync path that does not put the new key in shell history.
   Ensure the resulting local file is owner-readable only:

```bash
chmod 600 "$HOME/.blueprint-secrets/runpod_api_key"
chmod 600 "$HOME/.blueprint-secrets/vast_api_key"
chmod 600 "$HOME/.blueprint-secrets/lambda_api_key"
chmod 600 "$HOME/.blueprint-secrets/digitalocean_api_token"
```

4. Mark each provider rotated in the local ledger. Repeat once per provider:

```bash
python -m blueprint_pipeline.gpu_provider_key_rotation \
  --mark-rotated runpod \
  --rotation-record-uri "secret-manager://blueprint/runpod_api_key/versions/<version-or-ticket>" \
  --owner platform-security \
  --output output/provider-security/gpu_provider_key_rotation_manifest.json
```

Valid `--mark-rotated` values are `runpod`, `vast`, `lambda`, and
`digitalocean`.

5. Generate the launch evidence manifest and fail closed on stale or missing
   rotation proof:

```bash
python -m blueprint_pipeline.gpu_provider_key_rotation \
  --owner platform-security \
  --output output/provider-security/gpu_provider_key_rotation_manifest.json \
  --fail-on-blocked
```

The default freshness window is 90 days. Override only with an explicit launch
or security decision:

```bash
python -m blueprint_pipeline.gpu_provider_key_rotation \
  --owner platform-security \
  --max-age-days 30 \
  --output output/provider-security/gpu_provider_key_rotation_manifest.json \
  --fail-on-blocked
```

## Artifact contract

The manifest schema is `gpu_provider_key_rotation_manifest.v1`. The ledger schema
is `gpu_provider_key_rotation_ledger.v1`.

Required before a beta/provider launch:

- `status` is `passed`.
- Every required provider has `configured_secret_present: true`.
- Every required provider has `last_rotated_at`, `rotation_owner`, and
  `rotation_record_uri`.
- `days_since_rotation` is less than or equal to `max_age_days`.
- `secret_values_recorded` is `false`.

Keep this artifact separate from spend-guard, startup, teardown, review-media,
and semantic-success proof. It is credential-rotation evidence only.
