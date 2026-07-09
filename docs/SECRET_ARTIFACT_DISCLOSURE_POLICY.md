# Secret Artifact Disclosure Policy

Provider readiness, launch readiness, key-rotation, image-build, and object-store
staging artifacts may record whether required credential files are configured,
present, and owner-readable. Publishable release artifacts must not include:

- raw secret values
- secret hashes
- absolute local paths to credential files under `~/.blueprint-secrets` or any
  operator-specific temp/home directory
- fields such as `api_key_file_path`, `token_file`, or `secret_file.path` when
  the path points to local credential material

Allowed metadata:

- credential env var names, such as `RUNPOD_API_KEY_FILE`
- `path_source`, such as `env` or `default_blueprint_secret_file_path`
- boolean path-redaction markers
- `present`, `mode`, and `mode_is_0600`
- external rotation-record URIs or secret-manager version URIs that are safe for
  the launch review audience

Operator-only scratch artifacts may exist locally during paid canaries, but they
must be regenerated or redacted before inclusion in a release packet, buyer-facing
readout, PR evidence bundle, or launch readiness packet. A release-safe artifact
should include `secret_artifact_policy.local_secret_file_paths_recorded=false` or
equivalent evidence.

This policy does not convert credential presence into live provider proof.
Provider startup, runtime output collection, teardown, semantic task success, and
buyer delivery remain separate proof layers.
