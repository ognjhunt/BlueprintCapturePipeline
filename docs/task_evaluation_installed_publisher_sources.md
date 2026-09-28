# Installed publisher sources for production preparation

The production preparation worker can consume exact publisher bytes already
installed on its host without uploading them or downloading them again. This is
an ADP-009 scene-construction input binding, not a geometry or policy qualification.

The operator-owned service environment may set
`BLUEPRINT_TASK_EVALUATION_INSTALLED_SOURCE_BINDINGS_JSON` to a JSON list:

```json
[
  {
    "installation_receipt_path": "/var/lib/blueprint/task-evaluation-inputs/<packet>/public_scene_host_input_installation_receipt.v1.json",
    "publisher_intake_path": "/var/lib/blueprint/task-evaluation-inputs/<source>/publisher-intake.json",
    "publisher_intake_sha256": "sha256:<exact full-file SHA-256>"
  }
]
```

Publish this configuration through the production release/environment process;
never accept it from the submitted request. Both receipt paths and every resolved
member must remain inside the configured production input root, without symlinks.
The historical publisher receipt is pinned as complete bytes; it need not have a
self-digest. If it does have one, that digest must also verify.

Before enabling a binding, use the protected-main
`public_scene_host_input_intake stage` command on production to install the
rights-bound `public_scene_host_input_request.v2` packet. The installation receipt
must bind the exact submission execution commit and prove the executing service
account can read the immutable bytes. Run `public_scene_host_input_intake prepare`
for the source-context records separately. Do not hand-author installation success.

Requests retain the actual HTTPS publisher URLs, pinned to a full revision.
Only an exact publisher URL joined through the pinned publisher receipt to one
installed file digest and byte size gains host-only admission. No broad
Hugging Face allowlist or redirect permission is added. Other references continue
to use the existing operator-owned object-store prefix policy.

Every use revalidates the installation self-digest, commit, service identity,
publisher receipt pin, source path, full SHA-256, and byte size, even if the
preparation content-addressed store already contains those bytes. Changed or
invalid configured bindings fail closed without falling back to network.
The worker copies admitted bytes into its normal host content-addressed store,
retaining installation/publisher receipt digests and explicit no-network and
no-upload evidence in its result. Canonical publisher files are never modified.
Missing configuration preserves existing behavior.

Focused verification:
`tests/test_task_evaluation_installed_source_bindings.py` covers host-only
materialization, operator configuration, stale commits, invalid receipts,
digest/size joins, rejected paths/symlinks, non-pinned URLs, cache tampering,
no-network failures, and the actual preparation entrypoint.
