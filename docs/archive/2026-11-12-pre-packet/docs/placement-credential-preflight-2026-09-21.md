# Placement credential preflight and key renewal

ADP-009D / day-21. The website-origin selected evaluation reached deterministic CPU placement but refused before inference with `configured_controls_openai_authority_invalid`: the sealed intent referenced the older visual-review API key while the service used its renewed replacement. No model reservation was written.

The placement environment check now runs before geometry. The saved-input CPU rehearsal also validates credentials up front and constructs the actual cost gate before stopping; it never calls reserve or starts inference. This closes the previous rehearsal blind spot where replacing the entire gate builder skipped its deterministic checks.

A mismatch still refuses by default. An operator may provide `BLUEPRINT_OPENAI_PLACEMENT_KEY_ROTATION_FILE` pointing to a regular, non-writable-by-group/others JSON file with exactly these fields:

```json
{
  "schema_version": "configured_controls_key_rotation.v1",
  "project_id": "proj_example",
  "credential_role": "artifixer_visual_review",
  "paid_resource_class": "task_evaluation_configured_controls_robot_placement",
  "from_api_key_id": "key_previous",
  "to_api_key_id": "key_renewed",
  "authorization_reference": "operator-approved-key-renewal"
}
```

The legacy credential-role name does not invoke ArtiFixer. The mapping authorizes only the exact key replacement in the same project, role and placement lane. The original signed intent and maximum cost remain unchanged. The actual gate records the mapping digest, uses the renewed key's official cost scope, and still reserves before inference under the existing exclusive launch locks. An absent/mismatched mapping cannot substitute an arbitrary key.

Deployment preparation may refresh a selected evaluation that has only deferred inputs, deterministic CPU placement files, empty inference-attempt directories and cost-lock receipts. It preserves every byte and old reservation. Any official inference reservation, completion, accepted agent artifact, plan, unknown file or symlink prevents this refresh. No terminal stage is fabricated and no paid attempt is restarted.

Verification: focused credential/rotation/refusal tests, connected release handoff, CPU rehearsal boundaries and existing autostart tests. Host preflight must run as blueprint against the service environment before deployment. The live controller owns subsequent inference and evaluation.
