# Operator tools that nothing starts automatically

No systemd unit, operator door script, deploy tool, workflow, console script or
GPU bundle list names these 19 modules, so a reachability scan reports them as
unreachable. The November 2026 simplification audit classed them as operator
tools to document, not as removal candidates, and asked for this list (plan
P-A11) so that the next cleanup does not delete them by mistake. Whether each
one is still wanted is a separate question; this page only makes them findable.

Listing a tool here does not authorize running it:

- Classify it first, as [`../scripts/AGENTS.md`](../scripts/AGENTS.md) says:
  read-only check, local artifact writer, provider caller, GPU runner or deploy.
- The last column says whether `scripts/paid_api_call_sites.json` records a
  priced API call site in the file. A "no" there is not proof that the tool is
  free: a tool can also start paid compute through another module. Paid compute
  goes through `blueprint_pipeline.paid_resource_allocator` only (`AGENTS.md`).
- On the host, tools run through the operator door
  ([`OPERATOR_DOOR.md`](OPERATOR_DOOR.md)), never over SSH.

The description is the first line of each module's own docstring.

| Module (`blueprint_pipeline.…`) | What its docstring says | How it is started | Priced API call site on record |
|---|---|---|---|
| `agent_execution.comparison` | Frozen operational comparison of the SDK and managed session runtimes | `python -m blueprint_pipeline.agent_execution.comparison` | no |
| `atlas_preview_replay` | Replay an Atlas preview job's next controller steps against this tree, paying for nothing | `python -m blueprint_pipeline.atlas_preview_replay` | yes |
| `atlas_retained_handoff` | Adopt a completed background into its capture job without another provider call | no command line of its own | no |
| `claude_opus_model_preflight` | No-inference Opus 5.5 availability and published-price preflight | `python -m blueprint_pipeline.claude_opus_model_preflight` | yes |
| `cloud_tasks_dispatch` | Cloud Tasks dispatch for job envelopes — explicit-job transport lane | no command line of its own | no |
| `decision_evidence_handoff` | Verify the versioned Pipeline-to-WebApp Decision/Evidence handoff | `python -m blueprint_pipeline.decision_evidence_handoff` | no |
| `gear_sonic_container_smoke` | Container smoke for the pinned GEAR-SONIC WBC tree and real robot model XML | `python -m blueprint_pipeline.gear_sonic_container_smoke` | no |
| `groot_oscar_cached_footprint` | Audit measured host-image plus external-model cached worker footprint | `python -m blueprint_pipeline.groot_oscar_cached_footprint` | no |
| `isaac_api_contract` | Check our calls into Isaac Lab and Arena against the sources production runs | `python -m blueprint_pipeline.isaac_api_contract` | no |
| `nvidia_siggraph_completion` | Requirement-to-evidence matrix for the SIGGRAPH 2026 implementation memo | `python -m blueprint_pipeline.nvidia_siggraph_completion` | no |
| `omniverse_library_preflight` | Fail-closed ovrtx and ovphysx external-worker preflight contracts | `python -m blueprint_pipeline.omniverse_library_preflight` | no |
| `openai_prompt_cache_canary` | Bounded five-call GPT-6.1 Sol explicit prompt-cache mechanics canary | `python -m blueprint_pipeline.openai_prompt_cache_canary` | yes |
| `portable_evidence_receipt_mirror` | Mirror digest-bound evidence receipts without copying raw evidence bytes | no command line of its own | no |
| `postshot_license_transport` | Single-use, delete-acknowledged transport for Postshot licence material | no command line of its own | no |
| `run_summary_aggregation` | Aggregate per-capture pipeline run summaries without upgrading evidence claims | `python -m blueprint_pipeline.run_summary_aggregation` | no |
| `seal_blast_radius` | Answer "what breaks if I amend this seal?" before anyone amends it | `python -m blueprint_pipeline.seal_blast_radius` | no |
| `vast_authorized_probe_runner` | Coordinate the next authorized Vast Blueprint bundle probe | `python -m blueprint_pipeline.vast_authorized_probe_runner` | no |
| `vast_safe_status` | Print an allowlisted Vast instance summary without provider-secret fields | `python -m blueprint_pipeline.vast_safe_status` | no |
| `website_atlas_operator` | Resume one admitted Atlas operation through the canonical allocator | `python -m blueprint_pipeline.website_atlas_operator` | yes |

`atlas_preview_replay` is also a documented command in `AGENTS.md`, and
`website_atlas_operator` is described in
[`runbooks/atlas-beta.md`](runbooks/atlas-beta.md).

When one of these tools gets a unit, a door script or a workflow, or is
deleted, remove its row.
