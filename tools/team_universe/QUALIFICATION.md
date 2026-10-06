# Audited robot offering qualification

The old discovery/screen records verify source quotations and company identity. Their robot-form,
task-family and API/simulation/model signals use broad vocabularies. They do **not** prove that a
company offers a physical robot, deployed robot control or robot integration, or that Blueprint can
currently evaluate its stack. In particular, a name containing Robotics, an industrial installation,
MES/analytics/RPA, an inspection camera, a world model or simulation asset is insufficient.

Ranking writes `ranked.team-rank.v2.json`. It preserves the old v1 output, raw result/page bytes,
paid-query schemas, receipt identities and spending protections. It reads no page and calls no
provider. There is no CRM/Render consumer integration. The private audit is a reviewer assessment;
its digests and reference are neither owner approval nor authority for a paid run, contact or upload.

## Independent axes

- **Physical offering:** an actual robot product, deployable robot control/policy stack with identified
  hardware and physical task, or an integrator's actual robot deployment/support. Software can qualify:
  a palletizing controller for an identified robot or robotic CNC integration is a capability. Generic
  software without that concrete offering relation stays reference or pending.
- **Current task fit:** a reviewed physical task and current offering/deployment evidence. Fresh funding,
  discovery dates, geography or an invitation cannot make old hardware/task evidence current.
- **Blueprint compatibility:** a current independently pinned Blueprint runtime/profile support contract
  must bind the exact working embodiment, physical task, observation/action interfaces and runnable
  controller. API/simulation/model words, vendor-page facts and a profile label cannot establish it.
  `not_verified` is the default. No current support registry is admitted by this version, so even a manual
  `supported` claim fails closed with `blueprint_support_contract_unavailable`; all legacy screens remain
  unready for beta. A future reviewed registry extension would still confer no execution admission.
- **Invitation:** an own-page explicit public design-partner/pilot/customer/general invitation is
  recorded independently. It implies no relationship or willingness to work with Blueprint.

`capability_prospect` requires a current physical offering and task fit plus verified identity. It does
not require a high funding/openness score or evaluation compatibility. `beta_candidate` additionally
requires supported compatibility, positive family weight, published business contact route and the
reviewed score threshold. `reference_only`, `pending`, and `insufficient` remain separate statuses;
no row is automatically labelled a prospect merely because it was discovered.

## Private overlay contract

`blueprint.team-manual-eligibility-audit.v1` has exactly these root keys:

| Key | Meaning |
| --- | --- |
| `schema_version` | Exact audit version above |
| `team_list_sha256` | SHA256 of retained `workspace.teams_path()` bytes; its teams must match the recomputed list |
| `decisions_sha256`, `auditor_script_sha256` | Auditor provenance digests, not approvals |
| `audited_at` | UTC ISO timestamp, no later than the assessment date |
| `auditor_reference` | Printable stable reviewer reference |
| `scope_sha256` | `qualification.scope_manifest(teams, screens)["sha256"]` |
| `teams` | Exactly one assessment for every recomputed discovered/screened team |

The helper returns bindings sorted by `team_key`. Each binding holds `team_key`, `domain`,
`discovery_sha256` (SHA256 of `site_screen.canonical(full_recomputed_team).encode()`, or null),
`run_id`, `result_sha256`, `evidence_sha256` (all null if unscreened). Duplicate/partial/extra scope,
changed source bytes, wrong digests and duplicate JSON keys refuse before output is written.

Each row adds the required fields declared in `qualification.ROW_KEYS`: `capability_class`,
literal boolean `manual_current_task_fit`, `capability_reason`, `capability_evidence`,
`blueprint_current_evaluation_compatibility`, `published_partner_intent`,
`willingness_to_work_with_blueprint="unknown"`, and literal `promotion_allowed=false`.

Allowed classes are `physical_robot_task`, `embodied_control_task`, `robot_integrator_task`,
`physical_robot_task_fit_pending`, `embodied_control_task_pending`, `adjacent_reference`,
`insufficient_evidence`, and `unscreened`. Null screen bindings must use `unscreened`.

Positive classes additionally need `identified_hardware`, `physical_task`, `offering_relation`,
`capability_as_of`, `capability_current_basis`, `robot_forms`, and `task_families`. The concrete physical
form cannot be `software_only`; a software supplier's actual controlled robot form belongs here.
The three descriptions must be explicit phrases held in the cited retained independently read quotes, with
at least two words each. This is structural anchoring, **not semantic inference**: the independent
reviewer must establish the actual product/control/integration relation and specific hardware/task.
Ambiguous hardware or future orchestration remains pending even if its company is source verified.

The offering relation must have an owned-page anchor. Physical hardware/task detail can additionally
come from an independently read third-party report whose quote names the same company. Citation-only
excerpts and unattributed references to other robots cannot satisfy the positive offering gate.

`current_offering_page` dates the reviewed *currently offered* product to the retained own-page
`checked_on`; it must not be used for a page that only describes a historical or future deployment.
`dated_deployment` must use the retained `task_evidence_date` and include a task_evidence proof.
The capability date must not be future or older than the existing 548-day evidence window at ranking
time. Current evidence is checked again when ranking, not inherited from fresh funding or discovery.

Each proof has `field` (a retained screen proof stem), `url`, `quote`, `quote_sha256`, and `level`;
`verified_on_page` requires `page_sha256`, the SHA256 of the retained page text. Optional
`tool_result_sha256` binds its original read hash. Seal may redact email text after that original hash
was computed: the two digests can legitimately differ, but the exact retained text digest must match.
Original raw results/evidence and provenance hashes are preserved. The exact URL must be that retained field URL. A quote may
select additional informative text on that same retained page, but no new URL is admitted. The page
text digest and whole quote are rechecked. Citation-only quotes can support reference assessment;
they cannot pass positive hardware/task/interface or explicit invitation gates.

`published_partner_intent` is `{state, kind, evidence}`. State is `explicit_invitation`,
`not_established`, or `unknown`; kind is `design_partner`, `pilot`, `customer`, `general`, or `none`.
An explicit invitation needs separately retained own-page proof. The reviewer judges the meaning.

Compatibility is `not_verified`, `unsupported`, or `supported`. The latter needs `evaluation`
with exactly `profile`, `working_embodiment`, `physical_task`, `observation_interface`,
`action_interface`, `runnable_controller`, `support_reference`, `as_of`, and `evidence`.
The current profile is `arm-decision-proof-v1`; embodiment/task must exactly match the positive
offering; all interface/controller phrases must be held in current retained own-page proofs.
`support_reference` identifies the independent current support assessment, not an owner approval.
These vendor facts are candidate evidence only. `supported` currently fails closed regardless of them:
`support_reference` cannot be substituted for actual independently pinned Blueprint runtime artifacts.

## Offline operation

Prepare the complete overlay privately, review its evidence and freeze it before ranking:

```bash
python -m tools.team_universe rank --out /private/durable/team-universe \
  --family-weights /private/durable/site-weights.json --audit /private/durable/reviewed-audit.json
```

Omitting `--audit` produces no qualified team. CLI output contains counts, digests and stable codes
only; names, domains, contact routes, quotes and the audit reference remain in private files.
The ranked output records the overlay SHA, reference, current scope digest and assessment date.
Summary readback returns `stale_snapshot` with no tier counts if source scope/rules or date changed;
recompute offline with the current reviewed overlay before treating counts as current.
Do not invoke `verify` to requalify retained
screens: it may read missing pages. This rank path recomputes retained records offline.

Future discovery/screen prompts need a separately versioned physical-offering and interface schema.
They must ask for the concrete robot, physical action, currently offered control/deployment relation,
dated task evidence and explicit interface/controller proof; unknown remains unknown. Do not change
v1 paid-input hashes underneath the existing receipt journal or infer these facts from v1 booleans.
