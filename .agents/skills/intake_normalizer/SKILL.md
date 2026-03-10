# Intake Normalizer

Use when a qualification run needs structured workflow, zone, owner, systems, and success-criteria normalization before any readiness analysis begins. This is always the first skill in the pipeline. Nothing downstream should execute on unnormalized intake.

---

## Trigger

- A new `site_intake.json` is submitted for qualification.
- A previously normalized intake is resubmitted after recapture or scope change.
- Any downstream skill detects missing normalized intake fields.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `site_intake.json` | Yes | All fields below |
| `capture_package_manifest.json` | Yes | `capture_type`, `modalities`, `capture_date`, `capture_duration`, `device_info` |

---

## Required fields in `site_intake.json`

The following fields are **mandatory**. If any is missing, the normalizer must fail closed and return a structured error listing every missing field.

### Hard-required (fail if missing)

| Field | Type | Purpose |
|---|---|---|
| `workflow` | string | Plain-language description of the target workflow the humanoid would perform. Must be specific enough to scope qualification. "Pick and place" is insufficient. "Pick totes from AMR arrival station and place on outbound conveyor belt B3" is sufficient. |
| `zone` | string or object | Physical zone identifier. Must include zone name/ID and zone type (e.g., "Pick Module 3, each-pick zone"). A bare room name ("warehouse") is insufficient. |
| `success_criteria` | array of strings | What constitutes a successful qualification outcome. Must be measurable or verifiable. "Ready for humanoid" is insufficient. "Route clearance verified for all segments, no hard blockers, all manipulation targets within reach envelope" is sufficient. |
| `environment_type` | string | One of: warehouse, distribution_center, fulfillment, manufacturing, assembly, production, brownfield, dock, cold_storage, or a specific variant. This drives site-type knowledge skill selection. |
| `owner` | string | Person or role responsible for the qualification decision. |

### Soft-required (warn if missing, do not fail)

| Field | Type | Purpose |
|---|---|---|
| `systems` | array of strings | Connected systems (WMS, WES, PLC, conveyor controls, etc.) |
| `non_routine_modes` | array of strings | Known exception conditions, jam scenarios, mode changes |
| `traffic_notes` | string or array | Vehicle types, traffic patterns, shared zones |
| `vehicle_types` | array of strings | Forklifts, reach trucks, AGVs, AMRs, tow trains, pallet jacks |
| `shared_zones` | array of objects | Zones where humanoid shares space with other traffic |
| `privacy_limits` | string | Capture restrictions (faces, proprietary equipment, etc.) |
| `security_limits` | string | Physical security constraints (badge access, escorted zones) |
| `known_blockers` | array of objects | Issues already known before capture |
| `target_platform` | string | If a specific humanoid platform is targeted (e.g., "Digit", "Figure 02", "Apollo") |
| `shift_info` | object | Shift times, staffing levels, humanoid operating window |
| `floor_condition_notes` | string | Known floor issues (cracks, oil, slope, coating) |
| `charging_infrastructure` | string | Planned or existing charging setup |
| `estop_plan` | string | E-stop approach for the humanoid deployment |

---

## Required behavior

### 1. Validate hard-required fields

Check each hard-required field exists and is non-empty. If any is missing:
- Set `normalization_status` to `failed`.
- Return structured error with field name, requirement description, and example of acceptable input.
- Do NOT proceed to normalization.

### 2. Validate field quality

Even if fields exist, check for insufficient specificity:
- `workflow` must contain at least a verb, an object, and a location. "Pick and place" fails. "Pick totes from shelf A3 and place on conveyor C1" passes.
- `zone` must contain a specific identifier, not just a building name.
- `success_criteria` must contain at least one measurable criterion.

If field quality is insufficient, set `normalization_status` to `needs_clarification` and list specific deficiencies.

### 3. Normalize environment type

Map the `environment_type` to the canonical set. This determines which site knowledge skill applies:
- warehouse / distribution_center / fulfillment / logistics / dock / cross_dock / cold_storage / sortation -> `warehouse_site_knowledge`
- manufacturing / assembly / production / fabrication / machine_shop / brownfield / industrial / plant / cell / line -> `manufacturing_site_knowledge`
- If ambiguous or mixed, flag as `mixed_environment` and require both knowledge skills.

### 4. Normalize capture modality

From `capture_package_manifest.json`, classify the capture as:
- `video_only`: Only video captures, no structured scan or metric measurement.
- `video_plus_splat`: Video plus 3DGS/splat reconstruction.
- `metric_scan`: LiDAR, structured light, or photogrammetry with metric calibration.
- `metric_plus_splat`: Both metric scan and splat/3DGS.
- `full_capture`: Metric scan + splat + structured intake + QA.

**Critical rule:** `video_only` and `video_plus_splat` can support pre-screening only. They must NOT silently become decision-grade evidence. Set `evidence_grade` to `pre_screen` for these modalities.

### 5. Warn on missing soft-required fields

For each missing soft-required field, generate a warning with:
- Field name.
- Why it matters for humanoid qualification.
- What downstream skill will be limited without it.

### 6. Normalize known blockers

If `known_blockers` exists, validate each entry has:
- `description`: What the blocker is.
- `category`: One of the blocker taxonomy categories.
- `source`: Who reported it (site owner, capture operator, etc.).
- `severity`: Initial severity estimate.

If blockers exist but lack structure, normalize them. Do not discard unstructured blocker notes — convert them to structured entries with `source: "intake_raw"`.

### 7. Set normalization timestamp and status

Output must include:
- `normalization_timestamp`: ISO 8601.
- `normalization_status`: `passed`, `failed`, `needs_clarification`, or `passed_with_warnings`.
- `evidence_grade`: `pre_screen`, `qualification_ready`, or `insufficient`.
- `site_knowledge_skill`: which site knowledge skill applies.

---

## Output

Updated `site_intake.json` (or a `normalized_intake.json` overlay) with:
- All validated and normalized fields.
- `normalization_status`.
- `evidence_grade`.
- `site_knowledge_skill`.
- `warnings[]` for missing soft-required fields.
- `errors[]` for hard failures.

---

## Do not

- Make a readiness judgment. This skill normalizes input only.
- Invent missing workflow details. If the workflow is vague, fail with a specific error.
- Treat a splat or geometry artifact as a substitute for structured intake.
- Silently pass video-only evidence as decision-grade.
- Normalize away uncertainty — if something is unclear, keep it unclear and flag it.
- Accept "ready for humanoid" as a success criterion — it is circular.

---

## Fail-closed rules

- Missing `workflow`: hard fail. Cannot qualify without knowing what the humanoid will do.
- Missing `zone`: hard fail. Cannot qualify without knowing where.
- Missing `success_criteria`: hard fail. Cannot qualify without knowing what "done" means.
- Missing `environment_type`: hard fail. Cannot select site knowledge skill.
- Missing `owner`: hard fail. Cannot route qualification decisions.
- Vague `workflow` (no verb + object + location): needs_clarification.
- `capture_type` = video_only with no structured intake: `evidence_grade` = `pre_screen`. Downstream skills must honor this grade.

---

## Escalation rules

- If `known_blockers` includes anything related to safety, EHS, or regulatory: escalate immediately to `human_actions_required.json` with action type `safety_review_required`.
- If `environment_type` includes cold_storage or hazardous_atmosphere: escalate to `human_actions_required.json` with action type `environmental_review_required`.
- If `estop_plan` is missing and the scope includes shared spaces: warn that non_routine_operations_reviewer will flag this as a mandatory blocker.
