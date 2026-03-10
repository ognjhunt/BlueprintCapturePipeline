# Standards Retriever

Use when blocker categories, capability checks, or review findings need to be grounded in curated industrial standards and guidance references. This skill matches findings to the local reference corpus and returns applicable citations. It does not browse the web, make legal determinations, or provide regulatory approval.

---

## Trigger

- After blocker_taxonomist produces the blocker register.
- When any review skill needs standards context for a finding.
- When readiness_report_writer needs citations for the report.
- When oem_handoff_writer needs standards references for the handoff package.

---

## Exact inputs

| Artifact | Required | Fields to inspect |
|---|---|---|
| `blocker_register.json` | Yes | `primary_category`, `secondary_categories`, `zone`, `severity` for each blocker |
| `site_intake.json` | Yes | `environment_type`, `zone`, `traffic_notes`, `vehicle_types` |
| `capability_envelope.json` | If available | Capability domain status values |
| `references/curated_standards.json` | Yes | Full corpus |

---

## Required behavior

### 1. Match blockers to standards

For each blocker in the register, search `curated_standards.json` for entries where:
- Entry `categories` overlap with blocker `primary_category` or `secondary_categories`.
- Entry `applicability` matches the site context (environment type, zone type).
- Entry `site_types` (if present) includes the current `environment_type`.

Return all matching entries, ranked by relevance (category match count x applicability match).

### 2. Match capability domains to standards

For capability checks with status `fail` or `conditional`:
- Match the capability domain (route_clearance, reach, manipulation, force, etc.) to standards entries.
- Include dimensional thresholds from the standards where available.

### 3. Format citations

Each returned reference must include:
- `reference_id`: From curated_standards.json.
- `title`: Human-readable title.
- `standard_number`: ISO/OSHA/ANSI number if applicable.
- `summary`: What the standard says (1-2 sentences).
- `applicability_note`: Why this standard applies to this specific blocker/finding.
- `dimensional_values`: Any specific numbers (clearances, forces, speeds) from the standard.
- `limitation_note`: What this standard does NOT cover or where it stops being applicable.

### 4. Flag gaps in the reference corpus

If a blocker category or finding type has NO matching entry in the curated corpus:
- Return `no_matching_reference` with the blocker category.
- Recommend adding a reference entry for this category.

### 5. Apply standard hierarchy

When multiple standards apply, present them in this priority order:
1. **ISO standards** (international, highest authority for industrial robots).
2. **ANSI/RIA standards** (US national adoption, often mirrors ISO).
3. **OSHA requirements** (US regulatory, enforceable).
4. **Industry guidance** (best practices, manufacturer recommendations).
5. **Blueprint internal guidance** (project-specific curated notes).

Note when a standard is in draft form (e.g., ISO 25785-1) and cannot yet be cited as binding.

---

## Standards knowledge embedded in this skill

The following standards are the core references for industrial humanoid site qualification. This skill should know these even if the curated corpus is incomplete:

### Route clearance and traffic
| Standard | Key requirement | Category match |
|---|---|---|
| ISO 3691-4:2023 | 0.5 m clearance for 2.1 m height on each side of vehicle path. 1.2 m/s critical speed threshold. | geometry_clearance, traffic_shared |
| OSHA 1910.176 | "Sufficient safe clearance" for aisles (no specific number). | geometry_clearance |
| ANSI/ITSDF B56.1 | 3 ft wider than largest equipment in aisle. | traffic_shared |

### Collaborative safety and force limits
| Standard | Key requirement | Category match |
|---|---|---|
| ISO 10218-1/2:2025 | Now allows mobile platforms. TS 15066 integrated. Risk-based PL per safety function. | safety_force, safety_guarding |
| ISO/TS 15066:2016 | Biomechanical force limits per body region. SSM separation distance formula. 250 mm/s collaborative speed recommendation. | safety_force |
| ANSI/RIA R15.08 | Type C (mobile manipulator) safety requirements. | safety_force, traffic_shared |

### Humanoid-specific safety
| Standard | Key requirement | Category match |
|---|---|---|
| ISO 25785-1 (DRAFT) | Dynamically stable robot safety. Fall zone calculations. | safety_fall, safety_estop |
| ISO 13482:2014 | Personal care robot safety — covers mobile servant robots. | safety_force |

### E-stop and energy control
| Standard | Key requirement | Category match |
|---|---|---|
| IEC 60204-1 | E-stop categories 0 and 1. Category 0 = immediate power removal. | safety_estop |
| ISO 13850 | E-stop design requirements. | safety_estop |
| OSHA 29 CFR 1910.147 | LOTO for hazardous energy control. | loto_maintenance |

### Floor and facility
| Standard | Key requirement | Category match |
|---|---|---|
| ACI 117 | F-number system. FF50 recommended for AMR, FF25 minimum for bipedal. | geometry_floor |

### Functional safety
| Standard | Key requirement | Category match |
|---|---|---|
| ISO 13849-1 | Performance Levels PLa-PLe. PLd minimum for AGV/AMR safety functions. | safety_guarding |
| IEC 61508 | Safety Integrity Levels SIL1-SIL4. | safety_guarding |
| IEC 62443 | Industrial cybersecurity — required by ISO 10218:2025. | systems_integration |

---

## Output

`standards_notes.json` with:
- Per-blocker matched references.
- Per-capability-domain matched references.
- Reference gap list (categories with no matching corpus entry).
- Standards hierarchy notes where multiple standards apply.

---

## Do not

- Present guidance as regulatory approval. Standards are informational context, not compliance certification.
- Make legal conclusions. "ISO 3691-4 requires 0.5 m clearance" is factual. "This aisle is in violation" is a legal conclusion.
- Return uncited advice. Every statement must trace to a specific standard or guidance entry.
- Browse the web. Use the local curated corpus only. Flag gaps for corpus updates.
- Present draft standards (ISO 25785-1) as binding requirements. Note their draft status.
- Assume US standards apply globally — note jurisdiction when relevant.

---

## Fail-closed rules

- If `curated_standards.json` is missing or empty: return error. Cannot provide standards context without corpus.
- If a blocker has a `safety_*` category and no matching standard exists in the corpus: flag as `critical_reference_gap` — the corpus must be updated before this qualification can proceed.
- If the environment type is not recognized: use the broadest applicable standards and flag for corpus expansion.

---

## Escalation rules

- If multiple conflicting standards apply to the same finding: present all with their hierarchy position and flag for human interpretation.
- If a standard is jurisdiction-specific and the site jurisdiction is unknown: flag for clarification.
- All `critical_reference_gap` findings must appear in `human_actions_required.json`.
