# CAD authoring skills installed into the agent harness (2026-08-09)

Per explicit user direction on 2026-08-09, the agents whose job is creating
CAD/SimReady objects must utilize the following harness-installed skills.

## Pinned installs

| Source | Commit | License | Installed skills |
| --- | --- | --- | --- |
| [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad) | `4fd71ea75fbb8a80b0d7c76862e0fd73c52a8989` (2026-07-31) | MIT | `cad`, `cad-viewer`, `urdf`, `sdf`, `srdf`, `step-parts`, `implicit-cad`, `dxf`, `gcode` |
| [Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) | `42737c408534e7c00c63081d73ce7565a9464e56` (2026-08-09) | MIT | `multi-agent-cad` (thin wrapper skill over the pinned clone; build123d kernel) |

- Skill directories: `~/.claude/skills/<name>/` (Claude Code user-level harness).
- Pinned source clones: `~/workspace/cad-skills/{text-to-cad,Multi-Agent-CAD}`.
- Deliberately not installed: `bambu-labs`, `sendcutsend` (external
  print/fabrication ordering services; outside this program's scope).

## Usage contract for CAD/SimReady authoring agents

- Use these skills for candidate/parametric geometry authoring, inspection,
  robot-description (URDF/SDF/SRDF) work, and format handoff during SimReady
  object construction.
- Always retain the generating script, parameters, kernel identity, and the
  exported artifact together; a mesh without its generator is not reproducible
  evidence.
- Claim boundary (binding, from `docs/arm_decision_proof_v1/north_star_contract.json`
  `asset_authoring_contract`): generated geometry is a `development_only`
  candidate or proposal. Geometry authority for ADP evidence remains measured
  scan / manufacturer CAD / checked-in parametric geometry with independent
  deterministic verification — never VLM inference. Observed source-derived
  geometry (for scene `840796`, the registered SAGE-derived articulated source
  asset) outranks any generated candidate for exterior surfaces; unobserved
  surfaces (interiors) stay labeled `generated_candidate_geometry`.
- Entering the ADP evidence chain (for example as a SimReady authoring
  comparison lane next to NVIDIA USD Content Agents) additionally requires an
  exact component admission packet under the released-code rule
  (`docs/arm_decision_proof_v1/PUBLIC_EVIDENCE_LADDER.md`) and passage of the
  independent static/native validators
  (`src/blueprint_pipeline/articulated_simready_replacement.py`,
  `public_scene_simready_*`).
