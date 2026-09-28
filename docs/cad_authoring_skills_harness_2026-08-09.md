# CAD authoring skills installed into the agent harness (2026-08-09)

Per explicit user direction on 2026-08-09, the agents whose job is creating
CAD/SimReady objects must utilize the following harness-installed skills.

## Pinned installs

| Source | Commit | License | Installed skills |
| --- | --- | --- | --- |
| [earthtojake/text-to-cad](https://github.com/earthtojake/text-to-cad) | `4fd71ea75fbb8a80b0d7c76862e0fd73c52a8989` (2026-07-31) | MIT | `cad`, `cad-viewer`, `urdf`, `sdf`, `srdf`, `step-parts`, `implicit-cad`, `dxf`, `gcode` |
| [Pan-Chera/Multi-Agent-CAD](https://github.com/Pan-Chera/Multi-Agent-CAD) | `42737c408534e7c00c63081d73ce7565a9464e56` (2026-08-09) | MIT | `multi-agent-cad` (thin wrapper skill over the pinned clone; build123d kernel) |

- Developer skill directories: `~/.claude/skills/<name>/` and
  `~/.agents/skills/<name>/`.
- Developer pinned source clones:
  `~/workspace/cad-skills/{text-to-cad,Multi-Agent-CAD}`. These paths are not a
  production dependency.
- Production deployment provisions the same commits beneath
  `/var/lib/blueprint/task-evaluation-inputs/sources/cad-authoring`, verifies
  commit, tree, clean status, license digest, and required skill files, and
  writes an immutable `production_cad_skill_sources.v1` receipt.
- Every scene-configuration Content Agents release packages the exact two
  source archives plus their digest-bound source receipt. The provider worker
  validates and extracts them under its released runtime, exposes the root as
  `BLUEPRINT_PRODUCTION_CAD_SKILLS_ROOT`, and refuses the stage before model
  spend if an archive or identity differs.
- The checked-in `skillpacks/cad_authoring` manifest publishes all ten names
  to both supported production agent layouts during normal runtime skill sync.
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
- Merely having the skills in the runtime does not qualify an asset. A
  production stage that authors a replacement or passive destination must
  invoke the sealed source, retain its program/parameters/kernel/export
  evidence, and pass the same independent rights, static, native-import,
  geometry, and scene-placement gates.
