---
name: multi-agent-cad
description: Run the pinned Multi-Agent CAD build123d pipeline for reproducible development-only CAD candidates.
---

# Production Multi-Agent CAD

Use `Pan-Chera/Multi-Agent-CAD` at exact commit
`42737c408534e7c00c63081d73ce7565a9464e56`, packaged by the production
scene-configuration release. Read `multi_agent_cad/WORKFLOW.md` and `README.md`
from that sealed source before execution. Retain the generated build123d
program, parameters, kernel/environment identity, and STEP/STL exports together.
The candidate is `development_only`; independent deterministic static/native
validators and a component admission receipt remain mandatory.

The sealed source root is
`${BLUEPRINT_PRODUCTION_CAD_SKILLS_ROOT}/Multi-Agent-CAD` in a provider
component, or
`/var/lib/blueprint/task-evaluation-inputs/sources/cad-authoring/multi-agent-cad-42737c40`
on the production control plane.
