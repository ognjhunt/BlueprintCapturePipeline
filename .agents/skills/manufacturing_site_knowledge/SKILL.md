# Manufacturing Site Knowledge

Use when qualification artifacts describe a manufacturing facility, assembly line, brownfield industrial site, machine shop, fabrication area, or production cell environment. This skill provides domain-specific knowledge that other skills reference to make manufacturing-grounded qualification decisions.

This is a knowledge skill. It does not produce a standalone output artifact.

---

## Trigger

- `site_intake.json` field `environment_type` or `site_type` contains: manufacturing, assembly, production, fabrication, machine_shop, brownfield, industrial, plant, cell, line.
- Any review skill encounters manufacturing-specific zones, equipment, or operational patterns.

---

## Manufacturing Zone Types

### Assembly Line / Production Line
- **Typical layout:** Sequential stations, conveyor transport between stations, fixed tooling, parts staging.
- **Traffic:** Workers move between stations, forklifts deliver parts to line-side. Automated guided vehicles (AGVs) may serve kitting.
- **Humanoid concerns:** Cycle time requirements are strict (seconds to minutes per station). Humanoid task execution speed must match line takt time or the cell goes idle. Station spacing is optimized for human operators — may be tight for humanoid footprint + balance envelope. Overhead tooling (torque drivers, air lines, welding equipment) creates head/arm collision risk.
- **Readiness implications:** Cycle time evidence must exist in `task_scope_record.json`. Station clearance must be measured. Overhead obstruction map required. Takt time misMatch is a hard blocker.

### Manufacturing Cell / Workcell
- **Typical layout:** Machine tool(s), fixture(s), part staging, chip/coolant management, quality check station.
- **Traffic:** Cell operator, maintenance, occasionally forklift for material delivery.
- **Humanoid concerns:** Machine tool interactions (load/unload chuck, press buttons, close doors) require specific reach geometry and force. Coolant/oil on floor is a slip hazard. Chip debris creates puncture risk for soft robot components. Machine doors may require force > humanoid payload to operate. Fixture clamping may require precise force control.
- **Readiness implications:** Machine interface geometry must be captured in detail — button positions, door handle heights, chuck dimensions, fixture locations. Floor condition in cells is often worse than general floor (coolant, chips). Machine guarding must be evaluated for humanoid access compatibility.

### Kitting / Lineside Supply
- **Typical layout:** Kitting stations, shelving, flow racks, tow trains, supermarket areas.
- **Traffic:** Tow trains, forklifts, pedestrians.
- **Humanoid concerns:** Kitting requires picking multiple parts into a kit container — moderate dexterity requirement. Tow train traffic is periodic but fast-moving. Parts variety (bolts, brackets, hoses, connectors) may exceed gripper capability.
- **Readiness implications:** Parts catalog with weight/size distribution needed. Tow train schedule and path overlap with humanoid routes must be verified.

### Quality / Inspection
- **Typical layout:** CMM rooms, visual inspection stations, go/no-go gauges, measurement fixtures.
- **Humanoid concerns:** Quality inspection tasks often require fine visual discrimination and precise tool handling that exceeds current humanoid sensor resolution and dexterity. CMM rooms are climate-controlled — humanoid thermal output may be a concern.
- **Readiness implications:** Most quality tasks are beyond current humanoid capability. Exception: simple visual inspection (presence/absence checks) or part transport to/from inspection stations.

### Maintenance / Tool Crib
- **Humanoid concerns:** Not a primary deployment zone. High variability, unpredictable layouts, hazardous stored energy.
- **Readiness implications:** Exclude from qualification scope unless explicitly requested.

### Dock / Shipping (Manufacturing)
- Similar to warehouse dock concerns, plus:
- Manufacturing docks often handle heavier, irregularly shaped parts.
- Packaging materials (crates, foam, stretch wrap) require manipulation beyond simple tote handling.

---

## Brownfield-Specific Qualification Concerns

Brownfield sites (existing facilities being retrofitted) have unique challenges vs. greenfield:

### Infrastructure
| Check | Requirement | Common blocker |
|---|---|---|
| Electrical capacity | 480V 3-phase for automation infrastructure | Older facilities may lack capacity |
| Network latency | < 20 ms for real-time humanoid ops, < 10 ms for multi-robot coordination | Metal structures cause WiFi dead zones |
| Network bandwidth | ~200 Mbps per robot for sensor streaming | Shared network may be saturated |
| Floor condition | FF50 preferred, FF25 minimum | Older industrial floors often degraded |
| Floor load capacity | 250 lbs/sq ft minimum for heavy automation zones | Usually met in manufacturing |
| Ceiling height | Platform height + manipulation envelope + 0.3 m | Low ceilings in older facilities |
| Column spacing | Irregular in brownfield — creates dead zones and path constraints | Cannot be changed |
| Doorway dimensions | Min 0.9 m wide x 2.0 m tall | Industrial doors usually wider, but may have thresholds |

### Layout Constraints
- Structural columns create irregular zones that cannot be repurposed.
- Legacy machine foundations create floor elevation changes.
- Utility runs (compressed air, hydraulic, electrical conduit) at head height create overhead hazards.
- Legacy paint/striping may conflict with robot navigation markers.

### Production Continuity
- Phased deployment required — cannot shut down production for full-facility retrofit.
- Zone-by-zone qualification is the practical approach.
- Existing automation (PLCs, conveyors, safety systems) must be preserved during integration.

---

## Manufacturing-Specific Blocker Categories

| Blocker category | Examples | Severity |
|---|---|---|
| `takt_time_mismatch` | Humanoid cycle time exceeds station takt time | hard_blocker |
| `machine_interface_unknown` | Button/door/fixture geometry not captured | high |
| `coolant_floor_hazard` | Coolant, oil, or cutting fluid on floor in work zone | high |
| `chip_debris` | Metal chips, shavings, or abrasive debris in work zone | medium-high |
| `overhead_obstruction` | Tooling, air lines, conduit at head/arm height | high |
| `legacy_guarding_conflict` | Existing safety fencing/light curtains incompatible with humanoid access | high |
| `power_infrastructure` | Insufficient electrical capacity for charging/automation | medium |
| `network_coverage` | WiFi dead zones in metal-structure areas | medium |
| `column_dead_zone` | Structural columns blocking path or creating blind spots | medium |
| `floor_elevation_change` | Machine foundations, floor patches, expansion joints > 25 mm | high |
| `force_requirement_exceeded` | Machine door/fixture requires force > platform payload | hard_blocker |
| `hazardous_atmosphere` | Welding fumes, paint spray, chemical exposure in zone | hard_blocker |
| `thermal_environment` | Heat treatment areas, furnace proximity, extreme ambient temperature | high |

---

## Manufacturing Deployment Patterns (from real pilots)

### Pattern: Sheet Metal Loading (Figure 02/BMW model)
- Humanoid picks sheet metal part from staging → loads into press/fixture with 5 mm tolerance.
- **What worked:** Demonstrated 90,000+ parts over 11 months. 2-second placement cycles. 37-second load time within 84-second total cycle. All cabling internal to limbs (no external routing to manage).
- **What to verify:** Part weight/size, fixture geometry, cycle time requirement, staging arrangement, press interface clearances.
- **Key learning:** This is currently the strongest evidence of manufacturing humanoid viability. But it was a single, highly controlled station on a major OEM line with dedicated integration support.

### Pattern: Assembly Kit Delivery (Apollo/Mercedes-Benz model)
- Humanoid retrieves kit from staging → delivers to line-side station.
- **Status:** Pilot phase. Limited published results.
- **What to verify:** Kit weight, delivery distance, path obstacles, delivery window (takt-synchronized).

### Pattern: Machine Tending (prospective)
- Humanoid loads/unloads machine tool (CNC, press, injection mold).
- **Status:** No confirmed production deployments as of 2026-03.
- **What to verify:** Machine door operation force, chuck/fixture interface geometry, part orientation requirements, coolant management, chip clearing.
- **Key concern:** Machine tending requires precise timing coordination with machine cycle. Current humanoid autonomy may not reliably meet this.

---

## Real Brownfield Integration Learnings

From Agility Robotics field deployments:
- "We don't have control over the floor finish or the lighting" — these are variables that must be tested on-site, not inferred from specs.
- Rubber feet and vision systems require field tuning in each facility.
- Integration took dedicated Agility specialists on-site.

From Figure AI BMW deployment:
- Tool used by six-axis arm can be redeployed to humanoid — tool modularity is viable.
- Tight cycle times (2-second placements) are achievable but require precise calibration.

From general brownfield integration:
- Budget significant resources for safety systems, software integration, and change management beyond hardware costs.
- Start with simple, repetitive tasks in controlled environments.
- Phased rollout: pilot learnings inform next deployment wave.
- Production downtime risk is the #1 concern for manufacturing operators.

---

## Do not

- Assume manufacturing environments are static — production layouts change.
- Treat machine interface geometry as known without explicit measurement evidence.
- Clear cells with active coolant/chips without verifying floor management.
- Assume existing safety guarding is compatible with humanoid access.
- Ignore takt time constraints — a humanoid that cannot keep up stops the line.
- Treat brownfield as greenfield — existing infrastructure constraints are real.

---

## Usage by other skills

- **blocker_taxonomist**: Use manufacturing-specific blocker categories.
- **capability_envelope_writer**: Use machine interface and cycle time data for capability bounds.
- **humanoid_workcell_risk_reviewer**: Use cell layout, machine interface, and floor condition knowledge.
- **humanoid_site_readiness_reviewer**: Use zone classification and brownfield constraints.
- **recapture_planner**: Prioritize machine interface and floor condition captures.
