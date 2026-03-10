# Warehouse Site Knowledge

Use when qualification artifacts describe a warehouse, distribution center, fulfillment center, logistics hub, or dock environment. This skill provides domain-specific knowledge that other skills reference to make warehouse-grounded qualification decisions.

This is a knowledge skill. It does not produce a standalone output artifact. Other skills reference it to ground blocker classification, capability checks, route access, and recapture planning in warehouse-specific reality.

---

## Trigger

- `site_intake.json` field `environment_type` or `site_type` contains: warehouse, distribution_center, fulfillment, logistics, dock, cross_dock, cold_storage, sortation.
- Any review skill encounters warehouse-specific zones, traffic, or operational patterns.

---

## Warehouse Zone Types

These are the standard functional zones in a warehouse. Each has distinct humanoid readiness implications.

### Receiving / Inbound Dock
- **Typical layout:** Dock doors (8-10 ft wide), staging lanes, conveyor induction points.
- **Traffic:** Forklifts, pallet jacks, dock workers, sometimes trucks backing in.
- **Humanoid concerns:** High traffic density, uncontrolled vehicle movement near dock edges, variable lighting (doors open/closed), wet/slippery floors near dock plates, dock leveler gaps (fall hazard for bipedal). Dock plate transitions create step changes of 1-4 inches.
- **Readiness implications:** Shared traffic zone — requires explicit traffic management protocol evidence. Dock plate transitions may exceed humanoid step tolerance. Lighting variability must be verified.

### Bulk Storage / Racking Aisles
- **Aisle types and widths:**
  - Conventional (wide) aisles: 12-13+ ft (3.7-4.0 m) — forklift two-way traffic
  - Narrow aisles: 8-10 ft (2.4-3.0 m) — reach truck single-direction
  - Very Narrow Aisles (VNA): 5-7 ft (1.5-2.1 m) — turret truck only, no pedestrians
- **Humanoid concerns:** VNA aisles may be too narrow for humanoid + safety clearance. Narrow aisles with reach trucks create single-occupancy constraint — humanoid and reach truck cannot coexist. Racking creates overhead falling-object hazard. Floor must support humanoid weight + payload (typically not an issue in racking zones designed for forklifts).
- **Readiness implications:** Aisle width must be measured, not inferred from video. VNA zones are likely exclusion zones for humanoids. Narrow aisles require traffic protocol evidence.

### Pick Zones / Each-Pick Areas
- **Typical layout:** Pick modules, shelving, flow racks, mezzanines, conveyor takeaway.
- **Traffic:** Pedestrians, carts, sometimes AMRs.
- **Humanoid concerns:** This is the most likely deployment zone for humanoid each-picking. Shelving height (typically 6-8 ft) within humanoid reach. Pick face accessibility, bin/tote orientation, barcode/label visibility. Mezzanine floors may have different load ratings and vibration characteristics.
- **Readiness implications:** Strongest humanoid fit for current capabilities (tote handling, simple pick-and-place). Must verify: shelf depth, pick face angle, bin lip height, conveyor induction height, label position.

### Sortation
- **Typical layout:** High-speed conveyors, divert points, chutes, accumulation lanes.
- **Traffic:** Primarily automated — limited pedestrian access during operation.
- **Humanoid concerns:** High-speed conveyor proximity is a pinch/strike hazard. Divert mechanisms create unpredictable object trajectories. Noise levels may interfere with audible safety signals.
- **Readiness implications:** Sortation zones near active conveyors likely require safety fencing or monitored zones. Not a primary humanoid deployment zone unless explicitly scoped for jam clearing or exception handling.

### Packing / Value-Added Services
- **Typical layout:** Pack stations, workbenches, tape/label machines, shipping conveyors.
- **Traffic:** Pedestrians, hand carts.
- **Humanoid concerns:** Fine manipulation (tape, labels, inserts) may exceed current humanoid dexterity. Workstation ergonomics designed for seated humans may not fit standing humanoids. Station spacing (typically 3-4 ft between stations) may be tight.
- **Readiness implications:** High dexterity requirement — verify against platform hand DOF. Station clearance must be measured. Task decomposition must separate humanoid-feasible from human-only subtasks.

### Shipping / Outbound Dock
- Same concerns as Receiving/Inbound Dock, plus:
- Staging lanes with time-critical trailer loading.
- Pallet build zones with stacking height requirements (humanoid reach limit ~1.8-2.0 m for overhead placement).

### Charging / Staging Area (for robots)
- **Humanoid concerns:** Charging infrastructure must be planned. Current humanoids need 1.5-5 hour charges for 4-5 hour runtime. Hot-swap battery option (Apollo) changes infrastructure needs. Floor space for charging stations, cable routing, fire suppression for lithium-ion batteries.
- **Readiness implications:** If intake does not include charging infrastructure plan, flag as a blocker for sustained operation qualification.

---

## Warehouse-Specific Clearance Requirements

| Check | Minimum | Source | Notes |
|---|---|---|---|
| Aisle width for humanoid passage | Platform width + 1.0 m (0.5 m each side) | ISO 3691-4 operating zone | 0.5 m clearance for 2.1 m height on each side |
| Overhead clearance | Platform height + 0.3 m | General practice | Must clear with payload raised |
| Doorway width | Platform width + 0.6 m | ADA/building code baseline | Standard doors 0.9 m usually sufficient |
| Doorway height | Platform height + 0.15 m | General practice | Standard doors 2.0 m usually sufficient |
| Dock plate transition | < 50 mm step, < 5 degree slope | Humanoid locomotion limits | Larger transitions are blockers |
| Floor flatness | FF25 minimum, FF50 preferred | ACI 117 F-number system | Bipedal stability more sensitive than wheeled AMR |
| Floor condition | No standing water, oil, loose debris | General safety | Bipedal slip risk higher than wheeled |

---

## Warehouse-Specific Blocker Categories

These extend the base blocker taxonomy for warehouse environments:

| Blocker category | Examples | Severity |
|---|---|---|
| `dock_transition` | Dock plates, leveler gaps, ramp slopes > 5 degrees | high |
| `vna_exclusion` | VNA aisles < 1.5 m width | high |
| `forklift_shared_aisle` | Active forklift traffic in proposed humanoid route | high |
| `reach_truck_conflict` | Narrow aisle reach truck zones with no traffic separation | high |
| `floor_condition` | Oil, water, debris, damaged concrete, low F-number | medium-high |
| `overhead_hazard` | Falling object risk from racking, unsecured loads above | medium |
| `lighting_variability` | Dock areas with variable natural light, dimly lit aisles | medium |
| `conveyor_proximity` | Active conveyor within 1.0 m of proposed humanoid path/station | medium |
| `charging_infrastructure` | No charging plan for sustained operations | medium |
| `cold_storage` | Temperatures below 5 C, condensation, ice on floors | high |
| `mezzanine_access` | Stairs, platform edge fall hazard, load rating unknown | high |

---

## Common Warehouse Deployment Patterns (from real pilots)

### Pattern: Tote Handling (GXO/Digit model)
- AMR brings totes to fixed station → humanoid picks totes from AMR → places on conveyor.
- **What worked:** Repetitive, predictable task in controlled zone. No forklift interaction. Fixed station eliminates navigation complexity.
- **What to verify:** Station height, tote weight range, AMR arrival cadence, conveyor induction height, tote grip geometry.

### Pattern: Pick-and-Place from Shelving
- Humanoid navigates pick zone → picks items from shelves → places in tote/conveyor.
- **What worked (limited pilots):** Simple, same-height picks with clear pick faces.
- **What to verify:** Shelf depth, bin lip height, pick face angle, item weight/size distribution, barcode visibility, path width between pick aisles.

### Pattern: Depalletization
- Humanoid picks cases/items from pallet → places on conveyor or into storage.
- **What worked:** Limited success — heavy cases exceed current payloads (16-25 kg). Top-of-pallet access only (no reaching into pallet center).
- **What to verify:** Case weight distribution, pallet height, stretch wrap handling, pallet condition variability.

---

## Do not

- Assume warehouse type from video alone — require explicit zone identification in intake.
- Treat all aisles as equivalent — width and traffic type vary dramatically.
- Assume floor condition from visual appearance — F-number measurement is the standard.
- Clear dock zones for humanoid access without explicit traffic management evidence.
- Assume cold storage is equivalent to ambient warehouse.

---

## Usage by other skills

- **blocker_taxonomist**: Use warehouse-specific blocker categories to extend base taxonomy.
- **capability_envelope_writer**: Use clearance values and zone types to bound capability claims.
- **humanoid_route_access_reviewer**: Use aisle width data and zone constraints for route checks.
- **humanoid_site_readiness_reviewer**: Use zone type classification for site-level review.
- **recapture_planner**: Use zone types to prioritize which areas need metric recapture.
