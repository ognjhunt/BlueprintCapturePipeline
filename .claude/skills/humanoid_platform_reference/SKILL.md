# Humanoid Platform Reference

Use when any qualification skill needs humanoid dimensional data, payload limits, sensor capabilities, deployment history, or known failure modes to ground a readiness check against a specific platform or platform class.

This is a knowledge skill. It does not produce a standalone output artifact. Other skills reference it to bound capability claims, reach envelopes, route clearance checks, and OEM handoff context.

---

## Trigger

- Any skill needs to compare site geometry against a humanoid envelope.
- capability_envelope_writer needs reach, height, width, or payload bounds.
- humanoid_route_access_reviewer needs platform width, height, or turning radius.
- humanoid_workcell_risk_reviewer needs manipulation DOF, grip force, or sensor FOV.
- oem_handoff_writer needs to reference platform-specific constraints.
- The intake specifies a target platform or platform class.

---

## Platform Specifications (as of 2026-03)

### Agility Robotics — Digit

| Parameter | Value |
|---|---|
| Height | 1.75 m (5'9") |
| Weight | 65 kg (143 lbs) |
| Payload capacity | 16 kg (35 lbs) |
| Walking speed | 5 km/h (1.4 m/s) |
| Arm DOF | 4 per arm (8 total) |
| Hand type | Parallel gripper (not dexterous) |
| Sensors | 1x LiDAR, 4x Intel RealSense depth cameras, MEMS IMU |
| Field of view | 360-degree spatial awareness (multi-camera) |
| Battery life | ~4 hours |
| Fall recovery | Self-recovery from falls confirmed |
| IP rating | Not published |
| Operating temp | Indoor warehouse ambient |
| Footprint (stance) | ~0.5 m x 0.4 m estimated |
| Shoulder width | ~0.55 m estimated |

**Known deployment history:**

1. **GXO — Flowery Branch, GA (2024-2025):** 100,000+ totes moved. Tote unloading from AMR to conveyor. Peak season deployment with no service-level impact. Integration pattern: AMR arrives → Digit unloads totes → places on conveyor. Key learning: floor finish and lighting are uncontrolled variables that "can truly only be learned about when on-site." Rubber feet and vision calibration required field tuning.

2. **Toyota TMMC — Woodstock, ON (2025+):** Multi-year RaaS for logistics, supply chain, and manufacturing tasks. Details limited.

3. **Spanx — Warehouse (2024):** Tote handling in e-commerce fulfillment.

**Known limitations:**
- Gripper is not dexterous — cannot manipulate small parts, tools, or articulated objects.
- 4-DOF arms limit reach flexibility in cluttered workcells.
- 16 kg payload excludes heavy manufacturing parts.
- Walking speed (1.4 m/s) exceeds ISO 3691-4 critical threshold (1.2 m/s) — must be governed in shared zones.

---

### Figure AI — Figure 02

| Parameter | Value |
|---|---|
| Height | 1.68 m (5'6") |
| Weight | 70 kg (154 lbs) |
| Payload capacity | 25 kg (55 lbs) |
| Walking speed | 1.2 m/s (4.3 km/h) |
| Hand DOF | 16 per hand (32 total) |
| Total DOF | 41+ |
| Sensors | Multi-camera array, upgraded from Gen 1 |
| Battery | 2.25 kWh lithium-ion |
| Battery life | 5 hours continuous |
| Charge time | 1.5 hours (rapid charge) |
| Placement accuracy | 5 mm demonstrated (BMW) |

**Known deployment history:**

1. **BMW Spartanburg (2024-2025, 11 months):** Sheet-metal loading on active assembly line. 90,000+ sheet metal parts loaded with 5 mm placement tolerance in 2-second cycles. Contributed to 30,000+ BMW X3 vehicles. Cycle time: load part within 37 seconds, full task in 84 seconds. All cabling integrated into limbs — no external cable routing. Key learning: production-line integration is viable with tight cycle time requirements.

**Known safety concerns (from litigation, 2025):**
- Impact testing allegedly showed forces "twenty times higher than the threshold of pain."
- Estimated force "more than twice the force necessary to fracture an adult human skull."
- Malfunction caused robot to strike steel refrigerator door, creating 1/4-inch deep gash.
- E-Stop certification project allegedly cancelled.
- CAHS (Center for Advancement of Humanoid Safety) formed in response, led by ex-Amazon Robotics staff. Testing protocol covers: stability, human detection, pet detection, safe AI behaviors, navigation injury prevention.

**Known limitations:**
- Force output is a serious concern in shared spaces — ISO/TS 15066 force limits likely exceeded at operational speed.
- Walking speed (1.2 m/s) is exactly at the ISO 3691-4 critical threshold.
- No published IP rating for industrial environments.
- Litigation creates OEM/integrator risk — flag in handoff.

---

### Apptronik — Apollo

| Parameter | Value |
|---|---|
| Height | 1.73 m (5'8") |
| Weight | 72.6 kg (160 lbs) |
| Payload capacity | 25 kg (55 lbs) |
| Runtime | 4 hours |
| Battery | Hot-swappable, replaceable in <5 min |
| Power option | Tethered for continuous operation |
| Designed for | Brownfield deployment — no facility modifications |

**Known deployment history:**

1. **Mercedes-Benz (2025+):** Assembly kit delivery, vehicle inspection, parts movement. Pilot stage.

**Known limitations:**
- Full hand DOF not published — manipulation capability unclear for fine tasks.
- Sensor suite not fully specified in public documentation.
- Marketed as "minimal facility modifications" but no published qualification guide.

---

### 1X Technologies — NEO (NEO Gamma)

| Parameter | Value |
|---|---|
| Height | 1.68 m (5'6") |
| Weight | ~30 kg (66 lbs) — significantly lighter than competitors |
| Payload (lift) | 70 kg |
| Payload (carry) | 25 kg |
| Total DOF | 75 |
| Hand DOF | 22 per hand (44 total) |
| Walking speed | 1.4 m/s |
| Running speed | 6.2 m/s (top) |
| Battery | 842 Wh |
| Runtime | ~4 hours |
| Charge rate | 6 min per hour of runtime (quick-charge) |
| Sensors | Dual 8.85 MP / 90 Hz stereo fisheye cameras |
| Compute | NVIDIA Jetson Thor |
| Noise | 22 dB |
| IP rating | Hands IP68, body IP44 |
| Safety features | Soft 3D-lattice polymer body wrap, pinch-proof joints, low-inertia tendon drives |

**Known deployment history:**
- Primarily consumer-focused (pre-order $20,000 or $499/month). No published industrial deployments.

**Known limitations:**
- No industrial deployment track record.
- Light weight (30 kg) may limit stability under payload.
- Running speed (6.2 m/s) far exceeds any safe shared-traffic threshold — must be governed.
- Consumer focus means industrial safety certification likely absent.

---

### Tesla — Optimus (Gen 3)

| Parameter | Value |
|---|---|
| Hand DOF | 22 (doubled from Gen 2's 11) |
| Sensors | 8x autopilot cameras (360-degree), stereo depth, foot force/torque, ultrasonic proximity |
| LiDAR | None |
| Radar | None |

**Known deployment history:**
- As of Q4 2025 earnings: "No robots are doing useful work yet" — learning and data collection only.
- Gen 3 production begun at Fremont factory.
- Internal factory deployment targeted mid-2026.

**Known limitations:**
- No external deployment history.
- No published site readiness documentation.
- No published safety certification.
- Reliance on vision-only sensing (no LiDAR) may limit depth accuracy in industrial environments.

---

### Sanctuary AI — Phoenix

| Parameter | Value |
|---|---|
| Height | 1.70 m (5'7") |
| Weight | 70 kg (154 lbs) |
| Payload | 25 kg |

**Known deployment history:**
- Magna International (automotive): Pilot phase only.
- No published site readiness documentation.

---

## Cross-Platform Envelope Summary

For qualification checks, use these bounding values across all current platforms:

| Dimension | Min | Max | Use for |
|---|---|---|---|
| Height | 1.68 m | 1.75 m | Overhead clearance, doorway checks |
| Width (shoulders) | ~0.45 m | ~0.60 m | Route width, choke point clearance |
| Weight | 30 kg | 72.6 kg | Floor load, fall impact energy |
| Payload | 16 kg | 25 kg | Task feasibility, manipulation targets |
| Walking speed | 1.2 m/s | 1.4 m/s | Traffic zone speed limits |
| Hand DOF | 4 (gripper) | 22 (dexterous) | Manipulation complexity feasibility |
| Runtime | 4 hrs | 5 hrs | Shift coverage, charging infrastructure |

**Fall impact energy estimate (worst case):**
- 72.6 kg at 1.75 m CG height ≈ 1,245 J potential energy
- This is a serious personnel hazard — ISO 25785-1 (draft) will require fall zone calculations.

---

## Do not

- Recommend a specific platform for deployment.
- Use platform marketing claims as evidence of site readiness.
- Treat any platform as having "solved" industrial safety.
- Present specs from one platform as representative of all humanoids.
- Ignore the gap between published specs and field-validated performance.

---

## Usage by other skills

- **capability_envelope_writer**: Use platform dimensions and payload to bound reach/manipulation claims.
- **humanoid_route_access_reviewer**: Use platform width, height, and speed for clearance checks.
- **humanoid_workcell_risk_reviewer**: Use hand DOF, payload, and sensor specs for task feasibility.
- **oem_handoff_writer**: Include platform-specific constraints and deployment history in handoff.
- **blocker_taxonomist**: Use known limitations to flag platform-specific blockers.
- **shared_traffic_reviewer**: Use walking speed vs. ISO 3691-4 threshold (1.2 m/s).
