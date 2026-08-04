# Arm Decision Proof v1 Partner Selection Packet

## What We Are Asking For

Blueprint is seeking one university robotics lab or robot team to answer one
real decision it already has:

> Which of two runnable policy checkpoints or configurations should receive the
> next scarce block of physical robot testing for one fixed-arm task, and under
> which site conditions should neither be trusted?

Blueprint will first ingest and qualify the partner's existing scene, capture,
CAD, calibration, and simulator assets. It will collect new site evidence only
for measured gaps, construct a bounded evaluation replica, run a prospective
simulated comparison, seal the result, and compare it with a randomized or
interleaved physical holdout administered under a frozen protocol.

This is a joint technical study, not a deployment certification, safety
approval, universal policy benchmark, or promise that simulation will agree.

## Required Partner Profile

The partner must have:

- a fixed robot arm in a stable workcell;
- fixed RGB or RGB-D cameras and a parallel-jaw gripper;
- one rigid-object pick-and-place task with a machine-verifiable outcome;
- a repeatable reset that can be written and followed;
- two real policy checkpoints or configurations that run through the same
  observation/action interface;
- a genuine reason to choose, eliminate, or allocate testing between them;
- authority to run the preregistered physical holdout;
- an operator who can provide task truth and record interventions;
- rights to the capture, policy execution receipts, and agreed case-study
  material;
- a plausible expectation of repeated policy iterations or otherwise expensive
  physical testing.

Preferred initial ecosystems are DROID/Franka or another arm with an already
working Gym-like or ROS-compatible policy interface. Existing partner runtime
and simulator infrastructure wins over Blueprint introducing a new engine.

## Immediate Rejection Criteria

Reject or defer a candidate partner when any of these is true:

- no real testing decision exists;
- only a scripted controller or fabricated policy comparison is available;
- the task depends on deformables, cables, liquids, granular media, tight
  insertion, force control, dynamic people, or mobile/humanoid locomotion;
- resets or outcomes require subjective retrospective judgment;
- candidate policies use incompatible observation/action interfaces that would
  make integration the whole project;
- there is no authority to reserve a disjoint physical holdout;
- capture, policy, publication, privacy, or outcome rights cannot be documented;
- direct physical testing is cheap, abundant, and unlikely to be reused;
- the partner wants a success guarantee or deployment/safety certification.

## Selection Scorecard

Score each item `0`, `1`, or `2`. Select only partners scoring at least `20/24`
with no zero in a required row.

| Criterion | 0 | 1 | 2 | Required |
| --- | --- | --- | --- | --- |
| Real decision | Academic demo only | Weak preference | Testing allocation will change | yes |
| Two candidates | Missing/incompatible | One can be made runnable | Both run today | yes |
| Task simplicity | Out of scope | Some avoidable complexity | Bounded rigid pick/place | yes |
| Outcome metric | Subjective | Human adjudication with rubric | Machine-verifiable | yes |
| Reset | Unbounded | Repeatable with judgment | Scripted/checklisted | yes |
| Observation/action interface | Unknown | Adapter work needed | Same stable interface | yes |
| Holdout authority | None | Tentative | Reserved and controlled | yes |
| Rights/privacy | Blocked | Negotiable | Written permission available | yes |
| Physical scarcity | Abundant/cheap | Moderate | Costly, slow, risky, or scarce | no |
| Reuse | One-off | Possible | Repeated checkpoints/conditions | no |
| Operator availability | None | Intermittent | Named task owner and operator | yes |
| Case-study value | Cannot disclose | Redacted only | Useful bounded publication | no |

## First Partner Interview

Record answers and evidence; do not convert verbal confidence into qualified
facts.

1. What exact decision will this experiment change?
2. What are the two candidate identities, and can both run today?
3. What observation tensors, action representation, control rate, and runtime do
   they use?
4. What is one trial, and what ends it?
5. How are success, partial success, failure, intervention, timeout, and invalid
   trial determined?
6. Who resets the scene, and how is reset equivalence checked?
7. Which conditions occur in normal operation, and at what approximate
   frequencies?
8. Which failure boundary would be useful to know before spending more robot
   time?
9. What minimum difference between candidates would change the decision?
10. How many physical trials are affordable, and what makes them scarce?
11. Which conditions can be reserved as a disjoint holdout?
12. What CAD, URDF/USD/MJCF, camera calibration, robot logs, and prior runs exist?
13. What capture, policy, data, privacy, publication, and commercialization
    restrictions apply?
14. Who has authority to approve the protocol and release the holdout?
15. If this works, how many future checkpoints or tasks could reuse the testbed?

## Day-7 Admission Packet

The partner is not admitted until the packet binds:

- partner and named task owner;
- robot, gripper, cameras, and controller versions;
- exact task and workcell;
- two candidate identities and runnable receipts;
- decision and minimum meaningful difference;
- task distribution draft and invalid conditions;
- reset and outcome definitions;
- physical trial budget and holdout custodian;
- capture, policy, outcome, privacy, and case-study rights;
- available source assets and their provenance;
- reuse hypothesis;
- risks, blockers, and named owners;
- approval signatures or equivalent durable receipts.

Absent fields remain blockers. Agents may assemble and validate the packet, but
they cannot infer authority, task truth, rights, safety, or physical outcomes.

## Partner Deliverable

The partner receives:

- the frozen protocol;
- a versioned rerunnable Site-Task Testbed package;
- per-condition simulation and physical evidence links;
- the prospective decision and final adjudication;
- uncertainty and known-invalid conditions;
- an exact statement of what was and was not shown;
- a reuse estimate for the next candidate pair.
