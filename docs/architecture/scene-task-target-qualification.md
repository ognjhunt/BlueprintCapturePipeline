# Scene task target qualification

This lane turns scene observations into bounded site-specific simulator targets. It is deliberately split into a dynamic proposal stage and deterministic authorization.

1. A provider-neutral analyzer consumes digest-bound rendered views and proposes visible objects, affordances, and task families.
2. Each proposal must bind its selected target to a 3D point or region using depth back-projection, a scene object bound, or multi-view ray intersection.
3. The compiler checks the frozen visual threshold, supporting-view digests, target binding, metric-scale status, collision support, and reach evidence.
4. The compiler authorizes a derived bounded simulator target, upgrades it to a metric simulator target when all metric gates are qualified, or abstains.

Authorized external reconstruction views remain useful when the original video is unavailable. They never become raw capture truth and cannot establish physical success, deployment readiness, or metric reach without the independent measurements required by the result contract.

The default manipulator is the official Isaac Franka Panda. Unitree G1 is selected only for humanoid task families.
