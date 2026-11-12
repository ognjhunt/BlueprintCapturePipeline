# Blueprint Pipeline Packet Factory

Use when work touches how this repository turns a site's video into its part of the job packet: the simulation of the site with the job's object in it, the success rule, and, where one exists, the early trial.

Primary references:
- The shared block of `BlueprintCapturePipeline/PLATFORM_CONTEXT.md`
- `BlueprintCapturePipeline/AGENTS.md`
- `BlueprintCapturePipeline/docs/SYSTEM_MAP.md`

Required behavior:
- Treat `BlueprintCapturePipeline` as the packet factory: video in, a simulation of the site with the job's object in it, and the success rule out.
- Treat the simulation as a means, not the product.
- Label Blueprint's own test "an early trial of openly available robot software". It is supporting evidence in the packet, never the site's answer.
- Keep the only hard stops: privacy, rights and money. Flag everything else and keep going.

Do not:
- Describe a scripted run as anything but a check of Blueprint's own simulation.
- Invent results, readiness or approvals that the pipeline's own records do not show.
- Start new work on what is paused: hardening the automatic test for any kind of job, several objects or unattended scale; standing re-tests of every robot on every open job.

Output:
- Analysis, implementation notes or operating guidance grounded in the pipeline's own records.
