# Website geometry controller

ADP-009B / public day 14. A confirmed website task now sends its retained
original frames to the existing admitted Vast reconstruction worker. Geometry
is estimated; it never becomes measured scale or physical evidence.

Visual delivery precedes that GPU job. CPU frame decoding, Gemini analysis,
hosted SAM tracking and image editing prepare Marble's input. The provider
publishes the finished visual world through the existing website callback.
Only then does the scene handoff request MapAnything and lift the retained SAM
tracks into estimated geometry. A missing GPU or geometry result holds native
construction, while the published visual world remains available. No second
tracking purchase is needed for this binding.

The service deployment supplies `BLUEPRINT_WEBSITE_MAPANYTHING_PROFILE`, the
path to one release-specific JSON profile. It is service configuration, never
an upload field or a per-capture command. The profile has:

- `schema_version`: `website_mapanything_runtime.v1`.
- `source_commit`: the admitted deployed Pipeline commit.
- `worker_image_digest`: the pinned container image reference, including digest.
- `maximum_cost_usd`, `max_hourly_rate_usd`, `hard_ttl_seconds`, and
  `minimum_gpu_ram_mb`: the bounded worker limits. The TTL is 120–3600 seconds;
  its maximum hourly cost must fit the cap.
- `runtime_files`: four objects with absolute `path` and `sha256:`-prefixed
  `digest`: the Pipeline wheel for that release, the pinned blueprint-contracts
  wheel, the pinned MapAnything wheel, and the hash-locked dependency file.
  The existing bootstrap installs exactly three wheels and one dependency file.

Canonical deployment builds this profile automatically when
`/etc/blueprint/website-mapanything-runtime.json` exists (or the deploy process
sets `BLUEPRINT_WEBSITE_MAPANYTHING_DEPLOYMENT_TEMPLATE`). This operator-owned
template uses `website_mapanything_deployment.v1`, the same worker image and
budget fields, and only the three pinned external runtime files: MapAnything,
blueprint-contracts, and the dependency lock. The release provisioner builds
the Pipeline wheel from the exact committed source tree without network
dependency resolution, copies and verifies the pinned inputs, and publishes
the profile under `system-runtimes/website-mapanything/<commit>/`. It reads the
files back as the service account and returns the profile binding through the
existing deployment environment. Subsequent captures reuse it; subsequent
releases rebuild it. Changed inputs or an unreadable runtime fail deployment.
This deployment artifact does not allocate a GPU or grant spending authority.

The existing object-store and Vast credentials remain service-owned. WebApp's
sponsorship configuration must include sufficient upstream allowance and current
Vast terms, as it already does for other preparation providers. The signed
reservation shares that allowance with SAM, image editing and Marble; no robot
team payment is required.

Controller sequence: validate the admitted release and runtime files; prepare
the input bundle; arm the existing independent watchdog for the exact resource
name; check capacity and provider inventory; obtain the signed budget grant;
stage the bundle; invoke the canonical allocator; retrieve and independently
validate outputs; verify teardown; close the watchdog and remove temporary
transport objects. Completed outputs replay without a new rental. A lost or
uncertain allocation response retains the watchdog and transport and refuses a
second rental until reconciliation. This is not a second job queue.

The ordinary website path does not accept `BLUEPRINT_WEBSITE_GEOMETRY_RESULT`
as a substitute for this controller handoff. That override remains only for
explicit component diagnostics without a confirmed website task.

Validation: focused tests cover the real bundle contract, exact-name watchdog,
controller reservation/dispatch, completed replay and uncertain allocation.
Deployment configuration and an actual controller-origin run are separate proof
requirements; a passing fixture or retained-output replay does not satisfy them.

The canonical control-plane deploy also installs the agent-run dispatcher
service and timer and restores the timer after deployment or reboot. The
service still requires `BLUEPRINT_AGENT_RUN_DISPATCH_ENABLED=true` and its
configured capture scope; timer installation alone does not authorize a run.
Its Python entrypoint is checked by the production startup guard.

CAD generation and repair share the pinned `gen_step()` shape-return contract:
the CAD CLI owns exports, and the agent returns geometry rather than a filename
or dictionary. This fixes the retained authoring failure in the reusable stage;
the failed standalone candidate remains unaccepted and is not controller proof.
