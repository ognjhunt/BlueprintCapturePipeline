# ARTISAN task-object authoring

Owner direction, October 12, 2026: replace custom Astra CAD/Blender authoring
for **new** rigid and articulated objects with official ARTISAN, and make the
authoring budget centrally editable. This is the ADP-009 construction dependency
for the day-14 scene rehearsal and day-21 diagnostic. It does not replace capture
reconstruction, appearance removal, scene registration, or native evaluation.

This implementation is awaiting runtime rehearsal and production activation.

## Switch: merged, off until go-live

Owner decision, October 13, 2026: ARTISAN is merged but **switched off**. One
setting, `BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND`, selects the builder for new
requests (`stage_three_configuration`, `articulated_stage_three_configuration`,
website compile, submission) and for deploys (`deploy_control_plane_commit.py`,
`provision_task_evaluation_scene_configuration_release.py`). Unset, it is
`astra_cad_blender_v1`, so deploys and new builds behave exactly as before this
change. Any value other than `artisan_v1` or `astra_cad_blender_v1` refuses.
A request or configuration that names its backend is never rewritten.

Go-live, in order:

1. Install the pinned runtime on the production host (see *Reproducible worker
   and release*).
2. Put all four keys in `/etc/blueprint/pipeline-control-plane.env`:
   `BLUEPRINT_TASK_OBJECT_AUTHORING_BACKEND=artisan_v1`, plus
   `BLUEPRINT_ARTISAN_UPSTREAM_ROOT`, `BLUEPRINT_ARTISAN_SITE_PACKAGES_ROOT` and
   `BLUEPRINT_ARTISAN_BLENDER_ROOT`. Every service loads that file, and the
   operator door launches a deploy with only its `DOOR_*` values, so the deploy
   reads these four keys from the same file. The process environment wins if it
   names a key. A symlinked file is ignored.
3. Deploy, then run one rigid and one articulated canary and confirm they are
   valid. A deploy provisions one builder's toolchain, ARTISAN or Astra, not both.

Rollback is unsetting the setting and deploying again. Issued configurations keep
their recorded backend.

At the $15 default (`BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD`), a rigid website
build quotes $21: $6 of provider compute plus $15. That is above the website's
default $20 build ceiling. A $14 cap keeps it under $20; otherwise an owner grant
is needed.

## Integration map

New submission/website configuration -> `artisan_v1` -> existing stage-three
producer slot -> neutral authoring entrypoint -> isolated official ARTISAN CLI
-> reviewed upstream candidate/exports -> final-byte source checks -> existing
static/native/scene-composition boundaries.

`content_agents_rigid_replacement` remains the historical stage/package slot
name. Its new receipts explicitly name `artisan_v1`; they do not claim Astra
or NVIDIA Content Agents execution. Missing runtime, bad source bytes, failed
confinement, failed framework status, and unsupported output semantics refuse.
There is no automatic backend fallback.

The upstream pin is
`physicalvision/Artisan@ad322b5b31e50bdb84155520f09622c00c6056c3`, tree
`6ed9c860e74afdf90bade4bc5ec4ac592cb1e920`. Preserve its Apache-2.0 license,
notices, third-party attributions and `uv.lock`. The tracked patch
`patches/artisan/0001-confine-generated-code.patch` routes generated model,
QA, probe and Blender execution through Blueprint's OS sandbox; admits a
declared one-part rigid model without a fake joint; and retains visual/collision
identity and opacity in USD. Design/Coding/Review, CAD, collision construction,
simulation, repair, review and exports remain upstream implementations.

## Inputs and authority

`task_object_construction_request.v1` retains the complete issued configuration,
source candidate, original frame bytes/hashes, complete handoff/view context,
object/scene/run identities, geometry/envelope bindings, nominal dimensions,
dimension authority/uncertainty, placement context, properties and constraints.
All admitted views are passed as repeatable `--image` arguments. There is no new
framework-wide frame-count ceiling. CPU preflight verifies decodable images
and retained physical-evidence bytes before a paid attempt.

Semantic body extents use the retained front +X/up Z frame rather than a rotated
world AABB. Final package readback checks metre/Z units, XY-centred/bottom-zero
bounds, dimensions within the separately recorded 1 mm construction tolerance,
identity, topology, bodies/joints, useful task travel, quaternion-aligned axes,
declared pivots, required parts, handle collisions, mass/COM/inertia, material
bindings and portable dependencies. Source uncertainty remains unchanged.

Upstream must author supplied mass correctly. A conflicting estimate is refused;
the adapter does not rescale geometry or recompute inertia to disguise it.
Friction/restitution use an explicit source overlay: verified measurements take
precedence; otherwise an admitted range midpoint is labelled an estimate.
Namespace/provenance/material edits produce new bytes, which are revalidated.
The earlier upstream physics report certifies only its original candidate.

Hollow assemblies additionally author
`assets/blueprint_candidate_annotations.v1.json` with a bound request digest and
explicit interior probe regions (the construction prompt describes the fields).
Retained cavity constraints cannot be changed. Generated probe regions remain
labelled candidate geometry. The existing independent cavity rays check exact
boxes or convex-hull collision pieces. A solid collider, missing back or
unproven convex-decomposition cooking refuses. This check does not establish
capture fidelity or native collision cooking.

Passive joints retain explicit roles and damping. Mimic relationships and
their unit-correct dependency graph are retained rather than dropped. The
current native settling adapter cannot certify coupled motion: static admission
refuses such candidates with `artisan_coupled_joint_requires_native_adapter_support`.
General coupled-motion qualification remains a named implementation dependency.

For a revolute task joint, the planner can supply `mechanism.gravity_response`
as `{"mode": "counterbalanced", "reason": "..."}` in the stage-three input.
The other modes are `gravity_loaded` and `unknown`; omission preserves authored
gravity. The immutable configuration retains the choice under `passive_dynamics`,
and the export records it on the exact joint as `blueprint:gravityResponse`.
Only an explicit counterbalanced declaration enables the runtime's ideal gravity
compensation prior. A horizontal hinge alone does not select it. The choice and
source bytes are checked during export, staging and replay, and existing spring
or moving drive behaviour is refused rather than silently overwritten. This
remains an estimated mechanism, not a physical measurement.
The existing assembly footage review can propose the same choice, with a reason,
for new captures. Conflicting view batches merge to `unknown`. Complete revision-5
coverage records remain reusable without another paid classification; their
missing gravity-response estimate stays missing.
An explicitly frozen task template also carries the choice into the native task
specification. Runtime compilation can therefore derive the same labelled prior
from an older qualified asset without regenerating its CAD or editing its sealed
bytes. The derivative receipt retains the choice and replays it against that
exact source; this does not retroactively change earlier episode results.

## Budget and recovery

Set **one** future-request setting, without changing Python:

```bash
export BLUEPRINT_ARTISAN_AUTHORING_MAX_COST_USD=40
```

The starting default is USD 15 for the entire object/assembly authoring stage,
including all roles, compaction, image generation and resumes. It is not a
per-part allowance or an immutable owner ceiling. Higher finite settings reach
quoting, intake validation, stage/parent profiles, bundle receipts, signed
admission and the CLI. A lower signed allocation still governs execution.
The resolved setting is frozen into the request; changing the environment does
not alter issued requests. Legacy profile ceilings remain historical contracts.

There is **no uncapped mode**. The pinned upstream CLI requires a finite positive
allocation. Zero, infinity, NaN and `uncapped` refuse; omitting its flag would
choose its default and would not remove the cap.

The model configuration is frozen too: GPT-6 Astra/high for text,
GPT-5.6 Sol compaction, and GPT-image-2.5-sunburst design boards. The controller
checks actual model availability before billed generation. It uses the existing
canonical scoped OpenAI key file, never an ambient key or plaintext login file.
Generated code receives neither the key nor its readable location.

The thin transport meter records text, compaction and image charges, disables
SDK transport retries, bounds text output against official input-token counts
and remaining allocation, and atomically persists cumulative usage. An in-flight
or unpriced charge blocks further inference until reconciliation. Image charges
remain upstream budget estimates, not final invoices. Compaction has no API
output-limit parameter; the whole remaining pool is reserved, and reported
overruns refuse. Official billing settlement remains necessary; local metering
does not establish an absolute provider-side billing limit.

One official stage reservation/baseline governs resumes. Failed attempt snapshots
are retained separately, preserving the original pool. Same-workspace upstream
`resume` retains the original run ID, budget, usage, request and absolute
deadline. The stage's 7,800-second clock and parent deadline are never refreshed
on retry; 600 seconds are retained for closeout. Controller/child teardown runs
on deadline or cancellation.

Completed component results and complete retained candidates are adopted after
byte/digest verification without runtime setup or inference. A completed upstream
run can also be adopted after local packaging failure. Failed adaptation attempts
retain separate candidate directories. The existing CPU pre-stage capsule and
same-root checkpoint protocol also admit ARTISAN prefixes.

## Reproducible worker and release

On a Linux x86_64 build host with governed network access, first obtain and inspect
the genuine upstream checkout and official installer:

```bash
git clone https://github.com/physicalvision/Artisan.git work/Artisan
git -C work/Artisan checkout --detach ad322b5b31e50bdb84155520f09622c00c6056c3
bash work/Artisan/install.sh --help
```

Run the inspected installer in its isolated Python 3.12 environment with the USD
extra, then its doctor. Do not put credentials in the source checkout. The
installer includes Blender 5.0.0 and a large locked dependency closure, including
benchmark dependencies. Record actual disk/package size before release admission.

From a clean, committed Blueprint checkout, build the local immutable component:

```bash
PYTHONPATH=src:. python scripts/build_task_evaluation_scene_configuration_artisan_component.py \
  --repository-root "$PWD" --expected-blueprint-commit "$(git rev-parse HEAD)" \
  --artisan-root /absolute/path/to/Artisan \
  --site-packages-root /absolute/path/to/isolated/python3.12/site-packages \
  --blender-root /absolute/path/to/blender-5.0.0-linux-x64 \
  --output-root /absolute/path/to/artisan-component
```

The builder requires the genuine clean revision/tree and exact locked installed
distributions, rejects vendor symlink escapes before copying, applies the patch,
tests isolated imports, and seals source/lock/patch/runtime hashes and notices.
The inference subprocess uses `-S` and only sealed upstream/bridge/library paths.
The parent `artisan_source_validation` wheelhouse contains deterministic source
validators and shared governance dependencies, without the old CAD/LangGraph
authoring closure. Workers perform no network dependency installation.

Release provisioning and the canonical deployment command accept
`--authoring-backend artisan_v1`, `--artisan-root`, `--artisan-site-packages-root`
and `--artisan-blender-root` (or the corresponding
`BLUEPRINT_ARTISAN_{UPSTREAM,SITE_PACKAGES,BLENDER}_ROOT` settings). They no longer
provision legacy CAD sources for this backend. These commands are preparation
instructions, not evidence of deployment or permission to activate production.

## Retirement, rollback and evidence

The new inference closure excludes custom Astra authors/workers, persistent local
Agents sessions, family/per-part builders, Multi-Agent-CAD/text-to-CAD, duplicate
appearance/property model reviews and custom SimReady construction. Independent
source/static/native validators remain. Generic cavity validation was extracted
from the legacy packager into a shared neutral module. Historical Astra/Content
Agents routes and manifests remain readable for explicitly issued work; their
backend IDs, digests and completed artifacts are unchanged.

Rollback uses the prior exact release and its admitted toolchain. It does not
rewrite ARTISAN requests as Astra or use legacy sources in an ARTISAN worker.
Explicitly issued legacy configurations select their original backend. Anthropic
or Agents API settings require that explicit legacy backend; they never silently
change a fresh ARTISAN request.

October 12 verification in this environment: pinned upstream text source and small
binary files were obtained through the GitHub connector after the configured
proxy blocked cloning. Fourteen large textures are still missing; this is **not**
a full clone. Installer/real CLI help and patch applicability were checked. The
full SDK is uninstalled (`manifold3d` and other locked dependencies are missing).
The host denies the required network namespace and reports Landlock unavailable.
CPU tests exercise actual USD bytes and local SDK HTTP cassettes, not live
framework generation. Retained real capture images and native Isaac execution
have not been verified here.

Code validation, framework execution, native scene validation, production
activation and physical equivalence are distinct milestones. Before activation,
finish a packaged-runtime rehearsal on a working sandbox host, a valid retained
rigid control and articulated source, applicable scoped execution/disclosure
authority, native cooking/settling/sweep/reset, official cost settlement and
closeout. Retain exact input/program/report/export hashes and limitations for each
real canary. All current candidate claims remain `development_only`.
