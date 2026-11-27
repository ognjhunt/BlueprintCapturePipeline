# Test job matrix ($0, in your own container)

One command runs many kinds of job through every step that can run without the host, at no cost, and says
where each one stopped.

```bash
python -m blueprint_pipeline.test_job_matrix --matrix tests/fixtures/job_matrix_v1.json -j 3
python -m blueprint_pipeline.test_job_matrix --matrix tests/fixtures/job_matrix_v1.json --only navigation
python -m blueprint_pipeline.test_job_matrix --matrix tests/fixtures/job_matrix_v1.json -j 3 --report out.json
```

Run it from a git checkout of the Pipeline (the replay set reads the source commit; a plain copy without `.git`
fails two replay records), with `PYTHONPATH=src:.` if your editable install points at another checkout. Call it
with `-m`: a script that imports `run_matrix` must guard its own code with `if __name__ == "__main__":`, because
each variation is a spawned process that imports the caller again. It needs no host, no door, no key and no network,
and exits 0 only when every variation ends where its row says. The whole matrix (35 rows) takes about 2-3 minutes at `-j 4`.

## What runs, and on what

| Step | Real code that runs | Stand-ins |
|---|---|---|
| routing | `website_task_semantics._operation_profile`; a job with no profile goes on as a room package | the reading is the row's (the Atlas review that writes it is a paid model call) |
| front_half | the replay set's Atlas preview, levelling and mesh registration replays | synthetic records; `not_run_here` when the row names none |
| scene_chain | `prepare_construction_stages` (via `tests/test_website_native_inputs.packet`), `validate_immutable_stage_configurations`, `host_chain_plan`, then the real lane on exactly that recipe | role doubles for stages 1-4 (ARTISAN, the website background) |
| publication | the lane's publication; reports `physics_authority` | as above |
| completion_enqueue | the downstream replay, only for a job whose record is the same kind (Isaac) | a not-run job says `not_run_here`: no harness takes the chain's own revision; whether a paid probe is queued is decided on the compile |
| website_compile | `team_scene_packet.materialize_team_scene` with `WEBSITE_COMPILE`, then `team_scene_packet.record_team_scene`, which decides whether the paid robot-free probe is queued; reports `physics_check`, `robot_free_probe` (from `record_team_scene`), `robot_station_absent`, `assumed_thresholds` | the rigid fixture revision with the stage-5 record the row names (the chain's own record names another object, so it feeds nothing here); the registry write and the probe enqueue are recorders, and a call to the enqueue stops the row `would_buy` |
| website_compile (room package) | for `compile.handoff: room_package` (navigation, an unknown kind; with `nothing_found: true`, a clean plate that found nothing of the task): the real `website_scene_handoff.prepare_website_scene_handoff`, which writes the room package (`write_room_package`) | `tests/test_website_scene_handoff_navigation.room_job`: real fixture geometry and splat, the Atlas lane and mask binding stood in |
| website_compile (several objects) | for `compile.further_objects`: the v3 decision `website_atlas_pipeline.job_object_review` seals, then the real `website_scene_handoff.prepare_website_scene_handoff` (`apply_to_removal`, `website_further_objects.sealed_inputs`, the preparation's compile: `place`, `author`, `static_check`, `compose`), written into the zip's capture; the half compiled with what `team_scene_packet.further_objects_for` reads of that preparation | `tests/test_website_further_objects`' cup on its table; the plan model's answer, the Atlas scene source, the mask binding and the scored object's own library step stood in (`tests/test_website_further_objects_removal._reviewed`, `_handoff`); the further objects' poses moved beside the half's room (`_in_room`) |
| company_zip | `website_company_simulation` builds and publishes the zip; a room package is read from the handoff, as on the host | the private store and website acceptance of its own tests |

No row is end to end: the scene chain and the website compile run on different fixture objects, and the command
says so under its table. The articulated row runs the real `prepare_construction_stages` with stage 3 replaced by a
stand-in articulated record (no articulated website fixture exists), so its stage validation is not run; its
website compile uses the drawer cabinet of `tests/test_team_scene_packet._articulated_scene` with the not-run record.
"no-clear-stance" and "look-file-quality-gate" are forced through the package's own test helpers; for
"furniture-moved" and "held-support-ships-with-the-flag", the half's sealed record is replaced with one carrying
an applied removal, or a held support record. "unknown-kind-one-object-rule-withheld" is compiled with the
handoff's sealed reason, as the worker reads it (the handoff's route is tested in
`tests/test_website_scene_handoff.py`), and the two "rigid-half-without-a-rule" rows drop the half's rule, with no
reason or one no handoff gives. The four "one-object-..." rows (stress round 2, 2026-11-24) bind the half to a
preparation whose stage-3 configuration the one-object route made (`zip.object_source`): a library model (the
library tests' fixture catalog, deployed as the row's own catalog, so its rights are read as compile and the package
read them), the plan's tote shape, a cylinder that does not fit and so leaves ARTISAN, and ARTISAN; each README says
what the object is. They test the package's
reaction, not how a job gets there.

The job-type rows (2026-11-24, `docs/test-footage/clip-mix-v1.json` is their real-clip half) do the same. The room
package rows run the real handoff with the fixture's own cups as the clean plate's objects: floor wiping (the family
route with no runtime installed, as on every host, so `room_job` runs without
`BLUEPRINT_TASK_FAMILY_RUNTIME_CATALOG_PATH`), the planner's removals not applied, several objects kept in the
scan or taken out of it, a rigid reading whose object opens, and navigation with two goal masks.
"depalletize-one-carton-parametric", "designed-object-not-filmed" (the synthetic-subject tests' designed ball) and
"glass-destination-surface-estimated" (only the base scene's estimate row, beside the bound preparation) bind the
half to a preparation as the one-object rows do.

The four "several-objects-..." rows with `compile.further_objects` (2026-11-27) are a job with the scored object (the
fixture's cup) and one further object, run from the Atlas job's v3 decision through the scene handoff and the
preparation's compile into the half and the zip, which is bound to that preparation (as a compile binds a website
revision's) and reads that decision from the same capture: a carton (the plan's parametric shape) and a mug box (the
fixture's sealed library model, credited) are taken out of the scan, built free and composed as `/World/Object_02`; a
can (its plan shape a cylinder, which can miss the size tolerance) and a toy that needs a generated model are kept in
the scan as filmed by the decision, flagged `scene_estimate:further_object_left_in_scan`, and the README names each
with why. Nothing is reserved for the toy: no host can author a paid further object
(`website_further_artisan.installed_runner`), so this is today's route. A library model would not change the can's
row: the decision reads a cylinder as not sure of a free build before any library model is picked for it
(`website_extra_objects.free_before_reconstruction`). The scored object in the zip is still the compile fixture's, its
README note the cup's preparation's. `_in_room` sets each further object's pose beside the room's task object without
seating it, so where a further object stands in these zips is not tested here (`tests/test_website_further_objects.py`
tests the seat).

## What it cannot show

It runs no paid or scoring caller, so "only the package accepts a not-run stage 5" is the chain's own test
(`test_every_paid_or_scoring_caller_refuses_a_not_run_scene`), not a row here. Providers (Atlas, SAM, ARTISAN, any GPU), a real video, room or object, and the host's disk, memory and time. No row holds what only a real clip has: its rotation, resolution, frame rate, length, light, sharpness, the
people in it or the size of the room; those are the clip mix's rows. The planner's handling of frames showing
people is `tests/test_atlas_scene_planner.py`, not a row here. The
network guard refuses and records, in every variation's process, `connect`, `connect_ex`, `sendto`,
`create_connection`, `getaddrinfo`, `gethostbyname`, `gethostbyname_ex`, `gethostbyaddr` and any program started
through `subprocess` except exactly `git -C <repo> rev-parse HEAD` and `git -C <repo> status --porcelain` (the
replay set's local reads, as a list; any other argument list, a shell or string command, a `-c` setting, a
`GIT_*` variable (each child starts without the caller's), or another git subcommand is refused); proxy settings point at a closed local port and credential-like variables are removed
there (never in the caller). Not covered: `os.system` and the `os.exec*`/`os.spawn*` calls, sockets opened by C
extensions, and the parent process (it runs only the standard library).

## Reading the report

One line per variation: PASS or FAIL, label, kind, last step, first stop (`kind:code`), the start of the zip
digest, the README variant and seconds. The variant reads the README's "How it was made" section exactly against
the matrix file's `readme_sentences` (literals, not the package's constants): its fixed text (`made`), the build
plan's notes (`plan_` names; the README puts them first), the sentences of one physics check, then the other notes,
and nothing else. It is the compile's `physics_check` when those are its
sentences (else `readme_disagrees_with:<it>`), then each note, then `unexpected_text:<the rest>` for any other
text, so a changed or added sentence fails rows until the matrix file is changed on purpose. The rest of the README
(title, file list, Terms) is pinned by each zip row's `readme_sha256`: any change fails the row as
`readme_sha256_differs`, with the README as built. Each FAIL lists what differed. The JSON (`--report`) holds every step with
its status, stop, `fed_from` and outputs. Two runs of one tree differ only in `seconds` and the work folder.

## Adding a variation

Add a row to `tests/fixtures/job_matrix_v1.json`: `label` (what it tests), `kind`, `reading` (`task_family`,
`steps`), optional `front_half` and `downstream` replay record ids, `scene` (`host_chain`, `destination`,
`articulated`, `isaac` or `none`), `compile` (`handoff: room_package` for a room package row, optionally `nothing_found: true`, else `strategy`; `stage_5`: `not_run`, `not_run_static_passed`,
`not_run_claiming_isaac`, `isaac_qualified` (the fixture's own passed Isaac record, saying Isaac ran) or `fixture`; `case`: `default`, `no_stance`, `quality_gate` or `owner`), optional `zip`
(`furniture_moved`, or `support_held`: the support record's typed reasons; `object_source`: `library`, `parametric`
with `shape` and `extents`, `generated` or `designed`, optionally with `surface_estimated: true`), a room package row's
`compile` may add `removal` (`one_cup` or `two_cups`), `all_targets`, `object_opens`, `second_goal`,
`job_objects_kept` and `planner_removals_not_applied` (a count), a rigid row's `compile` may add `further_objects` (the
further objects of a several-object job, by their target in `tests/test_website_further_objects`: `carton-1`, `mug-1`
with its library pick, `can-1`, `artisan-1`; not with `zip.object_source`, since the zip is bound to the job's own
preparation), and `expected`: `last_stage` and any of `kind`, `code`, `zip_built`, `readme_variant`, `readme_sha256`,
`robot_station_absent`, `physics_authority`, `physics_check`, `robot_free_probe`, `assumed_thresholds`. Change a
row's `expected` only with the change that moves it.
