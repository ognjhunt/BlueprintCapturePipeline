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
and exits 0 only when every variation ends where its row says. The whole matrix takes 35-75 s at `-j 3`.

## What runs, and on what

| Step | Real code that runs | Stand-ins |
|---|---|---|
| routing | `website_task_semantics._operation_profile` | the reading is the row's (the Atlas review that writes it is a paid model call) |
| front_half | the replay set's Atlas preview, levelling and mesh registration replays | synthetic records; `not_run_here` when the row names none |
| scene_chain | `prepare_construction_stages` (via `tests/test_website_native_inputs.packet`), `validate_immutable_stage_configurations`, `host_chain_plan`, then the real lane on exactly that recipe | role doubles for stages 1-4 (ARTISAN, the website background) |
| publication | the lane's publication; reports `physics_authority` | as above |
| completion_enqueue | the downstream replay, only for a job whose record is the same kind (Isaac, navigation) | a not-run job says `not_run_here`: no harness takes the chain's own revision; whether a paid probe is queued is decided on the compile |
| website_compile | `team_scene_packet.materialize_team_scene` with `WEBSITE_COMPILE`, then `team_scene_packet.record_team_scene`, which decides whether the paid robot-free probe is queued; reports `physics_check`, `robot_free_probe` (from `record_team_scene`), `robot_station_absent`, `assumed_thresholds` | the rigid fixture revision with the stage-5 record the row names (the chain's own record names another object, so it feeds nothing here); the registry write and the probe enqueue are recorders, and a call to the enqueue stops the row `would_buy` |
| company_zip | `website_company_simulation` builds and publishes the zip | the private store and website acceptance of its own tests |

No row is end to end: the scene chain and the website compile run on different fixture objects, and the command
says so under its table. The articulated row runs the real `prepare_construction_stages` with stage 3 replaced by a
stand-in articulated record (no articulated website fixture exists), so its stage validation is not run.
"no-clear-stance" and "look-file-quality-gate" are forced through the package's own test helpers; for
"furniture-moved" and "held-support-ships-with-the-flag", the half's sealed record is replaced with one carrying
an applied removal, or a held support record. They test the package's
reaction, not how a job gets there.

## What it cannot show

It runs no paid or scoring caller, so "only the package accepts a not-run stage 5" is the chain's own test
(`test_every_paid_or_scoring_caller_refuses_a_not_run_scene`), not a row here. Providers (Atlas, SAM, ARTISAN, any GPU), a real video, room or object, and the host's disk, memory and time. The
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
the matrix file's `readme_sentences` (literals, not the package's constants): its fixed text (`made`), the sentences
of one physics check, then the notes, and nothing else. It is the compile's `physics_check` when those are its
sentences (else `readme_disagrees_with:<it>`), then each note, then `unexpected_text:<the rest>` for any other
text, so a changed or added sentence fails rows until the matrix file is changed on purpose. The rest of the README
(title, file list, Terms) is pinned by each zip row's `readme_sha256`: any change fails the row as
`readme_sha256_differs`, with the README as built. Each FAIL lists what differed. The JSON (`--report`) holds every step with
its status, stop, `fed_from` and outputs. Two runs of one tree differ only in `seconds` and the work folder.

## Adding a variation

Add a row to `tests/fixtures/job_matrix_v1.json`: `label` (what it tests), `kind`, `reading` (`task_family`,
`steps`), optional `front_half` and `downstream` replay record ids, `scene` (`host_chain`, `destination`,
`articulated`, `isaac` or `none`), `compile` (`strategy`; `stage_5`: `not_run`, `not_run_static_passed`,
`not_run_claiming_isaac`, `isaac_qualified` (the fixture's own passed Isaac record, saying Isaac ran) or `fixture`; `case`: `default`, `no_stance`, `quality_gate` or `owner`), optional `zip`
(`furniture_moved`, or `support_held`: the support record's typed reasons), and `expected`: `last_stage` and any of `kind`, `code`, `zip_built`, `readme_variant`, `readme_sha256`,
`robot_station_absent`, `physics_authority`, `physics_check`, `robot_free_probe`, `assumed_thresholds`. Change a
row's `expected` only with the change that moves it.
