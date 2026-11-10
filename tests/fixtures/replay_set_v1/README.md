# Replay set v1 (synthetic)

Every record in this directory is **synthetic**. None of them was pulled from a host, and none was
derived from a real capture, clip or job record. They were built by `tests/replay_set_v1_build.py`
with the replay tools' own unit-test builders (`website_job` in
`tests/test_task_evaluation_downstream_replay.py`, and the navigation intake builders in
`tests/test_website_navigation_intake.py`). Every manifest entry says `"synthetic": true` and has a
`provenance` line. Do not describe any of them as a retained job.

## Run it

```bash
python -m blueprint_pipeline.replay_set --set tests/fixtures/replay_set_v1
```

The command prints one line per record and tool (record, job kind, tool, expected stop, actual stop,
PASS or FAIL) and exits 1 if any record's first stop differs from the expected one. A record also
fails if its report shows a network attempt, a new paid call, a provider mutation or a changed input.
The replays enforce their own boundaries (see each replay module's docstring), so the run costs $0 and
uses no host and no network. It takes about 5 seconds locally. CI runs it nightly and on pull requests
that touch preparation, levelling, screening or the replay modules (`.github/workflows/replay-set.yml`),
along with `tests/test_replay_set_v1.py`.

## Layout

- `manifest.json`: one entry per record. Each entry gives the job kind, the declared input variety,
  the replay tool or tools to run, each tool's inputs (relative to the record's host root), its
  `until` stage and its `expected_stop` (`kind`, `stage`, `code`; `{host}` in a code stands for the
  record's host root).
- `records/<record_id>/`: the record's files, laid out as the pulled host tree. JSON files over 32 KB
  are stored gzipped (`*.gz`) and are decompressed when the record is put back.

Some records name absolute paths. For example, a launch receipt names its result file by path, and
that path is covered by the receipt's digest. Each record was therefore built at
`/tmp/blueprint-replay-set-v1/<record_id>/host` (`materialize_root` in the manifest), and the runner
puts it back at exactly that path, holding a lock while it does, and removes it afterwards.

Every record is replayed at the wall-clock instant in the manifest's `replay_clock_epoch`, not at
today's clock: the records carry real grant times (consent and expiry), and the replayed chain
compares them with the clock. Rebuilding the set writes a new instant.

| record | kind | expected first stop |
| --- | --- | --- |
| `rigid-pick-place-1080p-short` | rigid pick-and-place | `completed` at `screening_plan` (plan frozen within the job ceiling) |
| `rigid-pick-place-4k-long` | rigid pick-and-place | `would_buy` at `probe_enqueue` (no retained probe trace, so the GPU probe would be bought; it is named and not made) |
| `rigid-pick-place-1080p-long-closure-missing` | rigid pick-and-place | `missing_input` at `completion` (post-teardown provider-zero receipt not retained) |
| `navigation-4k-short` | navigation | `refusal` at `completion`, `preparation_completion_not_adopted` |
| `navigation-1080p-long` | navigation | as above, under its own submission |

## What it proves, and what it does not

It proves that the harness works: one command replays several records of both job kinds through
`task_evaluation_downstream_replay` (completion, readiness, team packet, probe enqueue, screening
plan), offline and at $0, and fails when a stop moves. It also pins each job kind's path through that
chain. A rigid pick-and-place launch reaches its screening plan, stops at the paid probe, or names the
file it is missing. A navigation job (prepared with its scene kept as filmed, with no launch) is not
adopted by the launched-preparation join, and the refusal names the job.

It does **not** prove anything about real-scene variety. Resolution (1080p or 4K) and clip length
(short or long) are declared labels only (`"read_by_replayed_code": false`). The replayed chain does
not read either value, and the scenes are the same small synthetic fixture. Real reconstruction
quality, real task contexts and real capture defects are not covered. The other replay tools
(`task_evaluation_stage_replay`, `atlas_preview_replay`, `task_object_artisan_replay`) are not in the
set yet, because their inputs (stage directories, Atlas results, object exports) have no synthetic
stand-in that a record could carry without the host.

## Adding a redacted record of a retained job (only after owner item O-03 is decided)

Both repositories are public. Data derived from real clips may not be committed here until the owner
decides where it may live (O-03). Once that is decided, and only in the place it names:

1. Pull the job's inputs through the operator door, using the downstream replay's pull list
   (`docs/OPERATOR_DOOR.md`, the downstream replay section). Pull JSON only, plus the configured
   bundle zip. Never pull video, frames, splats or meshes.
2. Redact the inputs. Remove or replace site names, addresses, people, emails, free-text task
   descriptions and any URL or object key that identifies the site or the requester. Replace the
   registry's salt key with the replay's placeholder; never copy a secret. Re-seal every digest you
   changed. If re-sealing is not possible without changing what the chain reads, the job cannot be
   committed as a record.
3. Replay the redacted inputs once with `python -m blueprint_pipeline.task_evaluation_downstream_replay`
   at their final paths and record the first stop it reports. Do not take the stop from the job's
   production history.
4. Lay the files out under `records/<record_id>/` relative to a host root at
   `<materialize_root>/<record_id>/host`, gzip JSON over 32 KB, and add a manifest entry with
   `"synthetic": false`, a `provenance` line naming the retained job's internal reference (not the
   site) and the O-03 decision that permits it, the job kind, the tool, its inputs, `until` and
   `expected_stop`.
5. Run the set and `tests/test_replay_set_v1.py`. The test currently requires every record to be
   synthetic; change that assertion in the same pull request that adds the first retained record, and
   say so in its description.
