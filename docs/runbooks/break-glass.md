# Break glass: changing the control-plane host outside the door

The operator door (`docs/OPERATOR_DOOR.md`) is the only supported way to change
the control-plane host. Breaking glass means changing the host some other way:
by hand over SSH, or by deploying from a source the door would not use. It
stays possible, but it must leave a sealed note, and the next deploy reports
that note.

Why: on 2026-09-26 a deploy ran from a scratch checkout
(`/mnt/blueprint-work/control-plane-deploy-sources/...`) through a transient
unit. GPU admission cannot verify a release deployed from such a source, so for
hours it refused every sponsored GPU step
(`gpu_canary_deployed_release_receipt_unverified`), until a door deploy of a
newer commit replaced the release. Around it, operators and agents changed the
host by hand, and nothing recorded what they did.

## When SSH is allowed

Only when the door cannot do the job:

- the door itself is down or broken (its service, its runner, its Caddy route
  or its tokens) and has to be repaired;
- the host is in a state the door cannot fix, for example a disk so full that
  the door cannot write its request spool;
- something urgent that the door does not offer, such as stopping spend on a
  paid resource that automatic teardown missed.

Keep the change as small and as reversible as the situation allows, and go back
to the door as soon as it works. Anything the door offers (a deploy of a merged
commit, the unit actions listed in `docs/OPERATOR_DOOR.md`) is a door request,
not a reason to SSH.

## Record a note

Run this with sudo right after the change (for a deploy from an untrusted
source, right before it). Record one note per change, and name every action and
every path you touched.

```bash
cd /opt/blueprint/task-evaluation-control-plane
sudo env PYTHONPATH=src /opt/blueprint/BlueprintCapturePipeline/.venv/bin/python \
  -m blueprint_pipeline.control_plane_break_glass record \
  --reason "door upgrade left the runner failing; restored the previous door code by hand" \
  --action door-rollback --action unit-restart \
  --path /opt/blueprint/operator-door
```

- `--reason` is required: at most 500 printable characters on one line.
- `--action` is required and repeatable: a lowercase slug such as
  `unit-restart`, `unit-stop`, `config-edit`, `door-rollback` or
  `deploy-from-untrusted-source`. Only `deploy-from-untrusted-source` means
  anything to tooling (see below); the others are for whoever reads the
  receipt.
- `--path` is optional and repeatable: an absolute path that was changed.
- The operator is taken from `SUDO_USER`, else `USER`. The note also records
  `sudo_user`, the SSH client address (the first field of `SSH_CONNECTION`),
  the host name and the time.

The command prints the note and its path,
`/var/lib/blueprint/pipeline-control-plane/cleanup-receipts/<UTC time>-<digest12>.json`.
The note is sealed: `note_digest` is the sha256 of its canonical JSON, and the
file name carries the creation time and the digest prefix, so an edited or
renamed note no longer verifies. The digest catches edits; it is not a
signature. The directory is root-owned and `0755`, and every file in it is
`0644`, because the door runs as `blueprint`. No field name looks like a
credential, so the door's secret scanner lets it serve them.

To see what the next deploy will report:

```bash
sudo env PYTHONPATH=src /opt/blueprint/BlueprintCapturePipeline/.venv/bin/python \
  -m blueprint_pipeline.control_plane_break_glass list
```

## Deploying from an untrusted source

Before it reserves disk, writes provenance or moves anything,
`scripts/deploy_control_plane_commit.py` refuses a `--source-repo` that GPU
admission would not trust. Admission trusts a deploy receipt only when its
source checkout is:

- the canonical checkout, `/opt/blueprint/BlueprintCapturePipeline`, which is
  what the iteration and canary wrappers pass; or
- a root-owned clone that is not group- or world-writable, is not a symlink, and
  is a direct child of `/opt/blueprint/control-plane-config-tools`, like the
  door's `operator-door-source` clone.

Any other source is refused with
`deploy_source_repo_untrusted:<note code>: GPU admission would refuse the resulting release ...`.
If you still have to deploy from it, record a note with
`--action deploy-from-untrusted-source` and pass it to the deploy within 24
hours:

```bash
sudo /opt/blueprint/BlueprintCapturePipeline/.venv/bin/python <checkout>/scripts/deploy_control_plane_commit.py \
  --source-repo <checkout> \
  --source-commit <sha> \
  --release-root /opt/blueprint/task-evaluation-control-plane-releases \
  --state-root /var/lib/blueprint/pipeline-control-plane \
  --active-link /opt/blueprint/task-evaluation-control-plane \
  --iteration \
  --break-glass-note /var/lib/blueprint/pipeline-control-plane/cleanup-receipts/<note>.json \
  --receipt-out /var/lib/blueprint/pipeline-control-plane/deploy-receipts/iteration_<sha8>_break_glass.json
```

The deploy receipt records the note as `break_glass_note` (name, path, digest,
operator, reason, created_at, actions). The note does not change what GPU
admission trusts: sponsored GPU steps on that release keep refusing with
`gpu_canary_deployed_release_receipt_unverified` until a deploy from a trusted
source replaces it. Follow up with a door deploy as soon as the fix is on
`origin/main`.

| Note code | Meaning |
|---|---|
| `break_glass_note_missing` | No `--break-glass-note` was given. |
| `break_glass_note_unreadable` | The note path does not name a readable regular file. |
| `break_glass_note_expired` | The note is more than 24 hours old. Record a new one. |
| `break_glass_note_from_the_future` | The note is dated more than five minutes ahead of the host clock. |
| `break_glass_note_action_missing` | The note's actions do not include `deploy-from-untrusted-source`. |
| `break_glass_note_digest_mismatch` | The note was edited after it was sealed. |
| `break_glass_note_name_mismatch` | The note was renamed, or copied under another name. |
| `break_glass_note_not_json`, `break_glass_note_schema_invalid`, `break_glass_note_fields_invalid` | The file is not an intact note. |

## What the next deploy reports

Once every surface has moved, each CLI deploy lists the notes that no earlier
deploy reported:

- the receipt gets `break_glass_notes`, one row per note with `name`, `digest`,
  `operator`, `reason` and `created_at`. A note that does not verify is still
  listed, with `digest: null` and its `error` code, so damage is reported
  rather than hidden;
- the receipt's `alerts` gets `break_glass_notes_reported:<n>`;
- each note is appended to `cleanup-receipts/reported.jsonl` with the deploy
  commit, so it is reported once.

Reporting never fails a deploy. If the notes directory cannot be read, the
receipt has `break_glass_notes: null`, a `break_glass_notes_error` code and the
alert `break_glass_notes_unreadable:<code>`. If the notes cannot be marked (on
a full disk, say), the alert is `break_glass_notes_not_marked:<code>`, and the
next deploy reports them again.

## Never delete evidence by hand

Break glass is not a way to free disk. Evidence is offloaded behind a pointer,
never deleted by hand: use the reclaim and offload tools in
`docs/CONTROL_PLANE_STORAGE.md`. The notes are themselves `evidence_hot`
(`src/blueprint_pipeline/control_plane_storage_roots.py`), so never edit, move
or delete a note or `reported.jsonl`. To correct a mistaken note, record
another note that says so.
