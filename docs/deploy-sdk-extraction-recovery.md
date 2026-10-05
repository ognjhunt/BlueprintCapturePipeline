# Bound repeated wheel parsing during deployment

ADP-081, day-42 prerequisite: complete the reviewed immutable usage replay
release (#2597). This is installation recovery, not economics-gate completion.

After #2598, the host fetched the existing pinned dependency and produced its
locked SDK manifest. Four individually observed attempts of the same release
failed before activation while retained extraction advanced. The manifest has
21,085 files, including 5,620 in one wheel. The wrapper exposes only
deploy_scene_retirement_runtime_unproven; an internal deadline failure is not
directly observed. Source inspection finds a separate, reproducible cost:
reopening ZipFile reparses its entire central directory for every missing file.

Contract before editing:

- Reuse one parsed archive while extracting consecutive entries from that wheel.
  Keep descriptor use bounded and close it on archive changes and all exits.
- Continue the original protected ancestry open for every missing member, bind
  the cached reader to that exact file identity, and retain every size/hash,
  inode, deadline, disk floor, partial-prefix and immutable publication check.
- Existing complete files still require exact validation; partial files resume
  only their validated prefix. No timeout increase, dependency removal, authority
  change, cleanup, extra access or paid execution.
- Reproduce repeated metadata parsing using a many-member real wheel; exercise
  archive identity changes, archive switching and resource closure on refusal.
  Run the existing installer suite, independent exact-head review and required CI.
- Completion remains the normal deployment receipt, installed pins and module
  bytes. Prior active release and explicit stops must survive any refusal.

The isolated extraction repair has no competing open-PR owner in the fresh
ownership check. It is limited to the same deployment prerequisite.
