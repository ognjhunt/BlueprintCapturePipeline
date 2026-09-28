# Governance for the reviewed lane writers

This gate protects the known lane constructor, descriptor capability, G1 output
admission, Arena attempt admission and payload directory command, Arena launch
shell entrypoint, and installer lane-parent provisioning. It does not complete
Plan 12's historical writer inventory or authorize any deletion or deployment.

`lane-scratch-writer-manifest.json` records whole-file SHA-256 identities,
responsibilities, admission methods, characterization tests, review references,
and historical exceptions. Missing, changed, duplicate, unreadable, escaping, or
symlinked source/test paths fail. Any governed source change requires independent
spec and quality review before accepting its new digest. A matching digest proves
that pinned bytes are unchanged; CI cannot attest that the review occurred or
make a deliberately altered manifest trustworthy. PR review remains a separate
gate. The implementation review reference is finalized before merge.

The manifest check is an always-on impacted-test sentinel. It reads source bytes
only, without importing writers, running shell scripts, contacting providers, or
changing the host.

## Bounded discovery coverage

The verifier scans `.py` files below `src/blueprint_pipeline`, `scripts`, and
`deploy`, and `.sh` files below those directories. It flags:

- Python calls to `create_lane_scratch` and `create_leased_lane_scratch`, including
  direct imports, `as` aliases, module-qualified imports, and simple assignments
  of a constructor to another name.
- A Python file containing a directory call named `mkdir`, `makedirs`, `mkdtemp`,
  or `copytree` and a statically visible configured lane-root reference. Supported
  paths are string literals, `Path(...)`, single static variable assignments,
  string addition, `/` joins, and f-strings whose pieces all resolve statically.
- Shell tokens containing the literal configured lane roots, or
  `$WORK_VOLUME_ROOT/lanes`, `${WORK_VOLUME_ROOT}/lanes`,
  `$TASK_EVALUATION_INPUT_ROOT/lanes`, or `${TASK_EVALUATION_INPUT_ROOT}/lanes`,
  in a file with `mkdir` or `install -d` tokens.
- Shell token pairs `blueprint_pipeline.control_plane_arena_scratch prepare`.

This deliberately conservative association is at file scope, not a dataflow
proof: unrelated root references and directory operations may require an explicit
reviewed manifest entry. A read-only Python root reference alone does not count
as a writer. Shell tokenization is lexical; Bash syntax validation remains a
separate check.

Discovery cannot resolve arbitrary CLI/environment paths, imported path values,
multiple assignments or shadowing, dynamic `getattr`, `eval`, generated code,
foreign language writers, indirect shell wrappers, or commands constructed in
unrecognized strings. It does not follow directory symlinks during discovery.
Manifest entries themselves reject symlink ancestors. Dynamic noncoverage has
explicit fixtures; passing discovery is not proof that every writer is registered.

## Runtime directory boundary

`LeasedScratchDirectory` binds to exact root/lane/folder device and inode values,
an owner, the exact reference key and value, and a sealed lease digest. Creation
validates the full root before mutation and publishes through its retained
descriptor. Under the same root coordination lock, payload directory operations
reopen and verify the visible paths, require a live unreleased lease, and walk
relative payload components with `O_NOFOLLOW`.

Renewal changes the bound digest. An explicit `refresh()` validates the same
owner/reference and directory identity before adopting it. Expiry, release,
changed identity, malformed leases, traversal, and symlinks refuse writes.
Handles are context managers and must be closed.

G1 uses the handle for its diagnostics directory below a lane root. It still
refuses an existing output; there is no retry-in-place. Arena's payload command
reopens its admitted lease and preserves existing payload receipts. A retry
without supplied metadata uses the exact identity in the validated saved lease;
supplied metadata must match. Historical Arena paths remain readable only under
the existing proof rules and cannot receive a silent lease or payload write.

Returned `Path` values, plain `cp`, and downstream file writes are outside this
descriptor guarantee. Advisory locks coordinate participating writers; another
process with the same filesystem permissions can bypass them. Universal host
prevention needs root-owned lane parents and a privileged creation broker in a
separately approved operational change.
