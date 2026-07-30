# Protocol Amendment 3: Runtime-Source Symlink Handling

Frozen prospectively: 2026-07-30T11:02:00-0500

## Triggering evidence

The exact GitHub codeload archives for both `ea1293d2565399d4823e5e6a0c2b76141dff4347`
and `69056df5b54272497166de6a18aa210f9a21da08` contain the same two repository
symlink members:

- `.agents/skills/gstack`;
- `.claude/skills/gstack`.

Both point to local developer-tool installations outside the repository. They
are unrelated to the GPU runtime source tree. The source-overlay extractor
correctly refused to extract or follow links, but incorrectly treated the mere
presence of these frozen members as terminal. This latent failure was found by
offline archive inspection before a successor GPU was created.

## Frozen extraction rule

The overlay continues to reject absolute paths, traversal, multiple or wrong
roots, hard links, devices, FIFOs, and every other non-file/non-directory member.
It may skip, without extracting or following, only the two exact symlink paths
listed above. Any other symlink is terminal. The imported runtime must still
come from the verified archive's `src` tree, and the required bootstrap module
must be a regular file.

This is a source-transport correction, not a scientific threshold change. It
does not permit generated source, mutable branch archives, or unbound files.

## Retry identity

Because the extractor source changes, the SHA and input bundle produced after
Protocol Amendment 2 are superseded before use. The next live attempt requires
another clean pushed commit, a newly downloaded exact-commit codeload archive,
a new digest, and another versioned signed input and object-store key. All prior
archives and failed-attempt evidence remain preserved.

## Claim boundary

Successful source extraction proves only safe loading of the hash-bound runtime
overlay. It does not prove checkpoint loading, learned-policy inference, WAM
execution, repeated policy re-query, causal qualification, ranking, abstention,
physical outcomes, transfer, economics, or the thesis.
