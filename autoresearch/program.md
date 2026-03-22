# BlueprintCapturePipeline Autoresearch Program

This harness optimizes exactly one skill directory at a time.

Rules:

- Mutate only files listed in the selected target manifest.
- Default mutable file is the target `SKILL.md`.
- Optional mutable files may exist only inside the same target skill directory.
- Never edit runtime code, providers, harness code, tests, fixtures, deployment files, or adjacent skills.
- Keep mutations small and surgical.
- Every mutation must include a short hypothesis and a short change summary.
- Acceptance is score-only: higher score wins.
- Ties are accepted only when the candidate diff is smaller than the current best diff.
- Reject candidates with forbidden file edits, no meaningful diff, malformed output, or missing required artifacts.
- Store all iteration artifacts under `autoresearch/runs/...`.
- Do not write the winning candidate back into `skillpacks/`.
