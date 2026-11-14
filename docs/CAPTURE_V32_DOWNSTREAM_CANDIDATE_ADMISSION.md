# Capture V3.2 downstream-candidate admission

Pipeline accepts `downstream_candidate_manifest.v1` as a provider-neutral index
of retained iPhone observations. Local materialization validates and projects
the manifest URI/digest, exact raw-video hash evidence, video identity, and
coordinate-frame identity; it does not copy the registry into an agent prompt
and does not authorize a provider upload.

`build_capture_v32_reconstruction_admission()` then requires a
`task_site_frame_selection_profile.v1` bound to the exact candidate-manifest
digest and a current authoritative rights/revocation evidence digest. Supported
deterministic selectors are explicit encoded-frame ordinals, decoded-PTS
coverage, and an explicit tracking/relocalization/pose-eligibility quality
filter. There is no Capture or Pipeline default. Without the profile, the
stable smallest blocker is:

```text
task_site_evidence_profile_with_frame_selection_parameters
```

An admitted result instructs the existing reconstruction path to materialize
the exact decoded candidate images before frozen train/validation/held-out
splits. Its claim ceiling is retained-observation selection only. Subsequent
ARKit scaffold, reconstruction, held-out appearance, metric, collision,
OpenUSD, Isaac, Task Evaluation Run, and physical/deployment gates remain
separate.

The versioned fixture is
`tests/fixtures/capture_v32_downstream_candidate_manifest.json`. It contains no
real media, is byte-identical to Capture's V3.2 candidate fixture, and proves
only contract parsing, digest binding, and deterministic admission behavior.
