# Capture V3.2 reconstruction qualification

This lane turns one guided iPhone Pro walk into a deterministic downstream
decision without changing the authority of the raw capture.

## Inputs

The cloud Capture descriptor must retain these `capture_bundle` references:

- `sync_map_uri` and `video_frame_retention_uri`
- ARKit frames, poses, intrinsics, frame quality, feature points, planes, light,
  depth, confidence, meshes, and mesh manifest
- `downstream_candidate_manifest_uri`
- `reconstruction_qualification_request_uri`
- optional `device_calibration_uri`

The qualification command additionally requires:

1. an admitted post-capture source profile;
2. the exact derived geometry candidate;
3. the exact native 3DGS candidate when appearance is available;
4. a `capture_reconstruction_evidence_profile.v1` whose digest binds the task,
   site, capture, coordinate frame, every threshold, and every failure action;
5. independently produced `capture_reconstruction_measurement.v1` rows.

There are no fallback thresholds. A profile must define all seven checks in
this order: loop closure, tracking quality, depth reprojection error, mesh
coverage, floor/support continuity, physical collision probes, and registered
Postshot reconstruction.

Each measurement binds its bytes to the exact request digest, profile digest,
source-capture digest, coordinate-frame session, candidate-manifest digest,
and—where applicable—geometry, collider, and native-appearance digests. The
qualifier must be distinct from the producing system. Postshot therefore cannot
qualify its own registration.

## Device calibration

The profile decides whether a current known-rig device calibration is required.
The gate checks the hardware model, expiry, sample count, relative error, and
median absolute deviation against the profile. A passing calibration supports
only the device sensor-scale check. It never qualifies site geometry,
collisions, registration, task success, or deployment readiness by itself.

## Decision behavior

- Scale qualifies only after loop closure, tracking, and reprojection checks
  pass, plus calibration when the profile requires it.
- Collision geometry qualifies only after the scale checks, mesh coverage,
  floor/support continuity, and exact-collider probes all pass.
- Registered reconstruction qualifies only after collision geometry and the
  independently measured Postshot-to-ARKit registration both pass.
- Any missing, mismatched, stale, self-produced, or failed measurement causes
  abstention. The result contains the first task/site profile-defined targeted
  recapture or measurement action.

The output also preserves explicit false claims for physical-site surface
proof, task success, and deployment readiness. Collision qualification here is
for the exact derived collider under the supplied probe evidence; it is not a
claim that the real site has been physically validated.

## Run

```bash
blueprint-qualify-capture-reconstruction \
  --request reconstruction_qualification_request.json \
  --evidence-profile site_task_capture_profile.json \
  --candidate-manifest downstream_candidate_manifest.json \
  --source-profile post_capture_source_profile.json \
  --geometry-candidate derived_site_geometry.json \
  --appearance-candidate native_3dgs_candidate.json \
  --measurements reconstruction_measurements.json \
  --device-calibration device_calibration.json \
  --hardware-model-identifier iPhone18,1 \
  --evaluated-at 2026-08-03T15:00:00Z \
  --output-dir evidence/reconstruction-qualification
```

Omit `--device-calibration` when the profile does not require it. Outputs are
immutable and idempotent: a byte-identical rerun succeeds, while a conflicting
file fails closed. The command writes the decision, geometry qualification,
qualified geometry, optional registration qualification, and a registered or
abstained reconstruction artifact compatible with the post-capture evidence
spine.
