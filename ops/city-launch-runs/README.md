# City-launch evidence storage boundary

Generated city-launch runs are not source artifacts and must not be committed here. The
harness stores them outside the checkout under `BLUEPRINT_CITY_LAUNCH_OUTPUT_ROOT` or,
by default, the user's private state directory.

Each run uses schema `city-launch-harness-run.v2`, a private `0700` root, a complete
relative-path/SHA-256/size inventory, a seven-day launch-decision freshness limit, and a
365-day deletion date. Validate a run with:

```bash
python scripts/validate_city_launch_evidence.py /private/artifact/root/<city>/<run-id>
```

The manifest is internal operational evidence and is not approved for external
disclosure. It can contain local paths and service metadata. An external export requires
an independently reviewed redaction/approval process and must pass
`--require-disclosure-approval`. Artifact integrity and freshness do not prove launch
readiness; the current release evidence graph remains authoritative.
