# Configure beta policy screening

The reviewed default is `config/beta_policy_candidates.v1.json`. Set
`BLUEPRINT_BETA_POLICY_SCREENING_CATALOG_PATH` to an absolute, operator-owned
JSON file to replace it. The service accepts a root-owned or service-owned
regular file that is not writable by group or others. It rejects a symlink at
the file path. Keep the parent directory under operator control.

The catalog can hold dozens of distinct configurations. Each row pins its
checkpoint or hosted policy identity and declares its robot, cameras, image
processing, control rate, action horizon, normalization, runtime, license and
access requirements. A missing capability, unverified checkpoint, unknown
price or missing account stays visible as an unavailable or deferred candidate.
Enabling a row does not grant runtime admission. The launch bridge must validate
its exact native policy artifacts and prepared scene independently.

Hosted execution also needs an admitted adapter, an operator private credential
file where required, and a reviewable price per HTTP request with a bounded
request count. The launch descriptor reserves the simulator provider cap plus
the maximum API charge within the same site budget. A zero API charge must be
explicitly marked `free` or `caller_paid`; an unknown price is not free. Private
credential paths and contents never belong in the catalog or website results.

Run the CLI with the installed Pipeline Python environment, from its checkout:

```sh
python -m blueprint_pipeline.beta_policy_candidate_catalog validate
python -m blueprint_pipeline.beta_policy_candidate_catalog inspect
python -m blueprint_pipeline.beta_policy_candidate_catalog validate --catalog /absolute/operator/catalog.json
```

To change a catalog, copy it to an operator-owned draft, edit the candidate data
and bump `revision`. Seal to a **new** explicit output file, then inspect it:

```sh
python -m blueprint_pipeline.beta_policy_candidate_catalog seal --input /absolute/operator/catalog.draft.json --output /absolute/operator/catalog.next.json
python -m blueprint_pipeline.beta_policy_candidate_catalog inspect --catalog /absolute/operator/catalog.next.json
```

`seal` recalculates each configuration digest and the catalog digest. It refuses
to overwrite any existing output file. `validate` rejects changed data with an
old digest. Install the reviewed new file through the normal operator config
deployment and point the service at it.

Campaign plans are immutable. When access, runtime admission or cost settings
become ready after an unavailable or deferred campaign, bump the catalog
revision and reseal it to request a new plan. Merely editing a runtime setting
does not rewrite an existing plan. The fresh campaign uses the same persisted
**$15 budget for the site submission**, so a new revision, scene revision or
worker restart cannot reset that allowance. Quotes expire before new launches;
retained plans and results remain readable after expiry.

The planner gives selected candidates bounded screening cells (ten by default),
with a separate seed domain from finalist evaluation. Native whole-body G1
uses the same cell IDs and seeds with explicitly frozen scene geometry; its
cells do not claim the fixed-arm placement/camera perturbations. These
are diagnostic results, not the 200-episode evidence required for a recommended
pilot. The published catalog and plan JSON schemas live under `docs/schemas/`;
runtime validation additionally enforces the identity bindings and budget math.

For objects installed at a fixed place in the scene, screening preserves their
root position and orientation. Each omitted position or yaw draw stays in the
sealed cell's `runtime_coverage_gaps` and native episode evidence, explicitly
marked as untested. Repeating that position does not establish placement or yaw
robustness. All candidates receive the same constrained cells and seeds.
