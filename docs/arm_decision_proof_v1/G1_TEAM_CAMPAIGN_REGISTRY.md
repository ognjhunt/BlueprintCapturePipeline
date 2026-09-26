# G1 team campaign registry

ADP-009D day-21 development gate: bind one authenticated team to an exact,
retained Scene 841757 task packet so its members can choose Unitree G1 and the
approved book-manipulation or movement policy pair in the WebApp. This registry
only enables owner-scoped planning and request intake. It does not authorize a
GPU launch, qualify a policy result, or grant public media rights.

The operator obtains `owner.user_id` and `owner.organization_id` from the
authenticated WebApp scene-intake record for the same site and task. Neither
field is derived from an email address or invented by the Pipeline. A binding
contains these exact fields:

- `owner`: `user_id` and `organization_id` strings.
- `scene_id`, `task_id`, and `source_packet_receipt_digest`: the verified task
  packet identity shown by `make_packet_planning_setup`.
- `source_packet_dir`, `manipulation_packet_dir`, `movement_packet_dir`, and
  `publisher_source_dir`: absolute retained directories.
- `navigation_authority_path` and `runtime_source_receipt_path`: absolute
  retained files.
- `rights_review_paths`: one absolute review file for each of the four
  `native_g1_development_pair.PAIR_ORDER` candidates.

Write an array of one or more bindings to a private JSON file. From the deployed
Pipeline checkout, run:

```bash
PYTHONPATH=src /opt/blueprint/BlueprintCapturePipeline/.venv/bin/python \
  -m blueprint_pipeline.native_g1_team_campaign_registry \
  --bindings-file /absolute/path/to/reviewed-bindings.json \
  --output /var/lib/blueprint/task-evaluation-inputs/g1-team-campaign-registry.json
```

The output path must be new and its parent must already exist. The command
checks each owner, duplicate binding, retained path, and source task packet,
then writes the digest-bound registry and rereads each owner's setup catalog.
It performs no provider mutation. The signed intake and later dispatcher
recheck the installed registry; a changed or unavailable binding fails closed.

Keep the dispatcher timer in `execute=false` until the exact runtime dependencies
are reviewed, the merged release is deployed, and the shared paid-resource gate
admits the bounded run. Media remains private and `development_only`.
