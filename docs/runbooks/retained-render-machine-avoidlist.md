# Retained-render machine exclusions

ADP-009D/day-28 operator input handling: retain the selected-offer and adapter
receipts before classifying a failed machine. A completed bootstrap input-download
timeout is an observed machine-to-store transport failure, not a permanent
provider outage or a hardware diagnosis.

The render lane copies a profile-bound avoidlist into
`<attempt-root>/provider_machine_avoidlist.json`. Adapter learning updates that
attempt output; it does **not** update the shared source or automatically affect
another attempt. Reopen the retained evidence, then explicitly publish and bind a
successor snapshot when preparing the next profile. Preserve the original bytes.

For a root-operated successor, use the existing copy, exclusion writer, and
publication permission/readback helpers in this order. Here `source`, `target`,
`selected_offer`, `instance_id`, and `blockers` come from the inspected attempt;
`target` must be a new path under the approved production input root:

```python
import hashlib
from pathlib import Path
from blueprint_pipeline.core.common import utc_now_iso, write_json
from blueprint_pipeline.provider_machine_avoidlist import stage_machine_avoidlist_for_attempt
from blueprint_pipeline.vast_provider_adapter import _machine_avoidlist_reason, _record_machine_avoidlist_entry
from scripts.publish_task_evaluation_launch_profiles import _service_identity, _seal_published_profile

source, target = Path(source), Path(target)
assert source.resolve() != target.resolve() and not target.exists()
original = source.read_bytes()
reason = _machine_avoidlist_reason(blockers)
assert reason is not None
stage_machine_avoidlist_for_attempt(source_path=source, destination_path=target)
_record_machine_avoidlist_entry(path=target, generated_at=utc_now_iso(),
    selected_offer=selected_offer, instance_id=instance_id, blockers=blockers, reason=reason)
digest = "sha256:" + hashlib.sha256(target.read_bytes()).hexdigest()
account, group, uid, gid = _service_identity(target.parent, "blueprint", "blueprint")
_seal_published_profile(target, expected_digest=digest, account=account, uid=uid, gid=gid)
assert source.read_bytes() == original
write_json(target.with_name(target.name + ".service-readback.json"), {
    "successor_path": str(target), "sha256": digest, "size_bytes": target.stat().st_size,
    "service_account": account, "service_group": group, "mode": "0440",
    "full_byte_service_account_readback_passed": True, "original_input_unchanged": True,
})
```

Run from the deployed repository so the existing `scripts` helpers resolve.
`_seal_published_profile` installs group readability and hashes the actual file as
the service account; a root-process read is insufficient. If it fails, do not
report the successor ready or publish a profile that references it. Only bind the
new path and digest after this native readback succeeds.
