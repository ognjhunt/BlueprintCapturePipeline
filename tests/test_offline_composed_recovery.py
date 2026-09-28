"""A deterministic restart schedule across real admission, upload, billing and teardown seams."""
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from blueprint_pipeline import paid_lane_guard as guard
from blueprint_pipeline import paid_provider_allocation_lifecycle as lifecycle
from blueprint_pipeline import task_evaluation_scene_execution_authority as authority
from blueprint_pipeline import task_evaluation_scene_intake as intake
from blueprint_pipeline.vast_provider_transfer_upload import provider_output_upload_shell_fragment
from blueprint_pipeline.vast_official_billing_extractor import extract_vast_official_instance_charge, VastOfficialBillingExtractionError
from tests.test_task_evaluation_scene_intake import stage, attempt
from tests.test_vast_official_billing_extractor import _fixture, _refresh_response_binding, INSTANCE_A, LABEL_A


def test_interrupted_upload_expired_owner_posted_billing_and_recovered_teardown(tmp_path, monkeypatch):
    schedule = []
    def record(event):
        event["owner_clock_epoch"] = clock[0]
        schedule.append(event)
        (tmp_path / "injected-schedule.json").write_text(json.dumps(schedule))
    clock = [102]
    monkeypatch.setattr(guard.time, "time", lambda: clock[0])
    monkeypatch.setenv(intake.CLIENTS_ENV, "webapp")
    owner = tmp_path / "owner"
    intent = stage(owner)
    reservation = attempt(owner, intent)
    profile = {**authority.bind_scene_attempt(reservation), "source_commit": reservation["source_commit"],
        "allocator": {"max_spend_usd": 2, "argv": ["--provider", "vast"]}}
    def admit():
        authority.require_scene_execution_authority(profile, queue_root=owner, now=clock[0])
    admit()
    pending = guard.open_pending_teardown(provider="vast", lane="offline", run_id="composed",
        registry_dir=tmp_path / "pending")
    calls = {"synthetic_create": 0, "synthetic_terminate": [], "synthetic_inspect": []}
    class Provider:
        name = "vast"
        absent = False
        def terminate(self, instance):
            calls["synthetic_terminate"].append(instance)
            raise TimeoutError("delete acknowledgement lost")
        def inspect(self, instance):
            calls["synthetic_inspect"].append(instance)
            return {"http": 404 if self.absent else 503}
    provider = Provider()
    def create():
        admit()
        calls["synthetic_create"] += 1
        return {"status": "launched", "instance_id": str(INSTANCE_A)}
    adopted = lifecycle.adopt_launch_result(provider_obj=provider,
        launch=create(), pending_path=pending["path"],
        bind_pending=guard.bind_pending_teardown_instance, close_pending=guard.close_pending_teardown,
        teardown_proof_builder=lifecycle.teardown_proof_from_attempt)
    assert adopted["ready"] is True
    record({"step": 1, "state": "one_synthetic_allocation_bound", "instance": INSTANCE_A})
    source = tmp_path / "output.zip"
    source.write_bytes(b"retained-immutable-output")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    curl = fake_bin / "curl"
    curl.write_text('''#!/bin/bash
set -eu
while [ "$1" != "--upload-file" ]; do shift; done
source=$2
count=0
[ ! -f "$TEST_ROOT/transport-count" ] || count=$(cat "$TEST_ROOT/transport-count")
count=$((count + 1))
printf '%s' "$count" > "$TEST_ROOT/transport-count"
sha256sum "$source" >> "$TEST_ROOT/transport-identities"
if [ "$count" -le 3 ]; then
  head -c 5 "$source" > "$TEST_ROOT/partial-remote"
  printf 000
  exit 56
fi
cat "$source" > "$TEST_ROOT/delivered"
printf 200
''')
    curl.chmod(0o755)
    (fake_bin / "date").write_text('#!/bin/bash\nprintf "%s\\n" "${BLUEPRINT_TEST_NOW:-102}"\n')
    (fake_bin / "date").chmod(0o755)
    (fake_bin / "sleep").write_text("#!/bin/bash\nexit 0\n")
    (fake_bin / "sleep").chmod(0o755)
    command = ["bash", "-c", provider_output_upload_shell_fragment(scratch_root=str(tmp_path))
        + 'blueprint_upload_put https://fixture.invalid/immutable "$1"', "offline", str(source)]
    env = {"PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin", "TEST_ROOT": str(tmp_path)}
    first = subprocess.run(command, env=env, capture_output=True)
    assert first.returncode == 56
    assert not (tmp_path / "blueprint_provider_upload_response.json").exists()
    assert (tmp_path / "partial-remote").read_bytes() == source.read_bytes()[:5]
    record({"step": 2, "state": "three_transport_failures_outputs_retained"})
    clock[0] = 1001
    owner_bytes = {p: p.read_bytes() for p in owner.rglob("*.json")}
    with pytest.raises(authority.SceneExecutionAuthorityError, match="expired"):
        create()
    # A fresh transport process consumes the retained output; it cannot allocate.
    second = subprocess.run(command, env=env, capture_output=True)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (tmp_path / "delivered").read_bytes() == source.read_bytes()
    assert hashlib.sha256(source.read_bytes()).hexdigest() == identity
    assert (tmp_path / "transport-count").read_text() == "4"
    assert {line.split()[0] for line in (tmp_path / "transport-identities").read_text().splitlines()} == {identity}
    assert {p: p.read_bytes() for p in owner_bytes} == owner_bytes
    record({"step": 3, "state": "owner_expired_same_output_delivered", "transport_attempts": 4})
    fixture = _fixture(tmp_path / "billing")
    response = fixture["responses"][0]
    posted_bytes = response.read_bytes()
    rows = json.loads(posted_bytes)
    rows["results"] = rows["results"][1:]
    response.write_text(json.dumps(rows))
    _refresh_response_binding(fixture, 0)
    def bill():
        return extract_vast_official_instance_charge(provider_billing_source_receipt_path=fixture["receipt"],
            instance_id=INSTANCE_A, launch_label=LABEL_A)
    with pytest.raises(VastOfficialBillingExtractionError, match="unposted"):
        bill()
    record({"step": 4, "state": "billing_pending", "official_charge_usd": None})
    def teardown():
        return lifecycle.finalize_known_allocation(provider_obj=provider, instance_id=str(INSTANCE_A),
            pending_path=pending["path"], reason="offline-recovery",
            teardown_proof_builder=lifecycle.teardown_proof_from_attempt, close_pending=guard.close_pending_teardown,
            release_lane=lambda *_a, **_k: {"all_providers_terminal": provider.absent,
                "results": [{"status": "released" if provider.absent else "blocked"}]})
    assert teardown()["terminal"] is False
    assert json.loads(Path(pending["path"]).read_text())["status"] == "open"
    record({"step": 5, "state": "teardown_unproven_pending_obligation_retained"})
    response.write_bytes(posted_bytes)
    _refresh_response_binding(fixture, 0)
    charge = bill()
    assert bill() == charge and charge["official_charge_usd"] == 0.123
    provider.absent = True
    assert teardown()["terminal"] is True
    assert calls == {"synthetic_create": 1, "synthetic_terminate": [str(INSTANCE_A)] * 2,
                     "synthetic_inspect": [str(INSTANCE_A)] * 2}
    record({"step": 6, "state": "posted_charge_and_exact_synthetic_absence", "official_charge_usd": 0.123})
    assert source.read_bytes() == (tmp_path / "delivered").read_bytes()
