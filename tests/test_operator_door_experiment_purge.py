"""The door's owner-approved purge of old experiment folders: schema, sandbox, module and script. Temp dirs only."""
# Covers: deploy/operator-door/operator_door/experiment_purge.py
# Covers: deploy/operator-door/operator_door/requests.py
# Covers: deploy/operator-door/operator_door/spool_runner.py
# Covers: deploy/operator-door/door-purge-experiment-folder.sh
# Covers: deploy/operator-door/install.sh
# Covers: scripts/operator_door.py
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import pytest

from tests.test_operator_door_scripts import _run, _write_stub, env  # noqa: F401 - env is a fixture

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy" / "operator-door"))
FOLDER = "flora-episode-benchmarks"
REQUEST_ID = "20261118T230000Z-purge-experiment-folder-0000abcd"
OLD = time.time() - 30 * 86400


def _module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------------- the schema
@pytest.mark.parametrize("folder", ["atlas-beta-dishwasher-20261021", "flora-episode-benchmarks"])
@pytest.mark.parametrize("apply", [False, True])
def test_the_door_admits_only_the_two_approved_folders(folder, apply):
    from operator_door import requests as r

    assert r.required_scope("purge-experiment-folder") == "operate"
    assert "purge-experiment-folder" in r._SCOPES
    assert r.validate_request({"kind": "purge-experiment-folder", "folder": folder, "apply": apply}) == {
        "kind": "purge-experiment-folder", "folder": folder, "apply": apply}


@pytest.mark.parametrize(("body", "code"), [
    # In the owner's list, but the G1 trial's shared inputs still point at it.
    ({"folder": "g1-841757-development"}, "purge_folder_not_approved"),
    ({"folder": "../pipeline-control-plane"}, "purge_folder_not_approved"),
    ({"folder": "flora-episode-benchmarks/repair-a92944f73137"}, "purge_folder_not_approved"),
    ({"folder": "/var/lib/blueprint/pipeline-control-plane/flora-episode-benchmarks"}, "purge_folder_not_approved"),
    ({}, "purge_folder_not_approved"),
    ({"folder": FOLDER, "apply": "yes"}, "apply_invalid"),
    ({"folder": FOLDER, "apply": 1}, "apply_invalid"),
    ({"folder": FOLDER, "path": "/"}, "request_key_unknown:path"),
])
def test_the_door_refuses_any_other_folder_or_field(body, code):
    from operator_door.requests import RequestRefused, validate_request

    with pytest.raises(RequestRefused, match=f"^{code}$"):
        validate_request({"kind": "purge-experiment-folder", **body})


def test_g1_development_is_kept_because_the_trial_still_uses_it():
    from operator_door import experiment_purge

    shared = json.loads((ROOT / "config/native_g1_shared_inputs.v1.json").read_text(encoding="utf-8"))
    assert "/g1-841757-development/" in shared["robot_asset"]["manipulation_packet_dir"]
    assert "g1-841757-development" not in experiment_purge.FOLDERS
    assert "g1-841757-development is kept" in experiment_purge.APPROVAL


# ------------------------------------------------------------------------------------- the sandbox
@pytest.mark.parametrize("apply", [False, True])
def test_the_unit_has_no_network_and_writes_only_the_one_folder_on_apply(apply):
    from operator_door.config import DoorConfig
    from operator_door.spool_runner import _LAUNCHES

    launch, config = _LAUNCHES["purge-experiment-folder"], DoorConfig()
    request = {"kind": "purge-experiment-folder", "folder": FOLDER, "apply": apply}
    properties = launch.properties(config, request)
    assert launch.script == "door-purge-experiment-folder.sh" and launch.runtime_max == "2h"
    assert {"PrivateNetwork=yes", "ProtectSystem=strict", "NoNewPrivileges=yes", "ProtectHome=yes"} <= set(properties)
    writable = next(item for item in properties if item.startswith("ReadWritePaths=")).split("=", 1)[1].split()
    results = str(Path(config.spool_root) / "results")
    assert writable == ([f"-/var/lib/blueprint/pipeline-control-plane/{FOLDER}", results] if apply else [results])
    capabilities = next(item for item in properties if item.startswith("CapabilityBoundingSet="))
    assert "CAP_SYS_ADMIN" not in capabilities and "CAP_SYS_PTRACE" in capabilities
    hidden = next(item for item in properties if item.startswith("InaccessiblePaths="))
    assert all(f"-{path}" in hidden for path in config.hidden_paths)
    assert launch.environment(config, request) == {"DOOR_INSTALL_ROOT": config.install_root,
                                                   "DOOR_PURGE_FOLDER": FOLDER,
                                                   "DOOR_PURGE_APPLY": "1" if apply else "0"}
    assert FOLDER not in launch.label(request)


def test_install_ships_the_script():
    assert '"$source_dir"/door-purge-experiment-folder.sh' in (ROOT / "deploy/operator-door/install.sh").read_text()


# ------------------------------------------------------------------------------------- the module
@pytest.fixture()
def host(tmp_path):
    parent = tmp_path / "pipeline-control-plane"
    folder = parent / FOLDER
    (folder / "repair-a" / "frames").mkdir(parents=True)
    (folder / "repair-a" / "frames" / "0001.png").write_bytes(b"x" * 5000)
    (folder / "retained.json").write_text("{}", encoding="utf-8")
    outside = tmp_path / "outside.txt"
    outside.write_text("keep me", encoding="utf-8")
    (folder / "repair-a" / "link-out").symlink_to(outside)
    (folder / "repair-a" / "dir-link-out").symlink_to(tmp_path)
    for path in [*folder.rglob("*"), folder]:
        os.utime(path, (OLD, OLD), follow_symlinks=False)
    proc = tmp_path / "proc"
    (proc / "1").mkdir(parents=True)
    (proc / "1" / "mountinfo").write_text(
        "22 1 8:1 / / rw,relatime shared:1 - ext4 /dev/sda1 rw\n", encoding="utf-8")
    return {"parent": str(parent), "folder": folder, "outside": outside, "proc": proc,
            "result": str(tmp_path / "result.json")}


def _purge(host, apply, folder=FOLDER, now=None):
    from operator_door import experiment_purge

    document = experiment_purge.run(folder, apply, host["result"], REQUEST_ID, parent=host["parent"],
                                    proc_root=str(host["proc"]), now=now)
    assert json.loads(Path(host["result"]).read_text(encoding="utf-8")) == document
    return document


def test_a_plan_measures_and_deletes_nothing(host):
    document = _purge(host, apply=False)
    assert document["outcome"] == "planned" and document["would_stop"] == []
    assert (document["files"], document["symlinks"], document["directories"]) == (2, 2, 2)
    assert [entry["name"] for entry in document["top_level_entries"]] == ["repair-a", "retained.json"]
    assert document["bytes_apparent"] >= 5002 and "g1-841757-development" in document["approval"]
    assert (host["folder"] / "repair-a" / "frames" / "0001.png").exists()


def test_an_apply_empties_the_folder_without_following_symlinks(host):
    document = _purge(host, apply=True)
    assert document["outcome"] == "emptied" and document["entries_left"] == 0
    assert host["folder"].is_dir() and list(host["folder"].iterdir()) == []
    assert host["outside"].read_text(encoding="utf-8") == "keep me"
    assert Path(host["parent"]).parent.joinpath("proc", "1", "mountinfo").exists()
    assert _purge(host, apply=True)["outcome"] == "already_empty"


def test_a_recent_write_stops_the_apply(host):
    recent = host["folder"] / "retained.json"
    os.utime(recent, (time.time() - 3600, time.time() - 3600))
    assert _purge(host, apply=False)["would_stop"] == ["folder_recently_modified"]
    document = _purge(host, apply=True)
    assert (document["outcome"], document["code"]) == ("refused", "folder_recently_modified")
    assert recent.exists()


def test_a_process_using_the_folder_stops_the_apply(host):
    pid = host["proc"] / "4242"
    (pid / "fd").mkdir(parents=True)
    (pid / "cwd").symlink_to("/elsewhere")
    (pid / "fd" / "3").symlink_to(str(host["folder"] / "retained.json"))
    (pid / "maps").write_text("", encoding="utf-8")
    (pid / "comm").write_text("python3\n", encoding="utf-8")
    document = _purge(host, apply=True)
    assert (document["outcome"], document["code"]) == ("refused", "folder_in_use_by_process")
    assert document["processes_holding"] == [{"pid": 4242, "process": "python3", "holds": "fd"}]
    assert (host["folder"] / "retained.json").exists()


def test_a_mapped_file_or_working_directory_counts_as_use(host):
    pid = host["proc"] / "77"
    (pid / "fd").mkdir(parents=True)
    (pid / "cwd").symlink_to("/elsewhere")
    library = host["folder"] / "repair-a" / "frames" / "0001.png"
    (pid / "maps").write_text(f"7f00-7f01 r--p 00000000 08:01 12 {library} (deleted)\n", encoding="utf-8")
    assert _purge(host, apply=False)["processes_holding"][0]["holds"] == "map"
    (pid / "cwd").unlink()
    (pid / "cwd").symlink_to(str(host["folder"] / "repair-a"))
    assert _purge(host, apply=False)["processes_holding"][0]["holds"] == "cwd"


def test_an_unreadable_process_stops_the_apply(host, monkeypatch):
    from operator_door import experiment_purge

    real = os.readlink

    def readlink(path, *args, **kwargs):
        if str(path).endswith("/9/cwd"):
            raise PermissionError(path)
        return real(path, *args, **kwargs)

    (host["proc"] / "9").mkdir()
    monkeypatch.setattr(experiment_purge.os, "readlink", readlink)
    assert _purge(host, apply=True)["code"] == "process_check_incomplete"


def test_a_mount_under_the_folder_stops_the_apply(host):
    escaped = str(host["folder"] / "repair-a").replace(" ", "\\040")
    with (host["proc"] / "1" / "mountinfo").open("a", encoding="utf-8") as stream:
        stream.write(f"40 22 0:50 / {escaped} rw - tmpfs tmpfs rw\n")
    document = _purge(host, apply=True)
    assert (document["outcome"], document["code"]) == ("refused", "folder_has_mount_point")


@pytest.mark.parametrize("folder", ["g1-841757-development", "..", "flora-episode-benchmarks/repair-a"])
def test_the_module_checks_the_name_again(host, folder):
    document = _purge(host, apply=True, folder=folder)
    assert (document["outcome"], document["code"]) == ("refused", "folder_not_approved")


def test_a_symlinked_folder_is_refused(host, tmp_path):
    real = tmp_path / "elsewhere"
    host["folder"].rename(real)
    host["folder"].symlink_to(real)
    document = _purge(host, apply=True)
    assert (document["outcome"], document["code"]) == ("refused", "folder_path_has_symlink")
    assert (real / "retained.json").exists()


def test_an_absent_folder_is_reported_not_failed(host):
    import shutil

    shutil.rmtree(host["folder"])
    assert _purge(host, apply=True)["outcome"] == "absent"


def test_the_module_runs_standalone_with_the_standard_library_only():
    text = (ROOT / "deploy/operator-door/operator_door/experiment_purge.py").read_text(encoding="utf-8")
    assert "from ." not in text and "import operator_door" not in text and "blueprint_pipeline" not in text


# ------------------------------------------------------------------------------------- the script
@pytest.fixture()
def purge_env(env, tmp_path):  # noqa: F811 - the scripts module's fixture
    module = tmp_path / "installed" / "operator_door" / "experiment_purge.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "import json, os, sys\n"
        "out = sys.argv[sys.argv.index('--result-out') + 1]\n"
        "open(os.environ['STUB_LOG'], 'a').write('purge ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        "json.dump(json.loads(os.environ['FAKE_PURGE']), open(out, 'w'))\n"
        "sys.exit(int(os.environ.get('FAKE_PURGE_RC', '0')))\n", encoding="utf-8")
    return {**env, "DOOR_INSTALL_ROOT": str(tmp_path / "installed"), "DOOR_PURGE_FOLDER": FOLDER}


@pytest.mark.parametrize(("apply", "row", "rc", "status", "code"), [
    ("0", {"outcome": "planned"}, 0, "planned", None),
    ("1", {"outcome": "emptied"}, 0, "emptied", None),
    ("1", {"outcome": "absent", "code": "folder_absent"}, 0, "absent", None),
    ("1", {"outcome": "refused", "code": "folder_in_use_by_process"}, 1, "refused", "folder_in_use_by_process"),
    ("1", {"outcome": "partly_emptied", "code": "delete_failed:EPERM"}, 1, "failed", "delete_failed:EPERM"),
    ("1", {"outcome": "bogus"}, 0, "failed", "purge_failed"),
])
def test_the_script_runs_the_installed_module_and_reports_its_outcome(purge_env, apply, row, rc, status, code):
    done, outcome, calls = _run("door-purge-experiment-folder.sh", purge_env, DOOR_REQUEST_ID=REQUEST_ID,
                                DOOR_PURGE_APPLY=apply, FAKE_PURGE=json.dumps(row), FAKE_PURGE_RC=str(rc))
    assert (outcome["status"], outcome["code"]) == (status, code)
    assert done == (0 if status in {"planned", "emptied", "absent"} else (rc or 1))
    purge_call = next(call for call in calls if call.startswith("purge "))
    assert purge_call.startswith(f"purge --folder {FOLDER} --request-id {REQUEST_ID}")
    assert ("--apply" in purge_call) is (apply == "1")
    assert outcome["result"].endswith(f"{REQUEST_ID}.purge-experiment-folder.json")


@pytest.mark.parametrize(("extra", "code"), [({"DOOR_PURGE_FOLDER": "g1-841757-development"}, "purge_folder_not_approved"),
                                             ({"DOOR_PURGE_APPLY": "yes"}, "apply_invalid")])
def test_the_script_checks_the_folder_and_mode_before_running_anything(purge_env, extra, code):
    done, outcome, calls = _run("door-purge-experiment-folder.sh", {**purge_env, "DOOR_PURGE_APPLY": "0", **extra},
                                DOOR_REQUEST_ID=REQUEST_ID, FAKE_PURGE="{}")
    assert (done, outcome["status"], outcome["code"]) == (2, "refused", code)
    assert not any(call.startswith("purge ") for call in calls)


# ------------------------------------------------------------------------------------- the client
@pytest.mark.parametrize("apply", [False, True])
def test_the_client_sends_one_named_folder(monkeypatch, apply):
    client = _module(ROOT / "scripts" / "operator_door.py", "operator_door_client_purge")
    sent = []
    monkeypatch.setattr(client, "_submit", lambda body, args: sent.append(body) or 0)
    argv = ["purge-experiment-folder", FOLDER] + (["--apply"] if apply else [])
    assert client.run(client.build_parser().parse_args(argv)) == 0
    assert sent == [{"kind": "purge-experiment-folder", "folder": FOLDER, "apply": apply}]
    with pytest.raises(SystemExit):
        client.build_parser().parse_args(["purge-experiment-folder", "g1-841757-development"])


@pytest.mark.parametrize(("status", "verdict"), [("planned", True), ("emptied", True), ("already_empty", True),
                                                 ("absent", True), ("refused", False), ("failed", False)])
def test_the_waiting_client_treats_each_purge_outcome_as_done_or_failed(status, verdict):
    client = _module(ROOT / "scripts" / "operator_door.py", "operator_door_client_purge_wait")
    state = {"request": {"kind": "purge-experiment-folder"}, "outcome": {"status": status}}
    assert client._terminal(state) is verdict
    # Other kinds keep their own sets: an "absent" retirement is not a success.
    assert client._terminal({"request": {"kind": "retire-scene-workspace"}, "outcome": {"status": "absent"}}) is False
