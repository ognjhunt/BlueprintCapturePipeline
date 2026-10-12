"""The guard reuses an entrypoint-import verdict only while nothing deciding it moved."""

from __future__ import annotations

import importlib
import re
import sys
import time
import uuid
from pathlib import Path

import pytest

import blueprint_pipeline.production_runtime_env_guard as guard
import blueprint_pipeline.production_runtime_import_verdict as verdict

ROOT = Path(__file__).resolve().parents[1]


class _Probe:
    """Two throwaway modules: an entrypoint that imports a sibling and reads data."""

    def __init__(self, root: Path) -> None:
        token = uuid.uuid4().hex[:12]
        self.root = root
        self.entry = f"bp_verdict_entry_{token}"
        self.dependency = f"bp_verdict_dep_{token}"
        self.data = root / f"bp_verdict_data_{token}.json"
        self.data.write_text('{"ok": true}\n', encoding="utf-8")
        (root / f"{self.dependency}.py").write_text("VALUE = 1\n", encoding="utf-8")
        self.write_entry("")
        self.imports = 0

    def write_entry(self, extra: str) -> None:
        (self.root / f"{self.entry}.py").write_text(
            "import json, pathlib\n"
            f"import {self.dependency}\n"
            f"DATA = json.loads(pathlib.Path({str(self.data)!r}).read_text())\n" + extra,
            encoding="utf-8",
        )

    def check(self) -> tuple[dict, list[str]]:
        # A fresh process: nothing of the probe is loaded yet.
        for name in (self.entry, self.dependency):
            sys.modules.pop(name, None)
        importlib.invalidate_caches()
        self.imports += 1
        try:
            importlib.import_module(self.entry)
        except Exception as exc:  # noqa: BLE001 - mirrors the guard's reporting
            return {"checked": [self.entry], "importable": False}, [
                f"control_plane_entrypoint_not_importable:{self.entry}:{type(exc).__name__}"
            ]
        return {"checked": [self.entry], "importable": True, "failed": []}, []

    def run(self, *, started_ns: int | None = None) -> tuple[dict, list[str]]:
        # Each run stands for a fresh unit start: the process begins now.
        return verdict.checked_entrypoint_imports(
            (self.entry,), self.check, started_ns=time.time_ns() if started_ns is None else started_ns
        )


@pytest.fixture
def probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Probe:
    package_root = tmp_path / "site"
    package_root.mkdir()
    monkeypatch.syspath_prepend(str(package_root))
    monkeypatch.setenv("BLUEPRINT_VALIDATION_VERDICT_ROOT", str(tmp_path / "verdicts"))
    # Units cannot write bytecode into the read-only release tree.
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return _Probe(package_root)


def test_a_passing_import_pass_is_reused_while_nothing_moved(probe: _Probe) -> None:
    detail, blockers = probe.run()
    assert blockers == [] and detail["verdict_reused"] is False and probe.imports == 1

    detail, blockers = probe.run()

    assert blockers == []
    assert detail["importable"] is True
    assert detail["verdict_reused"] is True
    assert probe.imports == 1


@pytest.mark.parametrize(
    "move",
    ["module_source", "dependency_source", "data_read_at_import", "new_file_in_searched_directory"],
)
def test_any_moved_import_input_recomputes(probe: _Probe, move: str) -> None:
    probe.run()
    if move == "module_source":
        probe.write_entry("# edited\n")
    elif move == "dependency_source":
        (probe.root / f"{probe.dependency}.py").write_text("VALUE = 2\n", encoding="utf-8")
    elif move == "data_read_at_import":
        probe.data.write_text('{"ok": false}\n', encoding="utf-8")
    else:
        # A new module in a searched directory can shadow a dependency.
        (probe.root / "shadow_candidate.py").write_text("", encoding="utf-8")

    detail, _blockers = probe.run()

    assert detail["verdict_reused"] is False
    assert probe.imports == 2


def test_environment_change_recomputes_but_systemd_invocation_ids_do_not(
    probe: _Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INVOCATION_ID", "first-tick")
    monkeypatch.setenv("JOURNAL_STREAM", "8:1")
    probe.run()

    monkeypatch.setenv("INVOCATION_ID", "second-tick")
    monkeypatch.setenv("JOURNAL_STREAM", "8:2")
    assert probe.run()[0]["verdict_reused"] is True

    monkeypatch.setenv("BLUEPRINT_REQUIRE_SHARED_ROBOT_EVAL_CONTRACT", "1")
    assert probe.run()[0]["verdict_reused"] is False
    assert probe.imports == 2


def test_retained_verdicts_stay_bounded_to_the_newest(
    probe: _Probe, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Each release and unit environment keys its own verdict; old ones are only cache.
    for index in range(verdict.RETAINED_VERDICTS + 3):
        monkeypatch.setenv("BLUEPRINT_PROBE_RELEASE", str(index))
        assert probe.run()[0]["verdict_recorded"] is True

    entries = sorted((tmp_path / "verdicts").glob(f"{verdict.VERDICT_NAME}-*.json"))
    assert len(entries) == verdict.RETAINED_VERDICTS
    # The newest environment is still served from its retained verdict.
    assert probe.run()[0]["verdict_reused"] is True


def test_a_failed_import_pass_is_never_stored(probe: _Probe) -> None:
    probe.write_entry("raise ImportError('missing runtime extra')\n")

    _detail, blockers = probe.run()
    detail, blockers_again = probe.run()

    assert blockers and blockers_again
    assert detail["verdict_reused"] is False
    assert probe.imports == 2


def test_excluded_per_invocation_variables_are_never_read_by_pipeline_code() -> None:
    """Leaving these out of the key cannot hide an input any module consults."""
    # Reviewed: reads INVOCATION_ID only as "running under systemd", which the
    # key still distinguishes (presence is hashed, only the value is not).
    presence_only = {("vast_independent_watchdog_control.py", "INVOCATION_ID")}
    for path in (ROOT / "src" / "blueprint_pipeline").rglob("*.py"):
        if path.name == "production_runtime_import_verdict.py":
            continue
        source = path.read_text(encoding="utf-8")
        for name in verdict.PER_INVOCATION_ENVIRONMENT:
            if re.search(rf"['\"]{re.escape(name)}['\"]", source):
                assert (path.name, name) in presence_only, (path, name)


def test_guard_reuses_the_verdict_only_for_its_own_process_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple] = []

    def fake_checked(entrypoints, check):
        calls.append(tuple(entrypoints))
        return {"checked": list(entrypoints), "importable": True, "failed": [], "verdict_reused": True}, []

    monkeypatch.setattr(guard, "checked_entrypoint_imports", fake_checked)
    for name, value in {
        "BLUEPRINT_LAUNCH_PROOF_MODE": "production",
        "PRIVACY_PIPELINE_ENABLED": "true",
        "PRIVACY_FAIL_CLOSED": "true",
        "PIPELINE_SYNC_REQUIRED": "true",
        "RETRIEVAL_REQUIRE_PRIVACY_SAFE_VIDEO": "true",
    }.items():
        monkeypatch.setenv(name, value)
    for name in ("VAST_LAUNCH_LOCK_FILE", "VAST_API_KEY_FILE", "BLUEPRINT_PIPELINE_REPO"):
        monkeypatch.delenv(name, raising=False)
    fakes = dict(
        reconcile_spend_authority=lambda: {"status": "reconciled"},
        reconcile_launch_catalog=lambda: {"status": "not_configured"},
        residency_report=lambda _root: {"status": "not_configured"},
    )

    report = guard.build_production_runtime_env_guard(**fakes)

    assert calls == [guard.CONTROL_PLANE_ENTRYPOINTS]
    assert report["control_plane_entrypoints"]["verdict_reused"] is True
    durations = report["check_durations_seconds"]
    assert set(durations) == {
        "control_plane_entrypoints",
        "spend_authority_ledger",
        "launch_profile_catalog",
        "launch_input_residency",
        "paid_launch_lock_slots",
        "runtime_source_identity",
        "capture_reconstruction_runtime",
    }
    assert all(isinstance(value, float) and value >= 0 for value in durations.values())

    # An injected environment or importer is a test of the check itself.
    calls.clear()
    guard.build_production_runtime_env_guard(
        env=dict(__import__("os").environ), import_module=lambda _name: None, **fakes
    )
    assert calls == []


class _PreloadedSite:
    """The guard's situation: packages imported before the pass filled the import caches."""

    def __init__(self, root: Path) -> None:
        token = uuid.uuid4().hex[:12]
        self.root = root
        self.package = f"bp_pre_{token}"
        self.consumer = f"bp_consumer_{token}"
        self.optional = f"bp_optional_{token}"
        package = root / self.package
        package.mkdir()
        (package / "__init__.py").write_text("", encoding="utf-8")
        (package / "mod.py").write_text("VALUE = 1\n", encoding="utf-8")
        (package / "other.py").write_text("", encoding="utf-8")
        (root / f"{self.consumer}.py").write_text(
            f"try:\n    import {self.optional}\nexcept ImportError:\n    pass\n", encoding="utf-8"
        )
        importlib.invalidate_caches()
        importlib.import_module(f"{self.package}.other")  # before the pass, as the guard's own imports are
        self.imports = 0

    def check(self) -> tuple[dict, list[str]]:
        # Like production: no cache invalidation, only a fresh import of the entrypoints.
        for name in (f"{self.package}.mod", self.consumer, self.optional):
            sys.modules.pop(name, None)
        self.imports += 1
        try:
            importlib.import_module(f"{self.package}.mod")
            importlib.import_module(self.consumer)
        except Exception as exc:  # noqa: BLE001
            return {"importable": False}, [f"not_importable:{type(exc).__name__}"]
        return {"importable": True, "failed": []}, []

    def run(self) -> tuple[dict, list[str]]:
        return verdict.checked_entrypoint_imports(
            ("preloaded",), self.check, started_ns=time.time_ns()
        )


@pytest.fixture
def preloaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _PreloadedSite:
    site = tmp_path / "site"
    site.mkdir()
    monkeypatch.syspath_prepend(str(site))
    monkeypatch.setenv("BLUEPRINT_VALIDATION_VERDICT_ROOT", str(tmp_path / "verdicts"))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    return _PreloadedSite(site)


def test_a_new_broken_optional_dependency_is_not_hidden_by_cached_listings(
    preloaded: _PreloadedSite,
) -> None:
    assert preloaded.run()[0]["verdict_recorded"] is True
    (preloaded.root / f"{preloaded.optional}.py").write_text(
        "raise RuntimeError('broken optional dependency')\n", encoding="utf-8"
    )

    detail, blockers = preloaded.run()

    assert detail["verdict_reused"] is False and blockers


def test_a_package_shadowing_an_imported_module_is_not_hidden_by_cached_listings(
    preloaded: _PreloadedSite,
) -> None:
    assert preloaded.run()[0]["verdict_recorded"] is True
    shadow = preloaded.root / preloaded.package / "mod"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('shadowed')\n", encoding="utf-8")

    detail, blockers = preloaded.run()

    assert detail["verdict_reused"] is False and blockers


def test_a_file_changed_after_its_import_refuses_the_verdict(
    probe: _Probe, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = probe.check

    def check_then_edit():
        result = original()
        probe.data.write_text('{"ok": "edited after import"}\n', encoding="utf-8")
        return result

    monkeypatch.setattr(probe, "check", check_then_edit)

    detail, _blockers = probe.run()

    # The bytes that were imported are not the bytes that could be hashed.
    assert detail["verdict_recorded"] is False


def test_a_missing_path_that_appears_as_a_file_invalidates_the_verdict(
    probe: _Probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    zipped = tmp_path / "late.zip"
    monkeypatch.setattr(sys, "path", [*sys.path, str(zipped)])  # searched, absent then
    assert probe.run()[0]["verdict_recorded"] is True

    zipped.write_bytes(b"PK\x05\x06" + b"\0" * 18)  # now a (empty) zip archive on sys.path

    assert probe.run()[0]["verdict_reused"] is False


def test_mapped_native_libraries_bind_the_verdict(
    probe: _Probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    library = tmp_path / "libfake.so.3"
    library.write_bytes(b"\x7fELF fake shared library")
    monkeypatch.setattr(verdict, "_mapped_files", lambda: {str(library)})
    assert probe.run()[0]["verdict_recorded"] is True

    library.write_bytes(b"\x7fELF upgraded shared library")

    assert probe.run()[0]["verdict_reused"] is False


def test_mapped_files_reads_only_real_paths_from_proc_maps() -> None:
    maps = (
        "7f00-7f01 r-xp 00000000 08:01 1234  /usr/lib/x86_64-linux-gnu/libssl.so.3\n"
        "7f01-7f02 r--p 00000000 00:00 0     [vdso]\n"
        "7f02-7f03 rw-p 00000000 00:00 0\n"
    )
    assert verdict._mapped_files(maps) == {"/usr/lib/x86_64-linux-gnu/libssl.so.3"}
    # A library replaced under the running process: what it runs is not on disk.
    deleted = maps + "7f03-7f04 r-xp 00000000 08:01 99    /usr/lib/libffi.so.8 (deleted)\n"
    assert verdict._mapped_files(deleted) is None


def test_a_file_changed_after_the_process_started_refuses_the_verdict(probe: _Probe) -> None:
    """The process may have loaded the earlier bytes (a pip install racing a unit start)."""
    started = time.time_ns()
    time.sleep(0.05)  # past the filesystem's coarse timestamp clock
    (probe.root / f"{probe.dependency}.py").write_text("VALUE = 1\n", encoding="utf-8")

    detail, _blockers = probe.run(started_ns=started)

    assert detail["verdict_recorded"] is False


def test_a_retargeted_symlink_invalidates_the_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = f"bp_linked_{uuid.uuid4().hex[:12]}"
    for tree, body in (("tree_a", "VALUE = 1\n"), ("tree_b", "raise RuntimeError('other tree')\n")):
        (tmp_path / tree).mkdir()
        (tmp_path / tree / f"{name}.py").write_text(body, encoding="utf-8")
    link = tmp_path / "site"
    link.symlink_to(tmp_path / "tree_a", target_is_directory=True)
    monkeypatch.syspath_prepend(str(link))
    monkeypatch.setenv("BLUEPRINT_VALIDATION_VERDICT_ROOT", str(tmp_path / "verdicts"))
    monkeypatch.setattr(sys, "dont_write_bytecode", True)

    def check() -> tuple[dict, list[str]]:
        sys.modules.pop(name, None)
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            return {"importable": False}, [type(exc).__name__]
        return {"importable": True, "failed": []}, []

    def run() -> tuple[dict, list[str]]:
        return verdict.checked_entrypoint_imports((name,), check, started_ns=time.time_ns())

    assert run()[0]["verdict_recorded"] is True
    link.unlink()
    link.symlink_to(tmp_path / "tree_b", target_is_directory=True)  # same names, other bytes

    detail, blockers = run()

    assert detail["verdict_reused"] is False and blockers


def test_a_verdict_is_reverified_at_least_hourly(probe: _Probe, monkeypatch: pytest.MonkeyPatch) -> None:
    """Inputs no audit event reports (a bare stat or which) are re-proven on a bound."""
    assert probe.run()[0]["verdict_recorded"] is True
    later = time.time() + verdict.MAX_VERDICT_AGE_SECONDS + 1
    monkeypatch.setattr(verdict, "_now", lambda: later)

    assert probe.run()[0]["verdict_reused"] is False
