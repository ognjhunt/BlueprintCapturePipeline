# Covers (for impacted-test selection):
#   src/blueprint_pipeline/remote_cpu_environment.py
#   src/blueprint_pipeline/task_evaluation_production_chain_preflight.py
"""ADP-009D/day-28, plan 14 PR 1: the host's compile environment is measured before any worker exists."""

from __future__ import annotations

import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import types
import zipfile
import zlib
from pathlib import Path

from blueprint_pipeline import remote_cpu_environment as environment
from blueprint_pipeline import task_evaluation_production_chain_preflight as preflight
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.remote_cpu_job_contract import forbidden_record_content

# The corpus is part of the host/worker parity contract: changing it changes every golden digest.
GOLDEN_CORPUS_SHA256 = "be44e31f6f9610571eaa5eabfb10a00ccba5e1da7f3d537a548cfb75b3d4f1fb"
FACTS_DISTRIBUTIONS = {
    "numpy", "usd-core", "msgpack", "jsonschema", "referencing", "rpds-py", "attrs", "webcolors", "pydantic",
    "pydantic-core", "pillow", "defusedxml", "packaging",
}


def _cpuinfo(path: Path, flags: str) -> Path:
    path.write_text(
        f"processor\t: 0\nvendor_id\t: GenuineIntel\nflags\t\t: {flags}\n\nprocessor\t: 1\nflags\t\t: {flags}\n",
        encoding="utf-8",
    )
    return path


def _raw_level6(data: bytes) -> bytes:
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    return compressor.compress(data) + compressor.flush()


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def test_environment_records_cpython_zlib_golden_deflate_simd_and_distributions(tmp_path: Path, monkeypatch) -> None:
    cpuinfo = _cpuinfo(tmp_path / "cpuinfo", "sse2 avx2 fma sse2")
    record = environment.environment_record(cpuinfo_path=cpuinfo)

    assert record["schema_version"] == "remote_cpu_environment.v1"
    assert record["interpreter"] == {
        "implementation": platform.python_implementation(), "version": sys.version,
        "machine": platform.machine(), "libc": list(platform.libc_ver()),
    }
    corpus = environment.golden_corpus()
    assert hashlib.sha256(corpus).hexdigest() == GOLDEN_CORPUS_SHA256
    assert record["zlib"] == {
        "runtime_version": zlib.ZLIB_RUNTIME_VERSION, "compile_version": zlib.ZLIB_VERSION,
        "corpus_digest": _sha(corpus), "zlib_level6_digest": _sha(zlib.compress(corpus, 6)),
        "raw_deflate_level6_digest": _sha(_raw_level6(corpus)),
    }
    # The raw digest is exactly what a level-6 zip member holds, as the packet builder writes them.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        info = zipfile.ZipInfo("member.bin", date_time=(1980, 1, 1, 0, 0, 0))
        archive.writestr(info, corpus, compress_type=zipfile.ZIP_DEFLATED, compresslevel=6)
    with zipfile.ZipFile(buffer) as archive:
        member = archive.getinfo("member.bin")
    data, offset = buffer.getvalue(), member.header_offset
    start = offset + 30 + int.from_bytes(data[offset + 26:offset + 28], "little") + int.from_bytes(
        data[offset + 28:offset + 30], "little")
    assert _sha(data[start:start + member.compress_size]) == record["zlib"]["raw_deflate_level6_digest"]

    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:  # numpy 1.x
        from numpy.core import _multiarray_umath as umath
    assert record["cpu"] == {
        "numpy_simd": {
            "baseline": sorted(umath.__cpu_baseline__), "dispatch": sorted(umath.__cpu_dispatch__),
            "enabled": sorted(name for name, on in umath.__cpu_features__.items() if on),
        },
        "flags": ["avx2", "fma", "sse2"], "flags_source": str(cpuinfo),
    }
    assert record["cpu_class"] == canonical_digest(
        {"machine": platform.machine(), "numpy_simd": record["cpu"]["numpy_simd"], "flags": ["avx2", "fma", "sse2"]}
    )

    assert set(environment.COMPILE_DISTRIBUTIONS) == FACTS_DISTRIBUTIONS
    assert [row["name"] for row in record["distributions"]] == sorted(FACTS_DISTRIBUTIONS)
    for row in record["distributions"]:
        assert row["version"] == importlib.metadata.version(row["name"]), row
    parity = {name: record[name] for name in ("interpreter", "zlib", "distributions")}
    assert record["environment_digest"] == canonical_digest(parity)
    assert record["executable"] == sys.executable

    other_cpu = environment.environment_record(cpuinfo_path=_cpuinfo(tmp_path / "other", "sse2"))
    unmeasured = environment.environment_record(cpuinfo_path=tmp_path / "absent")
    assert other_cpu["environment_digest"] == unmeasured["environment_digest"] == record["environment_digest"]
    assert len({record["cpu_class"], other_cpu["cpu_class"]}) == 2
    assert (unmeasured["cpu_class"], unmeasured["cpu"]["flags"], unmeasured["cpu"]["flags_source"]) == (None, [], None)
    missing = environment.environment_record(
        cpuinfo_path=cpuinfo, distributions=(*environment.COMPILE_DISTRIBUTIONS, "definitely-not-installed")
    )
    assert {"name": "definitely-not-installed", "version": None} in missing["distributions"]
    assert missing["environment_digest"] != record["environment_digest"]

    rebuilt = types.SimpleNamespace(**{name: getattr(zlib, name) for name in dir(zlib) if not name.startswith("__")})
    rebuilt.compressobj = lambda level, method, wbits: zlib.compressobj(1, method, wbits)
    monkeypatch.setattr(environment, "zlib", rebuilt)
    different_zlib = environment.environment_record(cpuinfo_path=cpuinfo)
    assert different_zlib["zlib"]["raw_deflate_level6_digest"] != record["zlib"]["raw_deflate_level6_digest"]
    assert different_zlib["environment_digest"] != record["environment_digest"]
    assert forbidden_record_content(record) == [] and json.loads(json.dumps(record)) == record


def test_chain_preflight_reports_the_interpreter_environment_without_findings(tmp_path: Path, monkeypatch) -> None:
    cpuinfo = _cpuinfo(tmp_path / "cpuinfo", "sse2 avx2")
    measured = environment.environment_record(cpuinfo_path=cpuinfo)
    monkeypatch.setattr(environment, "CPUINFO_PATH", cpuinfo)
    monkeypatch.setattr(preflight.os, "geteuid", lambda: 0)
    monkeypatch.setattr(preflight, "CHAIN_UNITS", ())
    monkeypatch.setattr(preflight, "active_release", lambda: (None, "", []))
    monkeypatch.setattr(preflight, "_service_ids", lambda account: (1000, 1000))
    for name in (
        "intent_checks", "binding_checks", "handoff_checks", "project_spend_checks", "spend_refresh_sandbox_checks",
        "owner_scope_checks", "credential_file_checks", "provider_credit_check", "disk_admission_check",
        "unit_health_checks", "intake_check",
    ):
        monkeypatch.setattr(preflight, name, lambda *args, **kwargs: [])
    output = tmp_path / "preflight"

    def run(label: str) -> dict:
        latest = output / f"{label}.json"
        args = preflight.build_parser().parse_args(
            ["run", "--skip-sandbox", "--json-out", str(latest), "--history-out", str(output / "history.jsonl")]
        )
        assert preflight.run_chain(args) == 0
        return json.loads(latest.read_text(encoding="utf-8"))

    report = run("measured")
    assert report["interpreter_environment"] == measured
    assert (report["blocker_count"], report["warning_count"], report["host_findings"]) == (0, 0, [])
    assert forbidden_record_content(report["interpreter_environment"]) == []
    assert sorted(path.name for path in tmp_path.rglob("*") if path.is_file()) == [
        "cpuinfo", "history.jsonl", "measured.json",
    ]

    def broken(**_kwargs):
        raise RuntimeError("census unavailable")

    monkeypatch.setattr(environment, "environment_record", broken)
    failed = run("failed")
    assert failed["interpreter_environment"] == {
        "schema_version": "remote_cpu_environment.v1", "status": "unavailable",
        "error": "RuntimeError: census unavailable",
    }
    assert (failed["blocker_count"], failed["warning_count"], failed["host_findings"]) == (0, 0, [])
    assert [json.loads(line)["blocker_count"] for line in (output / "history.jsonl").read_text().splitlines()] == [0, 0]
