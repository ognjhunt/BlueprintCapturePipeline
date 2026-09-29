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
GOLDEN_SIMD_INPUT_DIGEST = "sha256:ddbf088d1ba962388096bd276db57d40aff5f2b77fdfebfa49d3c4b592ea79d5"
FACTS_DISTRIBUTIONS = {
    "numpy", "usd-core", "msgpack", "jsonschema", "referencing", "rpds-py", "attrs", "webcolors", "pydantic",
    "pydantic-core", "pillow", "defusedxml", "packaging",
    # Re-derived from real fixture compiles in a fresh interpreter (plan 14 task 4.2,
    # test_parity_set_covers_every_distribution_a_fixture_compile_loads).
    "annotated-types", "idna", "jsonschema-specifications", "typing-extensions", "typing-inspection",
    "usd-convert-gsplat",
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


def _installed_version(name: str) -> str | None:
    # Hosted CI installs only the dev and policy_model_cpu extras, so some compile
    # distributions (webcolors arrives with build123d) may be absent there.
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _numpy_simd_view() -> dict:
    try:
        from numpy._core import _multiarray_umath as umath
    except ImportError:  # numpy 1.x
        from numpy.core import _multiarray_umath as umath
    return {
        "baseline": sorted(umath.__cpu_baseline__), "dispatch": sorted(umath.__cpu_dispatch__),
        "enabled": sorted(name for name, on in umath.__cpu_features__.items() if on),
    }


def test_environment_records_cpython_zlib_golden_deflate_simd_and_distributions(tmp_path: Path, monkeypatch) -> None:
    cpuinfo = _cpuinfo(tmp_path / "cpuinfo", "sse2 avx2 fma sse2")
    record = environment.environment_record(cpuinfo_path=cpuinfo)

    assert record["schema_version"] == "remote_cpu_environment.v1"
    assert record["python_version_info"] == list(sys.version_info)
    corpus = environment.golden_corpus()
    assert hashlib.sha256(corpus).hexdigest() == GOLDEN_CORPUS_SHA256
    assert record["golden_deflate"] == {
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
    assert _sha(data[start:start + member.compress_size]) == record["golden_deflate"]["raw_deflate_level6_digest"]

    # The golden SIMD digest is behaviour: NuRec's float32 exp and sigmoid and float64 norms.
    assert record["golden_simd"]["input_digest"] == GOLDEN_SIMD_INPUT_DIGEST
    assert environment.golden_simd() == record["golden_simd"]
    import numpy

    real_exp = numpy.exp
    monkeypatch.setattr(numpy, "exp", lambda values, *args, **kwargs: numpy.nextafter(
        real_exp(values, *args, **kwargs), numpy.float32(numpy.inf)).astype(values.dtype))
    perturbed = environment.golden_simd()
    monkeypatch.setattr(numpy, "exp", real_exp)
    assert perturbed["input_digest"] == GOLDEN_SIMD_INPUT_DIGEST
    assert perturbed["output_digest"] != record["golden_simd"]["output_digest"]

    simd = _numpy_simd_view()
    assert record["cpu_class"] == canonical_digest(
        {"machine": platform.machine(), "numpy_simd": simd, "flags": ["avx2", "fma", "sse2"]}
    )
    assert record["informational"] == {
        "python_version": sys.version, "implementation": platform.python_implementation(),
        "machine": platform.machine(), "libc": list(platform.libc_ver()),
        "zlib_runtime_version": zlib.ZLIB_RUNTIME_VERSION, "zlib_compile_version": zlib.ZLIB_VERSION,
        "numpy_simd": simd, "cpu_flags": ["avx2", "fma", "sse2"], "cpu_flags_source": str(cpuinfo),
        "executable": sys.executable,
    }

    assert set(environment.COMPILE_DISTRIBUTIONS) == FACTS_DISTRIBUTIONS
    assert [row["name"] for row in record["distributions"]] == sorted(FACTS_DISTRIBUTIONS)
    for row in record["distributions"]:
        assert row["version"] == _installed_version(row["name"]), row
    digested = {name: record[name] for name in environment.DIGESTED_FIELDS}
    assert set(digested) == {"python_version_info", "golden_deflate", "golden_simd", "distributions", "cpu_class"}
    assert record["environment_digest"] == canonical_digest(digested) == environment.environment_digest(record)

    other_cpu = environment.environment_record(cpuinfo_path=_cpuinfo(tmp_path / "other", "sse2"))
    unmeasured = environment.environment_record(cpuinfo_path=tmp_path / "absent")
    assert other_cpu["cpu_class"] != record["cpu_class"] and unmeasured["cpu_class"] is None
    assert len({record["environment_digest"], other_cpu["environment_digest"], unmeasured["environment_digest"]}) == 3
    assert (unmeasured["informational"]["cpu_flags"], unmeasured["informational"]["cpu_flags_source"]) == ([], None)
    missing = environment.environment_record(
        cpuinfo_path=cpuinfo, distributions=(*environment.COMPILE_DISTRIBUTIONS, "definitely-not-installed")
    )
    assert {"name": "definitely-not-installed", "version": None} in missing["distributions"]
    assert missing["environment_digest"] != record["environment_digest"]
    assert forbidden_record_content(record) == [] and json.loads(json.dumps(record)) == record


def test_environment_digest_covers_behaviour_not_build_strings(tmp_path: Path, monkeypatch) -> None:
    cpuinfo = _cpuinfo(tmp_path / "cpuinfo", "sse2 avx2")
    record = environment.environment_record(cpuinfo_path=cpuinfo)

    # Another build of the same interpreter, libc and zlib (zlib 1.2.12 and 1.3.1 deflate alike).
    monkeypatch.setattr(environment, "sys", types.SimpleNamespace(
        version="3.12.11 (main, Jan  1 2026, 00:00:00) [GCC 12.2.0]", version_info=sys.version_info,
        executable="/usr/local/bin/python3"))
    monkeypatch.setattr(environment, "platform", types.SimpleNamespace(
        python_implementation=platform.python_implementation, machine=platform.machine,
        libc_ver=lambda: ("glibc", "2.36")))
    other_zlib = types.SimpleNamespace(**{name: getattr(zlib, name) for name in dir(zlib) if not name.startswith("__")})
    other_zlib.ZLIB_RUNTIME_VERSION = other_zlib.ZLIB_VERSION = "1.2.12"
    monkeypatch.setattr(environment, "zlib", other_zlib)
    rebuilt = environment.environment_record(cpuinfo_path=cpuinfo)
    assert rebuilt["informational"] != record["informational"]
    assert rebuilt["informational"]["zlib_runtime_version"] == "1.2.12"
    assert rebuilt["environment_digest"] == record["environment_digest"]

    other_zlib.compressobj = lambda level, method, wbits: zlib.compressobj(1, method, wbits)
    deflates_differently = environment.environment_record(cpuinfo_path=cpuinfo)
    assert deflates_differently["golden_deflate"] != record["golden_deflate"]
    assert deflates_differently["environment_digest"] != record["environment_digest"]
    other_zlib.compressobj = zlib.compressobj

    monkeypatch.setattr(environment, "sys", types.SimpleNamespace(
        version=sys.version, version_info=(3, 12, 99, "final", 0), executable=sys.executable))
    assert environment.environment_record(cpuinfo_path=cpuinfo)["environment_digest"] != record["environment_digest"]


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
