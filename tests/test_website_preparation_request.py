"""The intake request and the compose-back record a website preparation hands on, pinned byte for byte.

Both were assembled inside ``website_task_preparation.compile_website_scene_preparation`` until they moved, unchanged,
to ``website_preparation_request``. Three fixture jobs of ``tests.test_website_task_preparation`` (a captured cup, a
designed ball, a drawer cabinet) are compiled as the job runs them and their outputs are compared with digests
computed on main before the move: ``preparation.json``, ``removal_manifest.composed.json`` and the intake request.
A pin changes only with a change that means to change what a job hands on.
"""
from __future__ import annotations

import hashlib
import json
import types
import zipfile
from pathlib import Path

import pytest

from blueprint_pipeline import website_task_preparation as preparation
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.task_object_authoring_backend import ARTISAN_BUDGET_ENV, BACKEND_SETTING_ENV
from tests import test_website_task_preparation as fixtures

JOBS = {
    "captured_cup": lambda root: fixtures._arguments(root),
    "designed_ball": lambda root: fixtures.synthetic_arguments(fixtures._arguments(root)),
    "articulated_drawer": lambda root: fixtures._arguments(root, fixtures._assembly_inputs(root)),
}

#: Computed on main at 1e2c40212, before the move. ``bytes`` is the sha256 of the file as written (of the intake
#: request, ``json.dumps(request, sort_keys=True)``). ``rounded`` is the sha256 of that JSON with each float to 6
#: significant digits (below 1e-6, zero) and without its own seal (``digest``, checked against the exact content):
#: where it is pinned, the registration's last bits follow the machine's BLAS kernel (the same job on OpenBLAS's
#: SkylakeX and Haswell kernels differs by 1e-16 there), and nothing this record assembles.
PINNED = {
    "captured_cup": {
        "preparation.json": ("rounded", "sha256:09f0942ec0e08f2418c7f7c2fcb74e48afae71cfdd3955a4ccf0f619adf8bf8a"),
        "removal_manifest.composed.json": ("bytes",
                                           "sha256:06bb3ef87694161e04ac80fd4a52e94d2ebc323c96df7c1cd6e8d1970b5b0f6f"),
        "intake_request": ("bytes", "sha256:b4c1642c7091d9986a9e293e00fc5b821ca0379f766d1a6812fce4c937172954"),
    },
    "designed_ball": {
        "preparation.json": ("bytes", "sha256:12c64ba171b65d042a2b98a1252d8cb7eb78f422b7988a8ab88d8ff08dd27c62"),
        "removal_manifest.composed.json": ("bytes",
                                           "sha256:fab92730726113ecc91e3dea787974ce3df1422fc0f18153720b554db38cd9a3"),
        "intake_request": ("bytes", "sha256:e8d47fb74f51a670d2dd5f462965ada04f34254fafa42235e5d9c81324710d7d"),
    },
    "articulated_drawer": {
        "preparation.json": ("rounded", "sha256:8e40ced76971642bcd4d7150f529f77ba70e4931a6a659ffc35155708bd21c0e"),
        "removal_manifest.composed.json": ("rounded",
                                           "sha256:8b1f03db863a49825b67512c89700aa2929eaf619f22a301922625d2d2f16713"),
        "intake_request": ("rounded", "sha256:843f2288f9d8f8c120b5eda097b96cf8d8953d528948c66acb1ab0c6109a83bc"),
    },
}


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _rounded(value):
    if isinstance(value, float):
        return 0.0 if abs(value) < 1e-6 else float(f"{value:.6g}")
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    return value


def digests(job: str, work: Path, monkeypatch) -> dict[str, dict[str, str]]:
    """Compile ``job`` under ``work`` and return each output's ``bytes`` and ``rounded`` digest.

    Paths are relative to ``work`` (the job's paths are in its digests), the fixtures' ``np.savez`` members carry a
    fixed zip clock (their bytes are in the frames' digests) and the builder settings are the defaults."""
    monkeypatch.chdir(work)
    monkeypatch.setattr(zipfile, "time", types.SimpleNamespace(time=lambda: 0.0,
                                                               localtime=lambda *_: (1980, 1, 1, 0, 0, 0, 1, 1, 0)))
    for name in (BACKEND_SETTING_ENV, ARTISAN_BUDGET_ENV):
        monkeypatch.delenv(name, raising=False)
    root = Path("job")
    root.mkdir()
    value = preparation.compile_website_scene_preparation(**JOBS[job](root))
    assert value["status"] == "intake_ready", value["blockers"]
    written = {name: (root / "out" / name).read_bytes()
               for name in ("preparation.json", "removal_manifest.composed.json")}
    assert json.loads(written["preparation.json"])["intake_request"] == value["intake_request"]
    written["intake_request"] = json.dumps(value["intake_request"], sort_keys=True).encode()
    found = {}
    for name, data in written.items():
        content = json.loads(data)
        if "digest" in content:
            assert content["digest"] == canonical_digest(content, digest_field="digest")
        unsealed = {key: item for key, item in content.items() if key != "digest"}
        found[name] = {"bytes": _sha256(data),
                       "rounded": _sha256(json.dumps(_rounded(unsealed), sort_keys=True).encode())}
    return found


@pytest.mark.parametrize("job", sorted(JOBS))
def test_a_jobs_preparation_and_intake_request_are_byte_for_byte_as_before(tmp_path, monkeypatch, job):
    found = digests(job, tmp_path, monkeypatch)
    assert {name: (kind, found[name][kind]) for name, (kind, _) in PINNED[job].items()} == PINNED[job]
