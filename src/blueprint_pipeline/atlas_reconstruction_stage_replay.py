"""Replay one $0 reconstruction stage of a website scene on retained inputs and report how it ended.

The Atlas preview controller (``atlas_preview_replay``) reaches mesh registration only after paid
pose, generation and mesh stages, and levelling runs later still, in task preparation
(``website_atlas_scene_source``), on a ready job's pose, splat, mesh and registration records. A
record that stood for all of that would carry the posed RGBD frames, the generated views, the splat,
the mesh and every provider receipt that binds them. This runs the stage itself instead, on the files
it reads: its splat and mesh, and the cameras or splat manifest it is given.

``first_stop.kind`` is ``completed`` (the stage's own status in ``code``), ``estimate`` (it continued
on the labelled estimates named in ``code``, as ``atlas_scene_estimates`` names them), ``refusal``
(its typed code; ``deterministic`` holds the record of a ``DeterministicRefusal``), or ``error``.

Boundaries: the stage writes only under ``replay_root``; the inputs are digested before and after
(``evidence.unchanged``). It makes no paid call (``new_paid_calls`` is always 0: neither stage has
one). A socket connection or name lookup is refused and reported in ``network_attempts``.
"""

from __future__ import annotations

import contextlib
import json
import socket
import tempfile
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

SCHEMA = "atlas_reconstruction_stage_replay_report.v1"
STAGES = ("levelling", "mesh_registration")
STOP_KINDS = ("completed", "estimate", "refusal", "error")


class _NetworkRefused(BaseException):
    """A BaseException, so no ``except Exception`` in a stage absorbs it."""


@contextlib.contextmanager
def _offline(attempts: list[str]) -> Iterator[None]:
    connect, connect_ex, resolve = socket.socket.connect, socket.socket.connect_ex, socket.getaddrinfo

    def refuse(target: Any) -> None:
        attempts.append(str(target))
        raise _NetworkRefused(str(target))

    def guarded(original: Callable[..., Any]) -> Callable[..., Any]:
        def call(sock: socket.socket, address: Any) -> Any:
            if sock.family == getattr(socket, "AF_UNIX", None):
                return original(sock, address)
            refuse(address[0] if isinstance(address, tuple) else address)
        return call

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if host in (None, "", "localhost", "127.0.0.1", "::1"):
            return resolve(host, *args, **kwargs)
        refuse(host)

    from .atlas_preview_replay import _set  # restores what it replaced, and removes what it added

    with contextlib.ExitStack() as stack:
        _set(stack, socket.socket, "connect", guarded(connect))
        _set(stack, socket.socket, "connect_ex", guarded(connect_ex))
        _set(stack, socket, "getaddrinfo", getaddrinfo)
        yield


def _levelling(paths: Mapping[str, Path], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    import numpy as np

    from .atlas_world_levelling import level_atlas_world
    from .local_reconstruction_adapters import _sha256_file

    cameras = json.loads(paths["cameras"].read_text(encoding="utf-8"))
    value = level_atlas_world(splat_path=paths["splat"], splat_digest=_sha256_file(paths["splat"]),
                              mesh_path=paths["mesh"], mesh_digest=_sha256_file(paths["mesh"]),
                              meters_per_unit=float(cameras["meters_per_unit"]),
                              camera_centres=np.asarray(cameras["camera_centres"], dtype=np.float64),
                              output_root=output_root)
    return value, {"levelling": value, "registration": {}}


def _mesh_registration(paths: Mapping[str, Path], output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    from .atlas_mesh_registration import register_atlas_mesh

    value = register_atlas_mesh(glb=paths["mesh"], splat_root=paths["splat_root"], output_root=output_root)
    return {**value, "status": value.get("status", "registered")}, {"levelling": {}, "registration": value}


_RUNNERS: dict[str, Callable[[Mapping[str, Path], Path], tuple[dict[str, Any], dict[str, Any]]]] = {
    "levelling": _levelling, "mesh_registration": _mesh_registration}


def replay(*, stage: str, inputs: Mapping[str, str | Path], replay_root: str | Path | None = None) -> dict[str, Any]:
    """Run ``stage`` on ``inputs`` (name -> path), writing only under ``replay_root``."""
    from .atlas_preview_replay import _file_digest, _tree_digest
    from .atlas_scene_estimates import scene_estimates
    from .deterministic_refusal import find as find_deterministic

    if stage not in _RUNNERS:
        raise ValueError(f"unknown stage: {stage}")
    paths = {name: Path(value) for name, value in inputs.items()}

    def digest() -> dict[str, str | None]:
        return {name: (_tree_digest(path) if path.is_dir() else _file_digest(path) if path.is_file() else None)
                for name, path in sorted(paths.items())}
    root = Path(replay_root or tempfile.gettempdir())
    root.mkdir(parents=True, exist_ok=True)
    output_root = Path(tempfile.mkdtemp(prefix=f"{stage}-", dir=root))
    before, attempts = digest(), []
    first: dict[str, Any]
    try:
        with _offline(attempts):
            value, records = _RUNNERS[stage](paths, output_root / "outputs")
        codes = sorted({row["code"] for row in scene_estimates(**records)})
        status = str(value.get("status"))
        if status not in ("levelled", "registered"):
            first = {"kind": "refusal", "stage": stage, "code": value.get("reason")}
        elif codes:
            first = {"kind": "estimate", "stage": stage, "code": ",".join(codes)}
        else:
            first = {"kind": "completed", "stage": stage, "code": status}
    except _NetworkRefused as stop:
        first = {"kind": "error", "stage": stage, "code": f"unstubbed_network_call:{stop}"}
    except ValueError as error:
        found = find_deterministic(error)
        first = {"kind": "refusal", "stage": stage, "code": str(error).partition(":")[0],
                 **({"deterministic": found.record()} if found is not None else {})}
    except Exception as error:  # noqa: BLE001 - a defect in the stage is the finding
        first = {"kind": "error", "stage": stage, "code": type(error).__name__, "exception": str(error)[:500]}
    after = digest()
    report = {"schema_version": SCHEMA, "stage": stage, "inputs": {name: str(path) for name, path in paths.items()},
              "first_stop": first, "new_paid_calls": 0, "network_attempts": attempts,
              "evidence": {"digest_before": before, "digest_after": after, "unchanged": before == after},
              "report_path": str(output_root / "atlas_reconstruction_stage_replay_report.v1.json")}
    Path(report["report_path"]).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n",
                                           encoding="utf-8")
    return report


__all__ = ["SCHEMA", "STAGES", "STOP_KINDS", "replay"]
