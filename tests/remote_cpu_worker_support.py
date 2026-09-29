"""Plan 14 PR 3 test support: one Cloud Run execution's world, built from PR 1 and PR 2's real code.

``WorkerWorld`` seals a descriptor with the real contract, stages its inputs and release source in
the B2 fake, and mints the transport with the allocator's own ``mint_transport``, so the worker
reads exactly what the host would write.  ``RecordingHttp`` is the worker's presigned-URL
transfer layer against that fake: it sees only URLs, records the object each request named, and
never stores a URL itself.
"""

from __future__ import annotations

import functools
import hashlib
import io
import json
import os
import tarfile
import threading
import zipfile
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlsplit

from botocore.exceptions import ClientError

from blueprint_pipeline import remote_cpu_job_contract as contract
from blueprint_pipeline import remote_cpu_worker as worker
from blueprint_pipeline.decision_evidence_contracts import canonical_digest
from blueprint_pipeline.remote_cpu_transport import INPUT_FILENAME, INPUT_KIND, SOURCE_FILENAME, SOURCE_KIND, cas_key
from tests.remote_cpu_allocator_fakes import (
    B2_BUCKET,
    B2_REGION,
    IMAGE,
    JOB_SHORT,
    OBJECT_PREFIX,
    T0,
    TRANSPORT_BUCKET,
    environment,
    queue_name,
    remote_cpu_config,
)
from tests.remote_cpu_fakes import FakeArtifactStore, FakeClock, FakeTransportBucket

ROOT = Path(__file__).resolve().parents[1]
STAGES_MODULE = "remote_cpu_worker_stages"
COMMIT = "a" * 40
EXECUTION = f"{JOB_SHORT}-x7k2p"
ROWS = "/var/lib/blueprint/pipeline-control-plane/task-evaluation-episode-compilations/processing"
REFERENCES = "/var/lib/blueprint/task-evaluation-inputs/prepared-references"
COMPILED = "/var/lib/blueprint/task-evaluation-inputs/compiled-episodes"
PREFIX_URL = f"{FakeArtifactStore.endpoint}/{B2_BUCKET}/blueprint/arm-decision-proof-v1/configured-scenes/"
WORKER_RECORD = environment()
BUNDLE_MEMBERS = {"runtime/model.bin": b"weights" * 512, "runtime/config.json": b'{"runtime": 1}\n'}


def digest_of(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def release_archive(commit: str = COMMIT, files: dict[str, bytes] | None = None, *,
                    extra: Callable[[tarfile.TarFile], None] | None = None) -> bytes:
    """A recipe-v2 archive as ``git archive`` writes one: a pax commit comment, then the members."""

    files = files if files is not None else {
        "pyproject.toml": b"[project]\nname = 'blueprint-pipeline'\n",
        "src/blueprint_pipeline/__init__.py": b"",
        "docs/schemas/example.v1.schema.json": b'{"type": "object"}\n',
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT, pax_headers={"comment": commit}) as tar:
        for name in sorted({str(Path(name).parent) for name in files} - {"."}):
            parts = name.split("/")
            for depth in range(1, len(parts) + 1):
                directory = "/".join(parts[:depth])
                if directory not in tar.getnames():
                    info = tarfile.TarInfo(directory)
                    info.type, info.mode, info.mtime = tarfile.DIRTYPE, 0o775, 1_790_000_000
                    tar.addfile(info)
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size, info.mode, info.mtime = len(data), 0o664, 1_790_000_000
            tar.addfile(info, io.BytesIO(data))
        if extra is not None:
            extra(tar)
    return buffer.getvalue()


@functools.lru_cache(maxsize=8)
def worker_release_archive(commit: str = COMMIT, *, entries: tuple[str, ...] = ("blueprint_pipeline.remote_cpu_worker",),
                           omit: tuple[str, ...] = ()) -> bytes:
    """A recipe-v2 archive of this checkout's working tree for a real stage child: the import closure of
    ``entries``, every schema but ``omit``, ``pyproject.toml``, and the test stages as ``src/remote_cpu_worker_stages.py``."""

    from blueprint_pipeline.task_evaluation_production_chain_preflight import import_closure

    modules: dict[str, Path] = {}
    for entry in entries:
        modules.update(import_closure(ROOT / "src", entry))
    files = {str(path.relative_to(ROOT)): path.read_bytes() for path in modules.values()}
    files.update({str(path.relative_to(ROOT)): path.read_bytes() for path in (ROOT / "docs" / "schemas").rglob("*")
                  if path.is_file() and str(path.relative_to(ROOT)) not in omit})
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    files[f"src/{STAGES_MODULE}.py"] = (ROOT / "tests" / f"{STAGES_MODULE}.py").read_bytes()
    return release_archive(commit, files)


def report(result: dict[str, Any] | None = None, *, misses: tuple[str, ...] = (),
           failures: tuple[str, ...] = ()) -> dict[str, Any]:
    """What a stage child reports: its result, the release paths it missed, and its own failures."""

    return {"result": result, "release_path_misses": list(misses), "failures": list(failures)}


def runtime_bundle() -> bytes:
    """A runtime-source bundle whose manifest names its members, as the preparation adapter seals one."""

    entries = [{"relative_path": name, "sha256": digest_of(data), "size_bytes": len(data)}
               for name, data in sorted(BUNDLE_MEMBERS.items())]
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        for name, data in sorted(BUNDLE_MEMBERS.items()):
            bundle.writestr(zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)), data)
        bundle.writestr(zipfile.ZipInfo(worker.BUNDLE_MANIFEST, (1980, 1, 1, 0, 0, 0)),
                        json.dumps({"entries": entries}, sort_keys=True))
    return buffer.getvalue()


class RecordingHttp:
    """The worker's presigned transfers against the B2 fake; it records ``(method, key)``, never a URL."""

    def __init__(self, store: FakeArtifactStore, *, before: Callable[[str, str], None] | None = None) -> None:
        self.store, self.before = store, before
        self.requests: list[tuple[str, str]] = []
        self._lock = threading.Lock()

    def _send(self, method: str, url: str, body: bytes = b"") -> Any:
        key = unquote(urlsplit(url).path).split("/", 2)[2]
        with self._lock:
            self.requests.append((method, key))
            if self.before is not None:
                self.before(method, key)
            return self.store.request(method, url, body=body)

    def read(self, url: str, *, max_bytes: int, timeout: float) -> bytes | None:
        response = self._send("GET", url)
        if response.status == 404:
            return None
        if response.status != 200:
            raise worker.TransferError(f"http_{response.status}")
        if len(response.body) > max_bytes:
            raise worker.TransferError("response_too_large")
        return response.body

    def download(self, url: str, destination: Path, *, max_bytes: int, timeout: float) -> None:
        response = self._send("GET", url)
        if response.status != 200:
            raise worker.TransferError(f"http_{response.status}")
        if len(response.body) > max_bytes:
            raise worker.TransferError("response_too_large")
        with open(destination, "xb") as stream:
            stream.write(response.body)

    def upload(self, url: str, *, size: int, body: Any, content_type: str, timeout: float) -> None:
        if not isinstance(body, bytes):
            sink = io.BytesIO()
            body(sink)
            body = sink.getvalue()
        if len(body) != size:
            raise worker.TransferError("length_mismatch")
        response = self._send("PUT", url, body)
        if response.status != 200:
            raise worker.TransferError(f"http_{response.status}")


class WorkerWorld:
    """One execution's B2, transport bucket, descriptor and environment variables, and its local ``/``."""

    def __init__(self, tmp_path: Path, *, archive: bytes | None = None, commit: str = COMMIT,
                 limits: dict[str, Any] | None = None, closure: dict[str, Any] | None = None,
                 environment_digest: str | None = None, stage: str = "episode_compilation") -> None:
        self.tmp_path, self.clock = tmp_path, FakeClock(T0)
        self.store = FakeArtifactStore(clock=self.clock, bucket=B2_BUCKET)
        self.bucket = FakeTransportBucket(TRANSPORT_BUCKET, clock=self.clock)
        self.http = RecordingHttp(self.store)
        self.fs = tmp_path / "fs"
        self.logs: list[str] = []
        self.archive = release_archive(commit) if archive is None else archive
        self.put_cas(SOURCE_KIND, self.archive, SOURCE_FILENAME)
        name = queue_name("prep-1")
        self.inputs = {
            f"{ROWS}/{name}": (b'{"envelope": "prep-1"}\n', "queue_envelope", "queue_envelope"),
            f"{REFERENCES}/prep-1/runtime.zip": (runtime_bundle(), "materialized_reference",
                                                 "execution_adapter.runtime_source_bundle"),
            f"{REFERENCES}/prep-1/robot.json": (b'{"robot": "franka"}\n', "materialized_reference", "robot.configuration"),
        }
        config = remote_cpu_config()
        from blueprint_pipeline import remote_cpu_job_allocator as allocator

        stage_limits = allocator.stage_limits(config, "episode_compilation")
        stage_limits.update(limits or {})
        self.descriptor = contract.build_descriptor(
            config=config, stage=stage, mode="shadow", attempt=1,
            queue_row={"queue": contract.STAGES[stage]["queue"], "name": name,
                       "envelope_digest": "sha256:" + name.removesuffix(".json").rsplit("-", 1)[1]},
            code={"source_commit": commit, "image": IMAGE,
                  "environment_digest": environment_digest or WORKER_RECORD["environment_digest"],
                  "source_archive": {"digest": digest_of(self.archive), "size_bytes": len(self.archive),
                                     "uri": f"s3://{B2_BUCKET}/{cas_key(SOURCE_KIND, digest_of(self.archive), SOURCE_FILENAME)}"}},
            environment={},
            inputs=[{"role": role, "contract_path": contract_path, "digest": digest_of(data), "size_bytes": len(data),
                     "mode": "0440", "materialize_at": path,
                     "uri": f"s3://{B2_BUCKET}/{self.put_cas(INPUT_KIND, data, INPUT_FILENAME)}"}
                    for path, (data, role, contract_path) in self.inputs.items()],
            outputs={"output_root": f"{COMPILED}/prep-1", "declared_scratch": [f"{COMPILED}/content-addressed/"],
                     "object_prefix": OBJECT_PREFIX},
            limits=stage_limits, closure=closure or {"class": "not_applicable", "source_appearance_digest": None},
            spend={"worst_case_usd": allocator.worst_case_usd(limits=stage_limits, rate_table=config["rate_table"]),
                   "rate_table_digest": canonical_digest(config["rate_table"])},
            nonce="1" * 32,
        )
        self.environ = self.mint()

    def put_cas(self, kind: str, data: bytes, filename: str) -> str:
        key = cas_key(kind, digest_of(data), filename)
        self.store.put_object(Bucket=B2_BUCKET, Key=key, Body=data, Metadata={"sha256": digest_of(data)[7:]})
        return key

    def mint(self, *, nonce: str = "a" * 32, mutate: Callable[[dict[str, Any]], None] | None = None) -> dict[str, str]:
        """Mint the transport as the allocator does and return the execution's environment variables."""

        from blueprint_pipeline import remote_cpu_job_allocator as allocator

        binding = {"attempt_id": self.descriptor["attempt_id"]}
        admission, grant = allocator.admit_remote_cpu_job(blockers=[], binding=binding, execute=True)
        name = f"gs://{TRANSPORT_BUCKET}/transport/{self.descriptor['job_id']}/{self.descriptor['attempt_id']}-{nonce}.json"
        minted = allocator.mint_transport(
            grant=grant, binding_digest=admission["allocation_binding_digest"], descriptor=self.descriptor,
            bucket=self.bucket, object_store=(self.store, B2_BUCKET, B2_REGION), clock=self.clock, object_uri=name)
        generation = minted["transport_generation"]
        if mutate is not None:
            object_name = name.split("/", 3)[3]
            transport = json.loads(self.bucket.get(object_name, generation=generation))
            mutate(transport)
            self.bucket.delete(object_name, generation=generation)
            generation = self.bucket.create(object_name, json.dumps(transport).encode(), if_generation_match=0)
        return {
            "CLOUD_RUN_JOB": JOB_SHORT, "CLOUD_RUN_EXECUTION": EXECUTION, "CLOUD_RUN_TASK_INDEX": "0",
            "CLOUD_RUN_TASK_ATTEMPT": "0", "CLOUD_RUN_TASK_COUNT": "1", "PATH": os.environ.get("PATH", ""),
            "BLUEPRINT_REMOTE_CPU_STAGE": "episode-compilation", "BLUEPRINT_REMOTE_CPU_OBJECT_PREFIX": PREFIX_URL,
            "BLUEPRINT_REMOTE_CPU_ATTEMPT_ID": self.descriptor["attempt_id"],
            "BLUEPRINT_REMOTE_CPU_DESCRIPTOR_SHA256": self.descriptor["descriptor_digest"],
            "BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT": name, "BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION": str(generation),
        }

    def transport(self) -> dict[str, Any]:
        name = self.environ["BLUEPRINT_REMOTE_CPU_TRANSPORT_OBJECT"].split("/", 3)[3]
        return json.loads(self.bucket.get(name, generation=int(self.environ["BLUEPRINT_REMOTE_CPU_TRANSPORT_GENERATION"])))

    def runtime(self, **changes: Any) -> Any:
        values = {"environ": self.environ, "http": self.http, "reader": self.bucket.reader(), "filesystem_root": self.fs,
                  "clock": self.clock, "measure": lambda: WORKER_RECORD, "log": self.logs.append,
                  "launch": lambda handoff: 0, **changes}
        return worker.WorkerRuntime(**values)

    def run(self, run_stage: Callable[..., dict[str, Any]] | None = None, *, stage: str = "compiled",
            **changes: Any) -> int:
        """Bootstrap, then execute in this process; ``run_stage`` stands in for the stage child when given."""

        execute_runtime = self.runtime(run_stage=run_stage, handlers={
            self.descriptor["stage"]: f"{STAGES_MODULE}:{stage}"}, **changes)
        return worker.bootstrap(["bootstrap"], self.runtime(
            launch=lambda handoff: worker.execute_attempt(handoff, execute_runtime), **changes))

    def key(self, name: str) -> str:
        return self.descriptor["outputs"]["staging_prefix"].removeprefix(f"s3://{B2_BUCKET}/") + name

    def versions(self, name: str) -> list[bytes]:
        return [version.data for version in self.store.buckets[B2_BUCKET].get(self.key(name), [])
                if version.data is not None]

    def staged(self, name: str) -> bytes | None:
        try:
            return self.store.get_object(Bucket=B2_BUCKET, Key=self.key(name))["Body"].read()
        except ClientError:
            return None

    def receipt(self) -> dict[str, Any] | None:
        data = self.staged("receipt.json")
        return None if data is None else json.loads(data)

    def heartbeats(self) -> list[dict[str, Any]]:
        return [json.loads(data) for data in self.versions("heartbeat.json")]

    def local(self, path: str) -> Path:
        return self.fs / path.lstrip("/")
