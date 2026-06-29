from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
pytest.importorskip("PIL")
from PIL import Image

from blueprint_pipeline.synthesis import cosmos_inference


def test_try_load_cosmos_official_repo_exposes_subprocess_env(monkeypatch, tmp_path: Path) -> None:
    repo_root = tmp_path / "cosmos-predict2.5"
    inference_entrypoint = repo_root / "examples" / "inference.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    inference_entrypoint.parent.mkdir(parents=True)
    python_bin.parent.mkdir(parents=True)
    inference_entrypoint.write_text("print('ok')\n", encoding="utf-8")
    python_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("COSMOS_DISABLE_GUARDRAILS", "true")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_DISABLE_GUARDRAILS", True)

    model = cosmos_inference._try_load_cosmos_official_repo("nvidia/Cosmos-Predict2.5-2B")

    assert model is not None
    assert model["backend"] == "official_repo_script"
    assert model["model_variant"] == "2B/post-trained"
    env = model["subprocess_env"]
    assert str((repo_root / ".venv" / "bin").resolve()) == env["PATH"].split(":")[0]
    assert str((Path.home() / ".local" / "bin").resolve()) in env["PATH"].split(":")
    assert "VIRTUAL_ENV" not in env
    assert "UV_PYTHON" not in env


def test_prepend_search_paths_deduplicates_entries() -> None:
    merged = cosmos_inference._prepend_search_paths(
        ["/custom/bin", "/usr/bin", "/custom/bin"],
        "/usr/bin:/bin:/usr/bin",
    )

    assert merged == "/custom/bin:/usr/bin:/bin"


def test_try_load_cosmos_official_repo_worker_sets_resident_worker_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "cosmos-predict2.5"
    inference_entrypoint = repo_root / "examples" / "inference.py"
    python_bin = repo_root / ".venv" / "bin" / "python"
    inference_entrypoint.parent.mkdir(parents=True)
    python_bin.parent.mkdir(parents=True)
    inference_entrypoint.write_text("print('ok')\n", encoding="utf-8")
    python_bin.write_text("", encoding="utf-8")

    monkeypatch.setenv("COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    monkeypatch.delenv("COSMOS_DISABLE_PERSISTENT_WORKER", raising=False)
    monkeypatch.setenv("PYTHONPATH", "/tmp/existing")
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))

    client = cosmos_inference._try_load_cosmos_official_repo_worker(
        "nvidia/Cosmos-Predict2.5-2B"
    )

    assert client is not None
    assert client.describe()["backend"] == "persistent_worker"
    env = client.worker_env
    assert env["COSMOS_DISABLE_PERSISTENT_WORKER"] == "1"
    assert env["COSMOS_SKIP_OFFICIAL_REPO_SCRIPT"] == "1"
    assert env["COSMOS_ALLOW_COLD_SUBPROCESS_FALLBACK"] == "0"
    assert str((repo_root / ".venv" / "bin").resolve()) == env["PATH"].split(":")[0]
    assert str(Path(cosmos_inference.__file__).resolve().parents[2]) == env["PYTHONPATH"].split(":")[0]


def test_try_load_cosmos_official_repo_worker_honors_python_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "cosmos-predict2.5"
    inference_entrypoint = repo_root / "examples" / "inference.py"
    repo_python = repo_root / ".venv" / "bin" / "python"
    override_python = tmp_path / "custom-python"
    inference_entrypoint.parent.mkdir(parents=True)
    repo_python.parent.mkdir(parents=True)
    inference_entrypoint.write_text("print('ok')\n", encoding="utf-8")
    repo_python.write_text("", encoding="utf-8")
    override_python.write_text("", encoding="utf-8")

    monkeypatch.setenv("COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    monkeypatch.setenv("COSMOS_WORKER_PYTHON_BIN", str(override_python))
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))

    client = cosmos_inference._try_load_cosmos_official_repo_worker(
        "nvidia/Cosmos-Predict2.5-2B"
    )

    assert client is not None
    assert client.python_bin == override_python


def test_cosmos_generation_model_cache_and_invocation_paths(monkeypatch, tmp_path: Path) -> None:
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    output = tmp_path / "view.jpg"
    assert cosmos_inference.generate_view(
        splatted_image=image,
        coverage_mask=np.ones((4, 5), dtype=bool),
        output_path=output,
        mode="splat_only",
    ) == output
    assert output.is_file()
    with pytest.raises(ValueError, match="Unknown generation mode"):
        cosmos_inference.generate_view(
            splatted_image=image,
            coverage_mask=np.ones((4, 5), dtype=bool),
            output_path=tmp_path / "bad.jpg",
            mode="bad",
        )

    class OfficialModel:
        generate = True

        def image_to_world(self, **kwargs):
            assert kwargs["num_frames"] == 2
            return [np.ones((2, 2, 3), dtype=np.uint8)]

    class CallableModel:
        def __call__(self, **_kwargs):
            return SimpleNamespace(frames=[[np.zeros((2, 2, 3), dtype=np.uint8)]])

    assert len(cosmos_inference._invoke_cosmos(
        model=OfficialModel(),
        conditioning_image=Image.fromarray(image),
        num_frames=2,
        width=8,
        height=8,
        guidance_scale=1.0,
        num_steps=1,
    )) == 1
    assert len(cosmos_inference._invoke_cosmos(
        model=CallableModel(),
        conditioning_image=Image.fromarray(image),
        num_frames=2,
        width=8,
        height=8,
        guidance_scale=1.0,
        num_steps=1,
    )) == 1
    assert cosmos_inference._extract_frames(SimpleNamespace(images=[1, 2])) == [1, 2]
    with pytest.raises(RuntimeError, match="Cannot extract frames"):
        cosmos_inference._extract_frames(object())
    with pytest.raises(RuntimeError, match="Unrecognised Cosmos"):
        cosmos_inference._invoke_cosmos(
            model=object(),
            conditioning_image=Image.fromarray(image),
            num_frames=2,
            width=8,
            height=8,
            guidance_scale=1.0,
            num_steps=1,
        )

    class FakeWorker:
        def __init__(self) -> None:
            self.generated = False

        def generate_image_to_world(self, **kwargs):
            self.generated = True
            Path(kwargs["output_path"]).write_bytes(b"video")
            return {"ok": True}

    worker = FakeWorker()
    monkeypatch.setattr(cosmos_inference, "PersistentCosmosWorkerClient", FakeWorker)
    assert cosmos_inference._cosmos_image_to_world(
        conditioning_image=image,
        output_path=tmp_path / "worker.jpg",
        cosmos_model=worker,
        num_frames=1,
        width=4,
        height=4,
        guidance_scale=1.0,
        num_steps=1,
    ).name == "worker.jpg"
    assert worker.generated is True

    original_save_video = cosmos_inference._save_video
    monkeypatch.setattr(cosmos_inference, "_invoke_cosmos", lambda **_kwargs: [np.ones((2, 2, 3), dtype=np.uint8)])
    monkeypatch.setattr(cosmos_inference, "_save_video", lambda frames, path, fps=28: path.write_bytes(b"mp4"))
    cosmos_out = cosmos_inference._cosmos_image_to_world(
        conditioning_image=image,
        output_path=tmp_path / "cosmos.jpg",
        cosmos_model=CallableModel(),
        num_frames=1,
        width=4,
        height=4,
        guidance_scale=1.0,
        num_steps=1,
    )
    assert cosmos_out.is_file()
    assert cosmos_out.with_suffix(".mp4").read_bytes() == b"mp4"

    imageio_mod = ModuleType("imageio")

    class Writer:
        def __init__(self) -> None:
            self.frames = []

        def append_data(self, frame) -> None:
            self.frames.append(frame)

        def close(self) -> None:
            return None

    writer = Writer()
    imageio_mod.get_writer = lambda *_args, **_kwargs: writer
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setattr(cosmos_inference, "_save_video", original_save_video)
    cosmos_inference._save_video([np.zeros((1, 1, 3), dtype=np.uint8), Image.fromarray(np.zeros((1, 1, 3), dtype=np.uint8))], tmp_path / "saved.mp4")
    assert len(writer.frames) == 2

    with cosmos_inference._LOADED_MODELS_LOCK:
        cosmos_inference._LOADED_MODELS.clear()
    monkeypatch.setattr(cosmos_inference, "_try_load_cosmos_official", lambda mid: {"backend": "official", "model_id": mid})
    model = cosmos_inference.load_cosmos_model("model-a")
    assert model["backend"] == "official"
    assert cosmos_inference.load_cosmos_model("model-a") is model
    with cosmos_inference._LOADED_MODELS_LOCK:
        cosmos_inference._LOADED_MODELS.clear()
    monkeypatch.setattr(cosmos_inference, "_try_load_cosmos_official", lambda mid: None)
    monkeypatch.setattr(cosmos_inference, "_try_load_cosmos_official_repo_direct", lambda mid: None)
    monkeypatch.setattr(cosmos_inference, "_try_load_cosmos_diffusers", lambda mid: None)
    monkeypatch.setattr(cosmos_inference, "_try_load_cosmos_official_repo_worker", lambda mid: None)
    monkeypatch.setattr(cosmos_inference, "_try_load_cosmos_official_repo", lambda mid: None)
    with pytest.raises(ImportError):
        cosmos_inference.load_cosmos_model("missing")
    monkeypatch.setattr(cosmos_inference, "load_cosmos_model", lambda model_id=None: {"backend": "mapping"})
    prewarm = cosmos_inference.prewarm_cosmos_model("model-b")
    assert prewarm["backend"] == "mapping"
    assert prewarm["model_id"] == "model-b"
    assert cosmos_inference.describe_cosmos_model(object())["backend"] == "object"


def test_persistent_worker_client_protocol_and_close(monkeypatch, tmp_path: Path) -> None:
    class FakeStdin:
        def __init__(self) -> None:
            self.writes: list[str] = []
            self.closed = False

        def write(self, value: str) -> None:
            self.writes.append(value)

        def flush(self) -> None:
            return None

        def close(self) -> None:
            self.closed = True

    class FakeStdout:
        def __init__(self, lines: list[str]) -> None:
            self.lines = lines
            self.closed = False

        def __iter__(self):
            return iter(self.lines)

        def close(self) -> None:
            self.closed = True

    class FakeProcess:
        def __init__(self, lines: list[str]) -> None:
            self.stdin = FakeStdin()
            self.stdout = FakeStdout(lines)
            self.returncode = 9
            self.terminated = False
            self.killed = False

        def poll(self):
            return None if not self.terminated and not self.killed else self.returncode

        def terminate(self):
            self.terminated = True

        def wait(self, timeout=None):
            raise subprocess.TimeoutExpired("worker", timeout)

        def kill(self):
            self.killed = True

    processes: list[FakeProcess] = []

    def fake_popen(*_args, **_kwargs):
        proc = FakeProcess([
            "\n",
            "not-json\n",
            json.dumps({"type": "ready", "backend": "fake-worker"}) + "\n",
            json.dumps({"type": "pong"}) + "\n",
            json.dumps({"type": "result", "request_id": "rid", "ok": True}) + "\n",
        ])
        processes.append(proc)
        return proc

    monkeypatch.setattr(cosmos_inference.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(cosmos_inference.uuid, "uuid4", lambda: SimpleNamespace(hex="rid"))
    client = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={"COSMOS_WORKER_LOG_PATH": str(tmp_path / "worker.log")},
        model_id="model",
        model_variant="2B/post-trained",
    )
    assert client.prewarm()["type"] == "pong"
    result = client.generate_image_to_world(
        conditioning_image=np.zeros((2, 2, 3), dtype=np.uint8),
        output_path=tmp_path / "worker.jpg",
        num_frames=1,
        width=2,
        height=2,
        guidance_scale=1.0,
        num_steps=1,
    )
    assert result["ok"] is True
    assert client.describe()["worker_backend"] == "fake-worker"
    client.close()
    assert processes[0].killed is True

    stopped = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    stopped._process = SimpleNamespace(stdin=None, poll=lambda: 1, returncode=1)
    with pytest.raises(RuntimeError, match="persistent_worker_not_running"):
        stopped._send_message({"type": "ping"})
    stopped._stdout_queue.put({"type": "error", "stage": "load", "error": "bad"})
    with pytest.raises(RuntimeError, match="load:bad"):
        stopped._await_message(message_type="ready", timeout_s=1)
    stopped._stdout_queue.put({"type": "protocol_error", "raw": "oops"})
    with pytest.raises(RuntimeError, match="persistent_worker_protocol_error"):
        stopped._await_message(message_type="ready", timeout_s=1)


def test_cosmos_backend_loader_and_official_repo_invocation_edges(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("COSMOS_ENABLE_PERSISTENT_WORKER", "false")
    assert cosmos_inference._persistent_worker_enabled() is False
    monkeypatch.setenv("COSMOS_SKIP_OFFICIAL_REPO_SCRIPT", "true")
    assert cosmos_inference._skip_official_repo_script() is True
    monkeypatch.setenv("COSMOS_ALLOW_COLD_SUBPROCESS_FALLBACK", "false")
    assert cosmos_inference._cold_subprocess_fallback_enabled() is False

    official_mod = ModuleType("cosmos_predict2_5")

    class Official:
        @classmethod
        def from_pretrained(cls, model_id):
            return cls()

        def eval(self):
            self.eval_called = True

    official_mod.CosmosPredict25 = Official
    monkeypatch.setitem(sys.modules, "cosmos_predict2_5", official_mod)
    assert isinstance(cosmos_inference._try_load_cosmos_official("model"), Official)
    monkeypatch.delitem(sys.modules, "cosmos_predict2_5", raising=False)
    assert cosmos_inference._try_load_cosmos_official("model") is None

    torch_mod = ModuleType("torch")
    torch_mod.bfloat16 = "bf16"
    torch_mod.cuda = SimpleNamespace(is_available=lambda: True)
    diffusers_mod = ModuleType("diffusers")

    class Pipe:
        moved_to = None

        @classmethod
        def from_pretrained(cls, **_kwargs):
            return cls()

        def to(self, device):
            self.moved_to = device
            return self

    diffusers_mod.DiffusionPipeline = SimpleNamespace(from_pretrained=lambda *args, **kwargs: Pipe())
    monkeypatch.setitem(sys.modules, "torch", torch_mod)
    monkeypatch.setitem(sys.modules, "diffusers", diffusers_mod)
    assert cosmos_inference._try_load_cosmos_diffusers("model").moved_to == "cuda"
    monkeypatch.delitem(sys.modules, "diffusers", raising=False)
    assert cosmos_inference._try_load_cosmos_diffusers("model") is None

    assert cosmos_inference._official_repo_model_variant("bad", "post-trained") is None
    assert cosmos_inference._official_repo_model_variant("nvidia/Cosmos-Predict2.5-14B", "pre-trained") == "14B/pre-trained"
    assert cosmos_inference._normalized_subprocess_env({"A": "b", 1: "skip", "C": 3}) == {"A": "b"}
    monkeypatch.setenv("HF_TOKEN", "hf")
    monkeypatch.setenv("UNSAFE_SECRET", "no")
    assert cosmos_inference._select_official_repo_env_vars()["HF_TOKEN"] == "hf"

    repo_root = tmp_path / "repo"
    python_bin = repo_root / ".venv" / "bin" / "python"
    inference_entrypoint = repo_root / "examples" / "inference.py"
    inference_entrypoint.parent.mkdir(parents=True)
    python_bin.parent.mkdir(parents=True)
    inference_entrypoint.write_text("ok", encoding="utf-8")
    python_bin.write_text("python", encoding="utf-8")
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", "")
    assert cosmos_inference._try_load_cosmos_official_repo_direct("nvidia/Cosmos-Predict2.5-2B") is None
    assert cosmos_inference._try_load_cosmos_official_repo_worker("nvidia/Cosmos-Predict2.5-2B") is None
    assert cosmos_inference._try_load_cosmos_official_repo("nvidia/Cosmos-Predict2.5-2B") is None
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    monkeypatch.setattr(cosmos_inference, "_persistent_worker_enabled", lambda: False)
    assert cosmos_inference._try_load_cosmos_official_repo_worker("nvidia/Cosmos-Predict2.5-2B") is None
    monkeypatch.setattr(cosmos_inference, "_skip_official_repo_script", lambda: True)
    assert cosmos_inference._try_load_cosmos_official_repo("nvidia/Cosmos-Predict2.5-2B") is None

    imageio_v3 = ModuleType("imageio.v3")
    imageio_v3.imiter = lambda path: [np.zeros((2, 2, 3), dtype=np.uint8)]
    imageio_pkg = ModuleType("imageio")
    imageio_pkg.v3 = imageio_v3
    monkeypatch.setitem(sys.modules, "imageio", imageio_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3)

    uuid_values = iter(["abcdef123456", "fedcba654321", "123456abcdef", "654321fedcba", "aaaaaaaabbbb"])
    monkeypatch.setattr(cosmos_inference.uuid, "uuid4", lambda: SimpleNamespace(hex=next(uuid_values)))

    def run_script(command, **kwargs):
        output_dir = Path(command[command.index("-o") + 1])
        sample_name = output_dir.name
        (output_dir / f"{sample_name}.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cosmos_inference.subprocess, "run", run_script)
    frames = cosmos_inference._invoke_cosmos_official_repo_script(
        model={
            "backend": "official_repo_script",
            "repo_root": str(repo_root),
            "python_bin": str(python_bin),
            "model_variant": "2B/post-trained",
            "disable_guardrails": True,
            "subprocess_env": {"PATH": "x"},
        },
        conditioning_image=np.zeros((2, 2, 3), dtype=np.uint8),
    )
    assert len(frames) == 1
    monkeypatch.setattr(cosmos_inference.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 2))
    with pytest.raises(RuntimeError, match="official_repo_inference_failed"):
        cosmos_inference._invoke_cosmos_official_repo_script(
            model={
                "backend": "official_repo_script",
                "repo_root": str(repo_root),
                "python_bin": str(python_bin),
                "model_variant": "2B/post-trained",
            },
            conditioning_image=np.zeros((2, 2, 3), dtype=np.uint8),
        )

    config_mod = ModuleType("cosmos_predict2.config")

    class InferenceArguments:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    config_mod.InferenceArguments = InferenceArguments
    monkeypatch.setitem(sys.modules, "cosmos_predict2.config", config_mod)

    class DirectInference:
        def __init__(self, output: Path | None):
            self.output = output

        def generate(self, _samples, output_dir):
            return [self.output] if self.output is not None else []

    direct_video = tmp_path / "direct.mp4"
    direct_video.write_bytes(b"mp4")
    assert len(cosmos_inference._invoke_cosmos_official_repo_direct(
        model={"output_root": str(tmp_path / "direct-out"), "inference": DirectInference(direct_video)},
        conditioning_image=np.zeros((2, 2, 3), dtype=np.uint8),
        num_frames=3,
        guidance_scale=9.0,
        num_steps=2,
    )) == 1
    with pytest.raises(RuntimeError, match="official_repo_direct_inference_missing"):
        cosmos_inference._invoke_cosmos_official_repo_direct(
            model={"output_root": str(tmp_path / "direct-missing")},
            conditioning_image=np.zeros((2, 2, 3), dtype=np.uint8),
            num_frames=1,
            guidance_scale=1,
            num_steps=1,
        )
    with pytest.raises(RuntimeError, match="official_repo_direct_output_missing"):
        cosmos_inference._invoke_cosmos_official_repo_direct(
            model={"output_root": str(tmp_path / "direct-empty"), "inference": DirectInference(None)},
            conditioning_image=np.zeros((2, 2, 3), dtype=np.uint8),
            num_frames=1,
            guidance_scale=1,
            num_steps=1,
        )


def test_cosmos_remaining_branch_edges(monkeypatch, tmp_path: Path) -> None:
    image = np.zeros((2, 3, 3), dtype=np.uint8)

    class RaisingCloser:
        def close(self) -> None:
            raise RuntimeError("close failed")

    closed_client = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    closed_client._process = SimpleNamespace(
        stdin=RaisingCloser(),
        stdout=RaisingCloser(),
        poll=lambda: 1,
        returncode=1,
    )
    closed_client._log_handle = RaisingCloser()
    closed_client.close()

    failing_generation = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    failing_generation._ensure_started = lambda: None
    failing_generation._send_message = lambda _payload: None
    failing_generation._await_message = lambda **_kwargs: {"ok": False, "error": "bad-generation"}
    with pytest.raises(RuntimeError, match="bad-generation"):
        failing_generation.generate_image_to_world(
            conditioning_image=image,
            output_path=tmp_path / "bad-worker.jpg",
            num_frames=1,
            width=2,
            height=2,
            guidance_scale=1,
            num_steps=1,
        )

    class StartedThenClosedProcess:
        stdin = None
        stdout = None
        returncode = 0

        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout=None) -> None:
            return None

    class FakeThread:
        def __init__(self, **_kwargs) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

    startup_failure = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={"COSMOS_WORKER_LOG_PATH": str(tmp_path / "startup.log")},
        model_id="model",
        model_variant="2B/post-trained",
    )
    monkeypatch.setattr(cosmos_inference.subprocess, "Popen", lambda *_args, **_kwargs: StartedThenClosedProcess())
    monkeypatch.setattr(cosmos_inference.threading, "Thread", FakeThread)
    startup_failure._await_message = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("startup bad"))
    with pytest.raises(RuntimeError, match="startup bad"):
        startup_failure._ensure_started()
    assert startup_failure._process is None

    no_stdout = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    no_stdout._drain_stdout()

    timeout_client = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    with pytest.raises(RuntimeError, match="persistent_worker_timeout:ready"):
        timeout_client._await_message(message_type="ready", timeout_s=0)

    exited_client = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    exited_client._process = SimpleNamespace(poll=lambda: 4, returncode=4)
    with pytest.raises(RuntimeError, match="persistent_worker_exited:4"):
        exited_client._await_message(message_type="ready", timeout_s=1)

    queue_empty_client = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    poll_values = iter([None, 6])
    queue_empty_client._process = SimpleNamespace(poll=lambda: next(poll_values), returncode=6)

    class EmptyQueue:
        def empty(self) -> bool:
            return False

        def get(self, timeout=None):
            raise cosmos_inference.queue.Empty

    queue_empty_client._stdout_queue = EmptyQueue()
    with pytest.raises(RuntimeError, match="persistent_worker_exited:6"):
        queue_empty_client._await_message(message_type="ready", timeout_s=1)

    skipped_message_client = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    skipped_message_client._stdout_queue.put({"type": "log"})
    skipped_message_client._stdout_queue.put({"type": "result", "request_id": "other"})
    skipped_message_client._stdout_queue.put({"type": "result", "request_id": "wanted", "ok": True})
    assert skipped_message_client._await_message(message_type="result", request_id="wanted", timeout_s=1)["ok"] is True

    continuing_client = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    continuing_client._process = SimpleNamespace(poll=lambda: None, returncode=None)

    class EmptyThenReadyQueue:
        def __init__(self) -> None:
            self.calls = 0

        def empty(self) -> bool:
            return False

        def get(self, timeout=None):
            self.calls += 1
            if self.calls == 1:
                raise cosmos_inference.queue.Empty
            return {"type": "ready"}

    continuing_client._stdout_queue = EmptyThenReadyQueue()
    assert continuing_client._await_message(message_type="ready", timeout_s=1)["type"] == "ready"

    closable_worker = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    close_calls: list[str] = []
    closable_worker.close = lambda: close_calls.append("closed")
    with cosmos_inference._LOADED_MODELS_LOCK:
        cosmos_inference._LOADED_MODELS.clear()
        cosmos_inference._LOADED_MODELS["worker"] = closable_worker
        cosmos_inference._LOADED_MODELS["plain"] = object()
    cosmos_inference._close_loaded_models()
    assert close_calls == ["closed"]

    real_cosmos_image_to_world = cosmos_inference._cosmos_image_to_world
    monkeypatch.setattr(cosmos_inference, "_cosmos_image_to_world", lambda **kwargs: Path(kwargs["output_path"]))
    assert cosmos_inference.generate_view(
        splatted_image=image,
        coverage_mask=np.ones((2, 3), dtype=bool),
        output_path=tmp_path / "generated.jpg",
        mode="cosmos_i2w",
    ).name == "generated.jpg"
    monkeypatch.setattr(cosmos_inference, "_cosmos_image_to_world", real_cosmos_image_to_world)

    worker_for_prewarm = cosmos_inference.PersistentCosmosWorkerClient(
        repo_root=tmp_path,
        python_bin=tmp_path / "python",
        worker_env={},
        model_id="model",
        model_variant="2B/post-trained",
    )
    worker_for_prewarm.describe = lambda: {"backend": "persistent_worker", "ready": True}
    worker_for_prewarm.prewarm = lambda: {"type": "pong"}
    monkeypatch.setattr(cosmos_inference, "load_cosmos_model", lambda model_id=None: worker_for_prewarm)
    prewarm_payload = cosmos_inference.prewarm_cosmos_model("model")
    assert prewarm_payload["type"] == "pong"
    assert cosmos_inference.describe_cosmos_model(worker_for_prewarm)["ready"] is True

    real_invoke_cosmos = cosmos_inference._invoke_cosmos
    monkeypatch.setattr(cosmos_inference, "_invoke_cosmos", lambda **_kwargs: [Image.fromarray(image)])
    saved_videos: list[Path] = []
    real_save_video = cosmos_inference._save_video
    monkeypatch.setattr(cosmos_inference, "_save_video", lambda _frames, path, fps=28: saved_videos.append(path))
    pil_output = cosmos_inference._cosmos_image_to_world(
        conditioning_image=image,
        output_path=tmp_path / "pil-frame.jpg",
        cosmos_model=object(),
        num_frames=1,
        width=3,
        height=2,
        guidance_scale=1,
        num_steps=1,
    )
    assert pil_output.is_file()
    assert saved_videos == [tmp_path / "pil-frame.mp4"]

    monkeypatch.setattr(cosmos_inference, "_invoke_cosmos", real_invoke_cosmos)
    real_official_repo_direct = cosmos_inference._invoke_cosmos_official_repo_direct
    real_official_repo_script = cosmos_inference._invoke_cosmos_official_repo_script
    monkeypatch.setattr(cosmos_inference, "_invoke_cosmos_official_repo_direct", lambda **_kwargs: ["direct"])
    monkeypatch.setattr(cosmos_inference, "_invoke_cosmos_official_repo_script", lambda **_kwargs: ["script"])
    assert cosmos_inference._invoke_cosmos(
        model={"backend": "official_repo_direct"},
        conditioning_image=Image.fromarray(image),
        num_frames=1,
        width=3,
        height=2,
        guidance_scale=1,
        num_steps=1,
    ) == ["direct"]
    assert cosmos_inference._invoke_cosmos(
        model={"backend": "official_repo_script"},
        conditioning_image=Image.fromarray(image),
        num_frames=1,
        width=3,
        height=2,
        guidance_scale=1,
        num_steps=1,
    ) == ["script"]
    monkeypatch.setattr(cosmos_inference, "_invoke_cosmos_official_repo_direct", real_official_repo_direct)
    monkeypatch.setattr(cosmos_inference, "_invoke_cosmos_official_repo_script", real_official_repo_script)
    assert cosmos_inference._extract_frames(SimpleNamespace(frames=[[1, 2]])) == [1, 2]

    imageio_mod = ModuleType("imageio")
    imageio_mod.get_writer = lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("writer failed"))
    monkeypatch.setitem(sys.modules, "imageio", imageio_mod)
    monkeypatch.setattr(cosmos_inference, "_save_video", real_save_video)
    cosmos_inference._save_video([image], tmp_path / "best-effort.mp4")

    repo_root = tmp_path / "official-repo"
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(repo_root))
    assert cosmos_inference._try_load_cosmos_official_repo_direct("nvidia/Cosmos-Predict2.5-7B") is None

    init_mod = ModuleType("cosmos_oss.init")
    init_mod.init_environment = lambda: None
    init_mod.init_output_dir = lambda *_args, **_kwargs: None
    config_mod = ModuleType("cosmos_predict2.config")

    class SetupArguments:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    config_mod.SetupArguments = SetupArguments
    inference_mod = ModuleType("cosmos_predict2.inference")

    class LoadedInference:
        def __init__(self, setup) -> None:
            self.setup = setup

    inference_mod.Inference = LoadedInference
    monkeypatch.setitem(sys.modules, "cosmos_oss", ModuleType("cosmos_oss"))
    monkeypatch.setitem(sys.modules, "cosmos_oss.init", init_mod)
    monkeypatch.setitem(sys.modules, "cosmos_predict2", ModuleType("cosmos_predict2"))
    monkeypatch.setitem(sys.modules, "cosmos_predict2.config", config_mod)
    monkeypatch.setitem(sys.modules, "cosmos_predict2.inference", inference_mod)
    direct_model = cosmos_inference._try_load_cosmos_official_repo_direct("nvidia/Cosmos-Predict2.5-2B")
    assert direct_model is not None
    assert direct_model["backend"] == "official_repo_direct"
    monkeypatch.delitem(sys.modules, "cosmos_oss.init", raising=False)
    assert cosmos_inference._try_load_cosmos_official_repo_direct("nvidia/Cosmos-Predict2.5-2B") is None

    missing_repo = tmp_path / "missing-repo"
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(missing_repo))
    monkeypatch.setattr(cosmos_inference, "_persistent_worker_enabled", lambda: True)
    assert cosmos_inference._try_load_cosmos_official_repo_worker("nvidia/Cosmos-Predict2.5-2B") is None
    worker_repo = tmp_path / "worker-repo"
    worker_python = worker_repo / ".venv" / "bin" / "python"
    worker_entrypoint = worker_repo / "examples" / "inference.py"
    worker_python.parent.mkdir(parents=True)
    worker_entrypoint.parent.mkdir(parents=True)
    worker_python.write_text("", encoding="utf-8")
    worker_entrypoint.write_text("", encoding="utf-8")
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(worker_repo))
    assert cosmos_inference._try_load_cosmos_official_repo_worker("nvidia/Cosmos-Predict2.5-7B") is None
    monkeypatch.setenv("COSMOS_WORKER_PYTHON_BIN", str(tmp_path / "absent-python"))
    fallback_worker = cosmos_inference._try_load_cosmos_official_repo_worker("nvidia/Cosmos-Predict2.5-2B")
    assert fallback_worker is not None
    assert fallback_worker.python_bin == worker_python

    monkeypatch.setattr(cosmos_inference, "_skip_official_repo_script", lambda: False)
    monkeypatch.setattr(cosmos_inference, "_cold_subprocess_fallback_enabled", lambda: True)
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", "")
    assert cosmos_inference._try_load_cosmos_official_repo("nvidia/Cosmos-Predict2.5-2B") is None
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(missing_repo))
    assert cosmos_inference._try_load_cosmos_official_repo("nvidia/Cosmos-Predict2.5-2B") is None
    monkeypatch.setattr(cosmos_inference, "_DEFAULT_COSMOS_OFFICIAL_REPO_ROOT", str(worker_repo))
    assert cosmos_inference._try_load_cosmos_official_repo("nvidia/Cosmos-Predict2.5-7B") is None
    assert cosmos_inference._official_repo_model_variant("nvidia/Cosmos-Predict2.5-7B", "post-trained") is None

    script_model = {
        "backend": "official_repo_script",
        "repo_root": str(worker_repo),
        "python_bin": str(worker_python),
        "model_variant": "2B/post-trained",
        "subprocess_env": {"PATH": "x"},
    }
    monkeypatch.setattr(cosmos_inference.uuid, "uuid4", lambda: SimpleNamespace(hex="11111111"))
    monkeypatch.setattr(cosmos_inference.subprocess, "run", lambda *_args, **_kwargs: subprocess.CompletedProcess([], 0))
    with pytest.raises(RuntimeError, match="official_repo_output_missing"):
        cosmos_inference._invoke_cosmos_official_repo_script(
            model=script_model,
            conditioning_image=image,
        )

    def run_script_with_video(command, **_kwargs):
        output_dir = Path(command[command.index("-o") + 1])
        sample_name = output_dir.name
        (output_dir / f"{sample_name}.mp4").write_bytes(b"mp4")
        return subprocess.CompletedProcess(command, 0)

    imageio_v3_empty = ModuleType("imageio.v3")
    imageio_v3_empty.imiter = lambda _path: []
    imageio_pkg = ModuleType("imageio")
    imageio_pkg.v3 = imageio_v3_empty
    monkeypatch.setitem(sys.modules, "imageio", imageio_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3_empty)
    monkeypatch.setattr(cosmos_inference.uuid, "uuid4", lambda: SimpleNamespace(hex="22222222"))
    monkeypatch.setattr(cosmos_inference.subprocess, "run", run_script_with_video)
    with pytest.raises(RuntimeError, match="official_repo_output_empty"):
        cosmos_inference._invoke_cosmos_official_repo_script(
            model=script_model,
            conditioning_image=image,
        )

    missing_config = ModuleType("cosmos_predict2.config")
    monkeypatch.setitem(sys.modules, "cosmos_predict2.config", missing_config)
    with pytest.raises(RuntimeError, match="official_repo_direct_import_failed"):
        cosmos_inference._invoke_cosmos_official_repo_direct(
            model={"output_root": str(tmp_path / "direct-import"), "inference": object()},
            conditioning_image=image,
            num_frames=1,
            guidance_scale=1,
            num_steps=1,
        )

    direct_config = ModuleType("cosmos_predict2.config")

    class InferenceArguments:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    direct_config.InferenceArguments = InferenceArguments
    monkeypatch.setitem(sys.modules, "cosmos_predict2.config", direct_config)

    class DirectInference:
        def __init__(self, outputs) -> None:
            self.outputs = outputs

        def generate(self, _samples, output_dir):
            return self.outputs

    with pytest.raises(RuntimeError, match="official_repo_direct_output_missing"):
        cosmos_inference._invoke_cosmos_official_repo_direct(
            model={"output_root": str(tmp_path / "direct-missing-path"), "inference": DirectInference([tmp_path / "missing.mp4"])},
            conditioning_image=image,
            num_frames=1,
            guidance_scale=1,
            num_steps=1,
        )

    direct_video = tmp_path / "decode.mp4"
    direct_video.write_bytes(b"mp4")
    imageio_v3_decode = ModuleType("imageio.v3")
    imageio_v3_decode.imiter = lambda _path: (_ for _ in ()).throw(RuntimeError("decode failed"))
    imageio_pkg.v3 = imageio_v3_decode
    monkeypatch.setitem(sys.modules, "imageio", imageio_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3_decode)
    with pytest.raises(RuntimeError, match="official_repo_direct_decode_failed"):
        cosmos_inference._invoke_cosmos_official_repo_direct(
            model={"output_root": str(tmp_path / "direct-decode"), "inference": DirectInference([direct_video])},
            conditioning_image=image,
            num_frames=1,
            guidance_scale=1,
            num_steps=1,
        )

    imageio_v3_empty_direct = ModuleType("imageio.v3")
    imageio_v3_empty_direct.imiter = lambda _path: []
    imageio_pkg.v3 = imageio_v3_empty_direct
    monkeypatch.setitem(sys.modules, "imageio", imageio_pkg)
    monkeypatch.setitem(sys.modules, "imageio.v3", imageio_v3_empty_direct)
    with pytest.raises(RuntimeError, match="official_repo_direct_output_empty"):
        cosmos_inference._invoke_cosmos_official_repo_direct(
            model={"output_root": str(tmp_path / "direct-empty-output"), "inference": DirectInference([direct_video])},
            conditioning_image=image,
            num_frames=1,
            guidance_scale=1,
            num_steps=1,
        )

    monkeypatch.setattr(cosmos_inference.os, "environ", {"PATH": "/bin", "HF_TOKEN": "hf", "BAD": object()})
    assert cosmos_inference._select_official_repo_env_vars() == {"PATH": "/bin", "HF_TOKEN": "hf"}
