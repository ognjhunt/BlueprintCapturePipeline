from __future__ import annotations

import importlib.util
import json
import subprocess
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "deploy/docker/robot_eval_worker/groot_oscar_closed_loop/repair_embedded_carrier.py"
)
SPEC = importlib.util.spec_from_file_location("repair_embedded_carrier", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _repo(path: Path) -> str:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)
    (path / "source.txt").write_text("sealed\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "sealed"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_repair_validates_sources_and_binds_offline_cosmos_selector(tmp_path: Path) -> None:
    roots = {name: tmp_path / name for name in ("wbc", "groot", "oscar")}
    revisions = {name: _repo(path) for name, path in roots.items()}
    sonic = tmp_path / "sonic"
    (sonic / "processor").mkdir(parents=True)
    (sonic / "config.json").write_text(
        json.dumps(
            {
                "blueprint_original_model_name": MODULE.COSMOS_REPO,
                "blueprint_model_revision": MODULE.COSMOS_REVISION,
                "model_name": str(tmp_path / "models/cosmos-reason2-2b"),
            }
        ),
        encoding="utf-8",
    )
    (sonic / "processor/processor_config.json").write_text(
        json.dumps({"processor_kwargs": {"model_name": MODULE.COSMOS_REPO}}),
        encoding="utf-8",
    )
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text("{}\n", encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    (models / "cosmos-reason2-2b").symlink_to(snapshot, target_is_directory=True)

    result = MODULE.repair(
        wbc_revision=revisions["wbc"],
        groot_revision=revisions["groot"],
        oscar_revision=revisions["oscar"],
        roots=roots,
        sonic=sonic,
        models_root=models,
    )

    assert result["status"] == "repaired"
    assert (roots["wbc"] / ".blueprint-source-revision").read_text().strip() == revisions["wbc"]
    config = json.loads((sonic / "config.json").read_text(encoding="utf-8"))
    processor = json.loads(
        (sonic / "processor/processor_config.json").read_text(encoding="utf-8")
    )
    assert "nvidia/Cosmos-Reason2-2B/../.." in config["model_name"]
    assert processor["processor_kwargs"]["model_name"] == config["model_name"]
    assert Path(config["model_name"]).resolve() == models / "cosmos-selector"
    assert (models / "cosmos-selector/config.json").resolve() == snapshot / "config.json"
