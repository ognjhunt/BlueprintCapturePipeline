"""A deploy that changes nothing a running job can see is not sent (scripts/operator_door.py)."""

# Covers (for impacted-test selection):
#   scripts/operator_door.py

from __future__ import annotations

import ast
import importlib.util
import io
import subprocess
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("operator_door_client_paths", REPO_ROOT / "scripts" / "operator_door.py")
client = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(client)

RUNTIME_ROOTS = ("src", "scripts", "deploy", "config")


def _named_strings(path: Path) -> list[str]:
    """Text a file could use as a path: Python string values (docstrings and comments
    excluded), or every non-comment line of any other text file."""
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []
    if path.suffix != ".py":
        return [line for line in text.splitlines() if not line.lstrip().startswith(("#", "//"))]
    tree = ast.parse(text)
    docstrings = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    return [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings]


def _allowlist_definition(path: Path) -> set[str]:
    """The allowlist's own entries, where the client defines it."""
    if path != REPO_ROOT / "scripts" / "operator_door.py":
        return set()
    return set(client.NON_RUNTIME_PATHS)


# Files reviewed by hand that use an allowlisted directory's bare name for some other path.
REVIEWED = {
    # ops/launch-readiness/ in the BlueprintCapture and WebApp checkouts, not this repo's docs/.
    "src/blueprint_pipeline/city_launch_autonomy_harness.py": "launch-readiness",
}


def _offences(path: Path, value: str) -> list[str]:
    """Which allowlisted directories ``value`` names: its whole path, or its last
    component alone (code that builds ``Path("docs") / "launch-readiness"``)."""
    found = []
    for entry in client.NON_RUNTIME_PATHS:
        whole = entry.rstrip("/")
        name = whole.rsplit("/", 1)[-1]
        relative = path.relative_to(REPO_ROOT).as_posix()
        if whole in value or value.strip("/") == name and REVIEWED.get(relative) != name:
            found.append(f"{relative}: {value[:120]!r}")
    return found


def test_no_runtime_code_names_a_non_runtime_path() -> None:
    """The allowlist stays sound: nothing that runs may read a path the client treats as non-runtime."""
    names = [entry.rstrip("/").rsplit("/", 1)[-1].encode() for entry in client.NON_RUNTIME_PATHS]
    offenders = []
    for root in RUNTIME_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*")):
            if not path.is_file() or path.is_symlink() or "__pycache__" in path.parts:
                continue
            # Every offence contains a directory's last component; most files have none.
            if not any(name in path.read_bytes() for name in names):
                continue
            own = _allowlist_definition(path)
            for value in _named_strings(path):
                if value not in own:
                    offenders += _offences(path, value)
    assert offenders == []


def test_the_allowlist_is_directories_never_docs_or_tests_wholesale() -> None:
    assert client.NON_RUNTIME_PATHS
    for entry in client.NON_RUNTIME_PATHS:
        assert entry.endswith("/") and not entry.startswith("/") and ".." not in entry.split("/")
        assert entry.rstrip("/") not in {"docs", "tests", "src", "scripts", "deploy", "config"}


def test_the_soundness_check_would_catch_a_reader(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    reader = tmp_path / "reader.py"
    reader.write_text('"""Mentions launch-readiness in a docstring only."""\n'
                      '# docs/launch-readiness/x.md in a comment\n'
                      'PATH = ROOT / "docs" / "launch-readiness" / "plan.md"\n', encoding="utf-8")
    monkeypatch.setattr(sys.modules[__name__], "REPO_ROOT", tmp_path)
    assert [hit for value in _named_strings(reader) for hit in _offences(reader, value)] == [
        "reader.py: 'launch-readiness'"]
    unit = tmp_path / "x.service"
    unit.write_text("# docs/launch-readiness\nExecStart=cat /opt/x/docs/launch-readiness/a.md\n", encoding="utf-8")
    assert [hit for value in _named_strings(unit) for hit in _offences(unit, value)] == [
        "x.service: 'ExecStart=cat /opt/x/docs/launch-readiness/a.md'"]


# --- the deploy command against a temporary git repository ---


def _git(repo: Path, *argv: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *argv], check=True, capture_output=True, text=True,
                          env={"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
                               "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
                               "HOME": str(repo), "PATH": "/usr/bin:/bin"}).stdout.strip()


def _commit(repo: Path, files: dict[str, str]) -> str:
    for name, text in files.items():
        (repo / name).parent.mkdir(parents=True, exist_ok=True)
        (repo / name).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c")
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture()
def repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    base = _commit(root, {"src/app.py": "x = 1\n", "docs/launch-readiness/plan.md": "a\n", "docs/schemas/s.json": "{}\n"})
    docs = _commit(root, {"docs/launch-readiness/plan.md": "b\n", "docs/launch-readiness/new.md": "n\n"})
    monkeypatch.setattr(client, "CLIENT_REPO", root)
    submitted: list[dict[str, Any]] = []
    monkeypatch.setattr(client, "_submit", lambda body, args: submitted.append(body) or 0)
    return {"root": root, "base": base, "docs": docs, "submitted": submitted}


def _active(monkeypatch: pytest.MonkeyPatch, commit: str | Exception) -> None:
    def status(method: str, route: str, *_a: Any, **_k: Any) -> Any:
        assert (method, route) == ("GET", "/status")
        if isinstance(commit, Exception):
            raise commit
        return {"active_release": {"commit": commit}}

    monkeypatch.setattr(client, "_json", status)


def _deploy(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    with redirect_stdout(out):
        code = client.main(["deploy", *argv])
    return code, out.getvalue()


def test_a_difference_only_in_non_runtime_files_is_not_deployed(repo, monkeypatch) -> None:
    _active(monkeypatch, repo["base"])
    code, out = _deploy(repo["docs"])
    assert code == 0 and repo["submitted"] == []
    assert out == (f"Not deployed: {repo['docs']} differs from the active release {repo['base']} only in 2 "
                   "non-runtime file(s) under docs/launch-readiness/, so the active release already runs this "
                   "code; add --even-if-no-runtime-change to deploy it anyway.\n")


def test_a_rollback_is_compared_the_same_way(repo, monkeypatch) -> None:
    _active(monkeypatch, repo["docs"])
    code, out = _deploy(repo["base"])
    assert code == 0 and repo["submitted"] == [] and out.startswith("Not deployed:")
    runtime = _commit(repo["root"], {"src/app.py": "x = 2\n"})
    _active(monkeypatch, runtime)
    assert _deploy(repo["base"])[0] == 0
    assert [body["commit"] for body in repo["submitted"]] == [repo["base"]]


@pytest.mark.parametrize("files", [{"src/app.py": "x = 2\n"}, {"docs/schemas/s.json": "[]\n"},
                                   {"docs/launch-readiness.md": "x\n"}, {"tests/t.py": "\n"}])
def test_a_difference_in_a_runtime_file_deploys(repo, monkeypatch, files) -> None:
    changed = _commit(repo["root"], files)
    _active(monkeypatch, repo["docs"])
    assert _deploy(changed)[0] == 0
    assert repo["submitted"] == [{"kind": "deploy", "commit": changed, "wait_for_idle": True}]


def test_a_rename_out_of_the_non_runtime_directory_deploys(repo, monkeypatch) -> None:
    _git(repo["root"], "mv", "docs/launch-readiness/new.md", "docs/moved.md")
    moved = _commit(repo["root"], {})
    _active(monkeypatch, repo["docs"])
    assert _deploy(moved)[0] == 0 and len(repo["submitted"]) == 1


def test_the_force_flag_deploys(repo, monkeypatch) -> None:
    _active(monkeypatch, repo["base"])
    assert _deploy(repo["docs"], "--even-if-no-runtime-change")[0] == 0
    assert [body["commit"] for body in repo["submitted"]] == [repo["docs"]]


def test_the_same_commit_deploys_again(repo, monkeypatch) -> None:
    """A redeploy of the active commit (one no receipt proves) is not a no-change skip."""
    _active(monkeypatch, repo["docs"])
    assert _deploy(repo["docs"])[0] == 0 and len(repo["submitted"]) == 1


@pytest.mark.parametrize("active", ["f" * 40, None, "", "not-a-sha"])
def test_an_unknown_or_unreadable_active_release_deploys(repo, monkeypatch, active) -> None:
    if active is None:
        _active(monkeypatch, client.DoorError(4, "door unreachable"))
    else:
        _active(monkeypatch, active)
    assert _deploy(repo["docs"])[0] == 0 and len(repo["submitted"]) == 1


def test_a_commit_not_in_the_local_clone_deploys(repo, monkeypatch) -> None:
    _active(monkeypatch, repo["base"])
    unknown = "e" * 40
    assert _deploy(unknown)[0] == 0
    assert [body["commit"] for body in repo["submitted"]] == [unknown]


def test_a_git_failure_deploys(repo, monkeypatch) -> None:
    _active(monkeypatch, repo["base"])
    monkeypatch.setattr(client, "CLIENT_REPO", repo["root"] / "missing")
    assert _deploy(repo["docs"])[0] == 0 and len(repo["submitted"]) == 1


def test_a_production_deploy_is_never_skipped(repo, monkeypatch) -> None:
    """It carries release provenance, which running code reads, so it is sent as before."""
    monkeypatch.setattr(client, "_json", lambda *_a, **_k: pytest.fail("no status read expected"))
    args = client.argparse.Namespace(even_if_no_runtime_change=False, release_provenance="p",
                                     release_provenance_sha256="0" * 64)
    assert client._no_runtime_change(repo["docs"], args) is False
