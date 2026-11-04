"""Executable code is never specialised to one customer's scene, capture or task.

Doctrine (AGENTS.md, CLAUDE.md): every fix is general. A capture id in a comment
or a docstring records where an incident was seen, and stays allowed. A capture
or scene id in a string the program evaluates (a comparison, a lookup key, a
hard-coded path, an f-string part, a bytes literal) makes behaviour depend on
whose video it is, and that is forbidden. So is comparing a value with the
words of one customer's task.

Category knowledge is general and stays allowed: "dining table" mapped to
typical sizes (``atlas_scene_measurements``), object-category vocabularies.

Every ``src/blueprint_pipeline`` module and every top-level script is parsed;
comments are not in the syntax tree and module, class and function docstrings
are skipped. A file whose text names no id or phrase at all is not parsed, which
keeps this an always-on sentinel (``impacted_test_selection.SENTINEL_TESTS``).
"""

from __future__ import annotations

import ast
import functools
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

SCENE_IDENTITY = re.compile(
    r"(?:site-|walkthrough-)?capture-[0-9a-f]{8}"  # capture-447ae4f4, walkthrough-capture-...
    r"|scene-[0-9a-f]{8,}"
    # One site capture by (part of) its uuid. The bare prefix names them all and is general.
    r"|(?<![0-9A-Za-z])site-capture-[0-9a-f]{4,}(?![g-zG-Z])"
)
# Every id above ends "capture-" or "scene-" and hex; finding that is 50 times faster.
_MAY_NAME_A_SCENE = re.compile(r"e-[0-9a-f]{4}")

#: Objects and tasks from real customer captures (capture-447ae4f4, capture-4b60431e),
#: flagged only where code compares a value with them. Never add category words
#: ("dining table", "drawer", "mug"): those are general knowledge.
CUSTOMER_TASK_PHRASES = ("wooden tray",)
_PHRASE_TEXT = re.compile("|".join(p.replace(" ", r"[\s_-]+") for p in CUSTOMER_TASK_PHRASES))

#: (file, dict key, exact value) -> why it is a recorded fact, not scene-specific logic.
RECORDED_PROVENANCE = {
    ("src/blueprint_pipeline/atlas_gap_composite.py", "basis",
     "capture-447ae4f4 retained frames, 2026-10-31"):
        "Calibration source written into each composite receipt and never read back; the receipt"
        " is inside every refined splat's sealed input_binding_digest, so rewording it would"
        " change sealed digests and refuse retained roots.",
}

#: Scene-specific logic that predates this guard: the defect it exists to stop. An
#: entry may only be removed (an entry nothing matches fails below), never added.
#: Empty since the drawer depth prior became a cabinet category prior.
SCENE_SPECIFIC_DEBT: dict[tuple[str, str | None, str], str] = {}


@dataclass(frozen=True)
class Hit:
    path: str
    line: int
    dict_key: str | None
    value: str
    match: str
    kind: str

    @property
    def allowance(self) -> tuple[str, str | None, str]:
        return (self.path, self.dict_key, self.value)


def _leaves(node: ast.AST):
    if isinstance(node, ast.Constant):
        yield node
    elif isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for element in node.elts:
            yield from _leaves(element)
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id in {"frozenset", "set", "tuple", "list"}):
        for argument in node.args:
            yield from _leaves(argument)


def findings(source: str, path: str) -> list[Hit]:
    nodes = list(ast.walk(ast.parse(source, filename=path)))
    docstrings, keys, compared, constants = set(), {}, [], []
    for node in nodes:
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(first.value)
        elif isinstance(node, ast.Dict):
            keys.update((value, key.value) for key, value in zip(node.keys, node.values)
                        if isinstance(key, ast.Constant) and isinstance(key.value, str))
        elif isinstance(node, ast.Compare):  # Constants a comparison tests a value against.
            for operand in (node.left, *node.comparators):
                compared.extend(_leaves(operand))
        elif isinstance(node, ast.MatchValue):
            compared.extend(_leaves(node.value))
        elif isinstance(node, ast.Constant) and isinstance(node.value, (str, bytes)):
            constants.append(node)
    hits = []
    for node in constants:
        if node not in docstrings:
            text = node.value.decode("latin-1") if isinstance(node.value, bytes) else node.value
            if match := SCENE_IDENTITY.search(text):
                hits.append(Hit(path, node.lineno, keys.get(node), text, match.group(0),
                                "names one customer scene or capture"))
    for node in compared:
        if isinstance(node.value, str) and (match := _PHRASE_TEXT.search(node.value.lower())):
            hits.append(Hit(path, node.lineno, keys.get(node), node.value, match.group(0),
                            "compares a value with one customer's task phrase"))
    return sorted(hits, key=lambda hit: (hit.line, hit.match))


@functools.lru_cache(maxsize=1)
def repository_hits() -> tuple[Hit, ...]:
    files = [*sorted((REPO_ROOT / "src" / "blueprint_pipeline").rglob("*.py")),
             *sorted((REPO_ROOT / "scripts").glob("*.py"))]
    hits = []
    for path in files:
        source = path.read_text(encoding="utf-8")
        if _MAY_NAME_A_SCENE.search(source) or _PHRASE_TEXT.search(source.lower()):
            hits.extend(findings(source, path.relative_to(REPO_ROOT).as_posix()))
    return tuple(hits)


def test_executable_code_names_no_customer_scene() -> None:
    allowed = RECORDED_PROVENANCE.keys() | SCENE_SPECIFIC_DEBT.keys()
    forbidden = [hit for hit in repository_hits() if hit.allowance not in allowed]
    assert not forbidden, (
        "Scene-specific logic is forbidden (CLAUDE.md and AGENTS.md: every fix is general,"
        " nothing specific to one scene, capture or task). Executable code here does:\n"
        + "\n".join(f"  {hit.path}:{hit.line}: {hit.kind}: {hit.match!r} in constant {hit.value!r}"
                    for hit in forbidden)
        + "\nMake the code work for every scene. Incident provenance belongs in a comment or a"
        " docstring. A recorded data string no code branches on may go in RECORDED_PROVENANCE"
        " (tests/test_no_scene_specific_code.py), keyed on file, dict key and value, with its reason."
    )


def test_every_allowance_still_matches_code() -> None:
    found = {hit.allowance for hit in repository_hits()}
    stale = sorted((RECORDED_PROVENANCE.keys() | SCENE_SPECIFIC_DEBT.keys()) - found)
    assert not stale, f"Remove allowances nothing matches any more: {stale}"


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ('"""Seen on capture-447ae4f4."""\nx = 1  # capture-447ae4f4\n', []),
        ('def f():\n    """walkthrough-capture-0123abcd hung."""\n', []),
        ('if capture_id == "capture-447ae4f4":\n    pass\n', ["capture-447ae4f4"]),
        ('path = f"/work/site-capture-7d655e25-{suffix}"\n', ["site-capture-7d655e25"]),
        ('key = b"scene-0123456789abcdef"\n', ["scene-0123456789abcdef"]),
        ('PREFIXES = ("scene-", "site-capture-")\n', []),
        ('URL = "/api/website-capture-withdrawals"\n', []),
        ('name = f"site-capture-{capture_id}"\n', []),
        ('if label in {"wooden tray", "box"}:\n    pass\n', ["wooden tray"]),
        ('label = "wooden tray"\n', []),
        ('SIZES = {"dining table": (1.6, 0.9, 0.75)}\nif label == "dining table":\n    pass\n', []),
    ],
)
def test_scanner_flags_code_and_spares_provenance(source: str, flagged: list[str]) -> None:
    assert [hit.match for hit in findings(source, "example.py")] == flagged
