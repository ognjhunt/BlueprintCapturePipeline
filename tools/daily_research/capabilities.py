"""Exact instruction-only files from the approved September 30 skill review.

Template GETs expose file paths/sizes, never inline bytes. Verify our preserved
bytes and supply them as session overrides before the agent starts. No template
mutation, skill registration, packages, credentials or additional network.
"""
import base64
import hashlib
from pathlib import Path

ROOT = "/workspace/capabilities/blueprint"
# Actual mounted-file checks in the preserved validation receipt:
# https://app.notion.com/p/3eb80154161d81b4a0ddff9bdcfe5af6
# deep-research source commit: 42dd24080fce6d731d00e2a1134f398c3da4171b
TEMPLATE_FILES = {
    "blueprint-evidence-qualification/SKILL.md":
        (3965, "5fdc5f3c6dde9cc252cd43bd4165d09b4b2d5686835b70961d00e0d7e91f0756"),
    "blueprint-evidence-qualification/references/prospect-contract.md":
        (1838, "34cf345832dea8226d14ed31b3197a44c293ac91dc903fb596a9b89d05299258"),
    "deep-research/LICENSE":
        (1072, "3a9cf254e155282014880e9569b9039bc17ce6a43919df23741cf14d24481244"),
    "deep-research/SKILL.md":
        (5386, "2646cdf3942d918e84febf020b289fbfb7b5cf601e43ee7e7349e6c5105941c5"),
}

# User-requested verification capability revision. Session inline overrides use
# these reviewed release bytes while preflight still checks the unchanged saved
# template's original discovery inventory. No live template mutation is needed.
FILES = dict(TEMPLATE_FILES)
FILES['blueprint-evidence-qualification/SKILL.md'] = (7714, '4d929210c18681d2136230778d719762ef63568a9ed33c8f2826bba1af76ffa4')
FILES['blueprint-evidence-qualification/references/prospect-contract.md'] = (2226, '39571718234ce2f7536a56f6e8440183536c259bf9b4e9a0d54944cc9aaaf6b3')


def inline_files():
    """Return only the four reviewed files, refusing missing/changed local bytes."""
    result = []
    for name, (size, expected) in FILES.items():
        path = Path(__file__).with_name("capabilities") / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("reviewed_skill_file_missing")
        with path.open("rb") as handle:
            raw = handle.read(size + 1)
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("reviewed_skill_file_changed")
        result.append({"type": "inline", "path": ROOT + "/" + name,
                       "data": base64.b64encode(raw).decode("ascii")})
    return result


def setup_commands():
    """Verify the actual reviewed overrides instead of the template's old bytes."""
    files = {ROOT + "/" + name: value for name, value in FILES.items()}
    script = (
        "import hashlib\nfrom pathlib import Path\n"
        f"files = {files!r}\n"
        "for name, (size, expected) in files.items():\n"
        "    path = Path(name)\n"
        "    assert all(not p.is_symlink() for p in (path, *path.parents)) and path.is_file(), 'reviewed_skill_file_missing'\n"
        "    assert path.stat().st_size == size, 'reviewed_skill_file_changed'\n"
        "    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'reviewed_skill_file_changed'\n"
    )
    return [{"command": "python3 - <<'PY'\n" + script + "PY"}]


def check_template(template):
    """Check the documented file-discovery setup, without pretending to hash GET bytes."""
    if (template.get("capability_directories") != [ROOT]
            or template.get("skills") != []
            or template.get("plugins") != []):
        raise ValueError("template_skill_discovery_mismatch")
    files = template.get("files")
    if not isinstance(files, list) or len(files) != len(TEMPLATE_FILES):
        raise ValueError("template_skill_files_mismatch")
    expected = {ROOT + "/" + name: size for name, (size, _) in TEMPLATE_FILES.items()}
    observed = {}
    for item in files:
        if (not isinstance(item, dict) or item.get("type") != "inline"
                or item.get("path") not in expected or item["path"] in observed
                or type(item.get("size_bytes")) is not int):
            raise ValueError("template_skill_files_mismatch")
        observed[item["path"]] = item["size_bytes"]
    if observed != expected:
        raise ValueError("template_skill_files_mismatch")
    inline_files()
    return {"discovery": "capability_directories", "local_file_hashes_verified": True,
            "template_inline_content_verified": False,
            "session_inline_files_required": True,
            "file_sha256": {ROOT + "/" + name: sha for name, (_, sha) in FILES.items()}}
