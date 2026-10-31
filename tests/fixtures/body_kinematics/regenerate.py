"""Regenerate the body-kinematics fixtures from the pinned recipes (development tool; needs git and network).

Each body is prepared exactly as Blueprint prepares it: the recipe's pinned public sources are
retained with ``deployment_embodiment_assets.materialize_sources`` (served from sparse git checkouts at
the pinned revisions, so every byte is checked against its git blob), then
``deployment_embodiment_runtime_configuration.prepare_runtime_configuration`` writes the robot URDF and
the manifest fields. This keeps the generated ``robot.urdf`` and ``runtime-configuration.json`` unchanged,
the source license files, and the Franka kinematics files that show FR3 and FER (Panda) share one chain.
Mesh files are not retained.

    python tests/fixtures/body_kinematics/regenerate.py <scratch directory>
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import unquote

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

from blueprint_pipeline import deployment_embodiment_assets as assets  # noqa: E402
from blueprint_pipeline import deployment_embodiment_runtime_configuration as runtime  # noqa: E402

BODIES = (
    "universal_robots_ur5e_robotiq_2f85_sim", "franka_research_3_robotiq_2f85_sim", "franka_research_3_bimanual",
    "kinova_gen3", "ufactory_xarm7_bimanual", "kuka_lbr_iiwa_14_robotiq_2f85_sim", "fanuc_crx_10ia_robotiq_2f85_sim",
    "fanuc_m710ic_45m_robotiq_2f85_sim", "yaskawa_gp12_robotiq_2f85_sim", "abb_gofa_crb15000_5_95_robotiq_2f85_sim",
    "trlc_dk1_bimanual", "i2rt_yam_bimanual", "agilex_piper_bimanual",
)
FRANKA = ("https://github.com/frankarobotics/franka_description", "robots/fr3/kinematics.yaml", "robots/fer/kinematics.yaml")


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _sources(identity: str) -> list[dict]:
    recipe = assets.recipe_for(identity)
    composition = recipe.get("composition") or {}
    rows = [recipe["source"]]
    if composition.get("arm_embodiment_id"):
        rows.append(assets.recipe_for(composition["arm_embodiment_id"])["source"])
    if composition.get("tool_source"):
        rows.append(composition["tool_source"])
    return list({(row["repository_url"], row["revision"], row["entry_path"]): row for row in rows}.values())


def _checkout(scratch: Path, url: str, revision: str, paths: set[str]) -> Path:
    folder = scratch / (url.removeprefix("https://github.com/").replace("/", "__") + "@" + revision)
    if not (folder / ".done").exists():
        folder.mkdir(parents=True, exist_ok=True)

        def git(*args: str) -> None:
            subprocess.run(["git", "-C", str(folder), *args], check=True)

        git("init", "-q")
        subprocess.run(["git", "-C", str(folder), "remote", "add", "origin", url], check=False)
        git("sparse-checkout", "set", "--no-cone", *sorted("/" + path for path in paths))
        git("fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", revision)
        git("checkout", "-q", revision)
        (folder / ".done").write_text("\n".join(sorted(paths)))
    return folder


def main(scratch: Path) -> None:
    wanted: dict[tuple[str, str], set[str]] = {}
    for identity in BODIES:
        for source in _sources(identity):
            wanted.setdefault((source["repository_url"], source["revision"]), set()).update(source["include_paths"])
    franka_revision = assets.recipe_for("franka_research_3_bimanual")["source"]["revision"]
    wanted.setdefault((FRANKA[0], franka_revision), set()).update(FRANKA[1:])
    checkouts = {url.removeprefix("https://github.com/"): _checkout(scratch / "git", url, revision, paths)
                 for (url, revision), paths in wanted.items()}

    def get(url: str, *, maximum: int) -> bytes:
        project, folder = next((p, f) for p, f in checkouts.items() if "/" + p + "/" in url)
        if "api.github.com" in url:
            revision = url.split("/git/trees/", 1)[1].split("?", 1)[0]
            listed = subprocess.run(["git", "-C", str(folder), "ls-tree", "-r", revision], capture_output=True,
                                    text=True, check=True).stdout.splitlines()
            tree = []
            for line in listed:
                meta, path = line.split("\t", 1)
                mode, kind, sha = meta.split()
                if (folder / path).is_file():  # outside the sparse checkout is never selected
                    tree.append({"path": path, "type": kind, "mode": mode, "size": (folder / path).stat().st_size,
                                 "sha": sha})
            return json.dumps({"truncated": False, "tree": tree}).encode()
        return (folder / unquote(url.split("/" + project + "/", 1)[1].split("/", 1)[1])).read_bytes()

    provenance = {"schema_version": "body_kinematics_fixtures.v1", "claim_ceiling": "development_only",
                  "notice": ("Generated derivatives of the listed public sources: xacro expansion and Blueprint "
                             "simulation composition by the named generator. They are not the publishers' files; "
                             "mesh files are not retained; each source's license file is kept under licenses/."),
                  "generator": "deployment_embodiment_runtime_configuration.prepare_runtime_configuration",
                  "bodies": {}}
    shutil.rmtree(HERE / "licenses", ignore_errors=True)
    for identity in BODIES:
        bundle, configured = scratch / (identity + "-sources"), scratch / (identity + "-configured")
        if not bundle.exists():
            assets.materialize_sources(identity, bundle, get=get)
        record = runtime.prepare_runtime_configuration(bundle, configured)
        receipt = json.loads((bundle / "source-bundle.json").read_text())
        target = HERE / identity
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir()
        for name in ("robot.urdf", "runtime-configuration.json"):
            shutil.copyfile(configured / name, target / name)
        licenses = []
        for row, source in zip(receipt.get("licenses") or [receipt["license"]], _sources(identity)[-2:]
                               if receipt.get("licenses") else _sources(identity)):
            repository = source["repository_url"].removeprefix("https://github.com/").replace("/", "__")
            prefix = "" if not receipt.get("licenses") else row["path"].split("/source/", 1)[0] + "/source/"
            files = [row["path"]] + [prefix + "NOTICE"] * ((bundle / "source" / prefix / "NOTICE").is_file())
            for path in files:
                kept = Path("licenses") / repository / path.removeprefix(prefix)
                (HERE / kept).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(bundle / "source" / path, HERE / kept)
                licenses.append({"path": kept.as_posix(), "sha256": _sha((HERE / kept).read_bytes()),
                                 "identifier": row["identifier"] if path == row["path"] else "notice"})
        provenance["bodies"][identity] = {
            "sources": [{"repository_url": s["repository_url"], "revision": s["revision"], "entry_path": s["entry_path"],
                         "license_id": s["license_id"]} for s in _sources(identity)],
            "source_bundle_digest": record["source_bundle_digest"], "configuration_digest": record["configuration_digest"],
            "robot_urdf_sha256": _sha((target / "robot.urdf").read_bytes()), "licenses": licenses}
    folder = checkouts[FRANKA[0].removeprefix("https://github.com/")]
    kept = HERE / "franka_kinematics"
    shutil.rmtree(kept, ignore_errors=True)
    kept.mkdir()
    for path in FRANKA[1:]:
        shutil.copyfile(folder / path, kept / (path.split("/")[1] + "_kinematics.yaml"))
    provenance["franka_kinematics"] = {
        "repository_url": FRANKA[0], "revision": franka_revision, "license_id": "Apache-2.0",
        "files": {path: _sha((folder / path).read_bytes()) for path in FRANKA[1:]},
        "claim": "FR3 and FER (Franka Emika Panda) declare byte-identical joint origins at this revision."}
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
