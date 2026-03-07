"""Security tests for cfg_args parsing in inpaint360gs_runner."""

from __future__ import annotations

from pathlib import Path

import sys

_repo_scripts = Path(__file__).resolve().parents[1] / "scripts"
if str(_repo_scripts) not in sys.path:
    sys.path.insert(0, str(_repo_scripts))

import inpaint360gs_runner as runner


class TestPatchCfgArgsSecurity:
    def test_patch_cfg_args_merges_without_eval(self, tmp_path: Path) -> None:
        cfg_args = tmp_path / "cfg_args"
        cfg_args.write_text("Namespace(model_path='/tmp/model', sh_degree=3)\n", encoding="utf-8")

        runner._patch_cfg_args(cfg_args, {"object_path": "/tmp/object", "sh_degree": 99})

        patched = cfg_args.read_text(encoding="utf-8")
        ns = runner._parse_cfg_args_namespace(patched.strip())
        assert ns.model_path == "/tmp/model"
        assert ns.object_path == "/tmp/object"
        # existing keys are preserved unless explicitly overwritten
        assert ns.sh_degree == 3

    def test_patch_cfg_args_rejects_malicious_expression(self, tmp_path: Path) -> None:
        cfg_args = tmp_path / "cfg_args"
        cfg_args.write_text(
            "Namespace(foo=__import__('os').system('echo pwned'))\n",
            encoding="utf-8",
        )

        original = cfg_args.read_text(encoding="utf-8")
        runner._patch_cfg_args(cfg_args, {"safe": 1})

        # parser failure should skip patch and preserve original file
        assert cfg_args.read_text(encoding="utf-8") == original
