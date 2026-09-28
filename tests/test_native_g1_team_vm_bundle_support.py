"""Execute launcher argument/receipt contracts without provisioning or GPU spend."""

import hashlib
import json
import subprocess
import sys
import textwrap

import pytest

from blueprint_pipeline.native_g1_team_vm_bundle_support import (
    BOOTSTRAP_RESULT_FILENAME, vm_host_entrypoint,
)


@pytest.mark.parametrize("provision_exit,host_exit", [(23, 0), (0, 7), (0, 0)])
def test_vm_shell_uses_canonical_paths_isolated_python_and_terminal_receipt(tmp_path, provision_exit, host_exit):
    # Deliberate test doubles prove launcher plumbing only. Actual Linux runtime
    # provisioning has a separately retained proof and does not run here.
    root = tmp_path.resolve() / "bundle with spaces"
    runtime = root / "provider_runtime"
    package = runtime / "blueprint_pipeline"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (runtime / "native_g1_team_provider_manifest.json").write_text(json.dumps({
        "manifest_digest": "sha256:" + "a" * 64}))
    (package / "native_g1_team_vm_bootstrap.py").write_text(textwrap.dedent(f'''
        import argparse,json,sys
        from pathlib import Path
        assert sys.flags.isolated and sys.flags.no_site and sys.dont_write_bytecode
        parser=argparse.ArgumentParser()
        for name in ('package-root','implementation-commit','destination-root',
                     'provider-runtime-root','provider-manifest-digest'):
            parser.add_argument('--'+name,required=True)
        args=parser.parse_args()
        for name in ('package_root','destination_root','provider_runtime_root'):
            value=Path(getattr(args,name))
            assert value.is_absolute() and value.resolve()==value, name
        destination=Path(args.destination_root)
        executable=destination/'python/bin/python3.12'
        executable.parent.mkdir(parents=True)
        executable.symlink_to({str(sys.executable)!r})
        (destination/'fixture-bootstrap-arguments.json').write_text(json.dumps(vars(args)))
        raise SystemExit({provision_exit})
    '''))
    (package / "native_g1_team_vm_host.py").write_text(textwrap.dedent(f'''
        import json,sys
        from pathlib import Path
        def main(args):
            assert sys.flags.isolated and sys.dont_write_bytecode
            assert args[0]=='--runtime-root' and args[2]=='--output-dir'
            for value in (args[1],args[3]):
                assert Path(value).resolve()==Path(value)
            (Path(args[3])/'fixture-host-arguments.json').write_text(json.dumps(args))
            return {host_exit}
    '''))
    script = runtime / "run_g1_team_vm_host.sh"
    script.write_text(vm_host_entrypoint("b" * 40))
    result = subprocess.run(["bash", str(script)], cwd=tmp_path, capture_output=True, text=True, timeout=15)
    assert result.returncode == (provision_exit or host_exit), result.stderr
    receipt = json.loads((root / "runtime_output" / BOOTSTRAP_RESULT_FILENAME).read_text())
    assert receipt["runner_exit_code"] == result.returncode
    assert receipt["stage_reached"] == ("host-python-provisioning" if provision_exit else "vm-host")
    assert receipt["status"] == ("blocked" if result.returncode else "host_exited")
    assert receipt["gpu_runtime_qualified"] is False
    assert receipt["provider_mutation_performed"] is False
    assert receipt["claim_ceiling"] == "development_only"
    digest = receipt.pop("receipt_digest")
    encoded = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert digest == "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
    arguments = json.loads((root / "host-python-runtime/fixture-bootstrap-arguments.json").read_text())
    assert arguments["implementation_commit"] == "b" * 40
    assert arguments["destination_root"] == str(root / "host-python-runtime")
    assert (root / "runtime_output/fixture-host-arguments.json").exists() is (not provision_exit)
    assert not list(root.rglob("__pycache__"))
