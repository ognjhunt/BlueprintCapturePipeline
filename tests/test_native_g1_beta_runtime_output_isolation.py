"""Real copied venv links cannot become evidence or prevent its retention."""
import venv
import zipfile

import pytest

from blueprint_pipeline.native_g1_beta_runtime import _retained_bytes
from blueprint_pipeline.provider_runtime_bundle_contract import g1_provider_output_contract
from tests.test_native_g1_beta_provider_plumbing import KIND, RESULT, _packaging


def runtime_output(tmp_path):
    output = tmp_path / "output"
    environment = output / "policy-runtime-build/policy-runtime"
    venv.EnvBuilder(with_pip=False, symlinks=False).create(environment)
    # This is the actual Linux venv behavior that the earlier fixtures missed.
    assert (environment / "lib64").is_symlink()
    (output / RESULT).write_text('{"status":"blocked"}\n')
    retained = [output / RESULT, output / "policy-runtime-build/build.log", output / "cells/cell/query.json"]
    for path in retained[1:]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("retained evidence\n")
    return output, retained


def test_real_copied_venv_does_not_block_scanner_or_actual_generated_packager(tmp_path):
    output, retained = runtime_output(tmp_path)
    # All exact execution-only roots remain excluded, including links that
    # target data outside the output tree. They are never traversed or copied.
    outside = tmp_path / "execution-inputs"
    outside.mkdir()
    (outside / "model.bin").write_bytes(b"private execution input")
    for prefix in g1_provider_output_contract(KIND)[2]:
        root = output / prefix.rstrip("/")
        if root.exists():
            continue
        root.parent.mkdir(parents=True, exist_ok=True)
        root.symlink_to(outside, target_is_directory=True)
    assert _retained_bytes(output) == sum(path.stat().st_size for path in retained)
    packed = _packaging(tmp_path, output)
    assert packed.returncode == 0, packed.stderr
    with zipfile.ZipFile(tmp_path / "adp_arena_provider_runtime_output.zip") as archive:
        assert set(archive.namelist()) == {path.relative_to(output).as_posix() for path in retained}
        for path in retained:
            assert archive.read(path.relative_to(output).as_posix()) == path.read_bytes()


@pytest.mark.parametrize("name", ["cells/cell/linked-query.json", "models/checkpoints-lookalike/linked.json",
                                  "policy-runtime-build/policy-runtime-evidence/linked.json"])
def test_retained_evidence_symlink_still_fails_both_scanner_and_packager(tmp_path, name):
    output, _ = runtime_output(tmp_path)
    link = output / name
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(output / RESULT)
    with pytest.raises(ValueError, match="g1_beta_output_symlink_forbidden"):
        _retained_bytes(output)
    packed = _packaging(tmp_path, output)
    assert packed.returncode != 0 and b"g1_beta_output_not_regular" in packed.stderr
    assert not (tmp_path / "adp_arena_provider_runtime_output.zip").exists()
