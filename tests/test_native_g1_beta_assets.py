import hashlib
import io
import time

import pytest

from blueprint_pipeline.native_g1_beta_assets import ORDER, selected_asset_inventory, selected_candidates
from blueprint_pipeline.native_g1_beta_download import _fetch


def test_dp_selected_inventory_does_not_include_pi_or_tokenizer():
    row = selected_asset_inventory([ORDER[0]])
    assert row["candidate_ids"] == [ORDER[0]]
    assert not row["pi_tokenizer_required"]
    assert {f["candidate_id"] for f in row["checkpoint_files"]} == {ORDER[0]}
    assert row["auxiliary_download_bytes"] == 46830571
    pi = selected_asset_inventory([ORDER[1]])
    assert pi["pi_tokenizer_required"] and not pi["auxiliary_files"]
    assert pi["checkpoint_download_bytes"] > row["checkpoint_download_bytes"]


@pytest.mark.parametrize("ids", [[], [ORDER[1], ORDER[0]], [ORDER[0], ORDER[0]], ["unknown"], ORDER[0]])
def test_selection_must_be_exact_ordered_subset(ids):
    with pytest.raises(ValueError):
        selected_candidates(ids)


class Response(io.BytesIO):
    headers = {}
    def geturl(self):
        return "https://example.test/verified"


def test_streaming_download_checks_bytes_deadline_and_no_retry(tmp_path):
    content = b"abc" * 100
    row = {"relative_path": "model", "url": "https://example.test/model", "size_bytes": len(content),
           "sha256": "sha256:" + hashlib.sha256(content).hexdigest()}
    calls = []
    def opener(*args, **kwargs):
        calls.append(args)
        return Response(content)
    result = _fetch(row, tmp_path / "model", deadline=time.time() + 30, opener=opener)
    assert result["size_bytes"] == len(content) and len(calls) == 1
    for label, data in (("oversize", content + b"x"), ("changed", content[:-1] + b"x"), ("short", content[:-1])):
        with pytest.raises(ValueError):
            _fetch(row, tmp_path / label, deadline=time.time() + 30, opener=lambda *a, **k: Response(data))
        assert not (tmp_path / label).exists()
        assert not (tmp_path / (label + ".partial")).exists()
    with pytest.raises(ValueError, match="deadline_exceeded"):
        _fetch(row, tmp_path / "expired", deadline=time.time() - 1, opener=opener)
    assert len(calls) == 1
