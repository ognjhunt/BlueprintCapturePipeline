"""Large review batches keep exact pixels and bounded owned-file retention."""
from io import BytesIO
import json
from types import SimpleNamespace

from PIL import Image
import pytest

from blueprint_pipeline import website_image_completion as review
from blueprint_pipeline.local_reconstruction_adapters import _sha256_file
from blueprint_pipeline.core.handoff_lease import HandoffLeaseError, bind_handoff_lease


class Files:
    def __init__(self):
        self.uploaded, self.deleted = [], []

    def upload(self, *, file, config):
        data = file.read()
        self.uploaded.append((data, config))
        name = f"files/review-{len(self.uploaded)}"
        return SimpleNamespace(name=name, uri="https://provider.test/" + name, expiration_time=None)

    def delete(self, *, name):
        self.deleted.append(name)


def case(tmp_path):
    path = tmp_path / "source.bin"
    image = Image.new("RGB", (80, 120), (75, 110, 130))
    data = BytesIO()
    image.save(data, format="JPEG", quality=93)
    path.write_bytes(data.getvalue())
    item = {"image_path": str(path), "image_digest": _sha256_file(path)}
    files = Files()
    types = SimpleNamespace(UploadFileConfig=lambda **kw: kw,
        Part=SimpleNamespace(from_uri=lambda **kw: {"file": kw}, from_bytes=lambda **kw: {"inline": kw}))
    options = dict(client=SimpleNamespace(files=files), types=types,
        pairs=[("original a", item), ("original b", item)], prompt="review both",
        output_root=tmp_path, binding_digest="sha256:" + "a" * 64)
    return path, files, options


def test_large_review_uploads_unique_exact_bytes_and_cleans_after_response(tmp_path, monkeypatch):
    path, files, options = case(tmp_path)
    monkeypatch.setattr(review, "INLINE_REVIEW_REQUEST_BYTES", 1)
    with review._review_contents(**options) as contents:
        assert len(contents) == 5
        assert contents[2] == contents[4]
        assert files.uploaded[0][0] == path.read_bytes()
        assert files.uploaded[0][1]["mime_type"] == "image/jpeg"
        assert files.deleted == []
    assert len(files.uploaded) == len(files.deleted) == 1
    receipt = json.loads(next(tmp_path.glob("review-media-*.json")).read_text())
    assert receipt["cleanup_status"] == "deleted" and receipt["model_response_returned"] is True
    assert receipt["images"][0]["source_digest"] == _sha256_file(path)


def test_small_review_remains_inline_with_the_real_mime(tmp_path):
    path, files, options = case(tmp_path)
    with review._review_contents(**options) as contents:
        assert contents[2]["inline"] == {"data": path.read_bytes(), "mime_type": "image/jpeg"}
    assert files.uploaded == files.deleted == []


def test_lost_inference_answer_retains_media_and_receipt(tmp_path, monkeypatch):
    _, files, options = case(tmp_path)
    monkeypatch.setattr(review, "INLINE_REVIEW_REQUEST_BYTES", 1)
    with pytest.raises(TimeoutError):
        with review._review_contents(**options):
            raise TimeoutError("unknown model-call outcome")
    assert len(files.uploaded) == 1 and files.deleted == []
    receipt = json.loads(next(tmp_path.glob("review-media-*.json")).read_text())
    assert receipt["cleanup_status"] == "retained_after_uncertain_request"
    assert receipt["model_response_returned"] is False


def test_changed_later_source_cleans_uploads_without_sending_inference(tmp_path, monkeypatch):
    _, files, options = case(tmp_path)
    monkeypatch.setattr(review, "INLINE_REVIEW_REQUEST_BYTES", 1)
    options["pairs"][1] = ("prepared", {**options["pairs"][1][1], "image_digest": "sha256:" + "b" * 64})
    with pytest.raises(ValueError, match="review_source_changed"):
        with review._review_contents(**options):
            pytest.fail("inference must never start")
    assert len(files.uploaded) == len(files.deleted) == 1


def test_cleanup_failure_is_retained_without_losing_the_returned_review(tmp_path, monkeypatch):
    _, files, options = case(tmp_path)
    monkeypatch.setattr(review, "INLINE_REVIEW_REQUEST_BYTES", 1)
    def refuse_delete(**_kw):
        raise TimeoutError("cleanup unavailable")
    files.delete = refuse_delete
    with review._review_contents(**options):
        pass
    receipt = json.loads(next(tmp_path.glob("review-media-*.json")).read_text())
    assert receipt["model_response_returned"] is True and receipt["cleanup_status"] == "incomplete"
    assert receipt["cleanup_errors"] == [{"name": "files/review-1", "error_type": "TimeoutError"}]


@pytest.mark.parametrize("already_lost", [False, True])
def test_lease_loss_stops_next_media_upload_and_cleans_only_already_owned_files(tmp_path, monkeypatch, already_lost):
    _, files, options = case(tmp_path)
    second = tmp_path / "second.png"
    Image.new("RGB", (80, 120), "red").save(second)
    options["pairs"][1] = ("prepared", {"image_path": str(second), "image_digest": _sha256_file(second)})
    monkeypatch.setattr(review, "INLINE_REVIEW_REQUEST_BYTES", 1)
    lost = already_lost
    upload = files.upload

    def check(boundary):
        if lost:
            raise HandoffLeaseError("lost", boundary)

    def upload_then_lose(**kwargs):
        nonlocal lost
        result = upload(**kwargs)
        lost = True
        return result

    files.upload = upload_then_lose
    with bind_handoff_lease(check), pytest.raises(HandoffLeaseError):
        with review._review_contents(**options):
            pytest.fail("a lost worker must not reach inference")
    assert len(files.uploaded) == (0 if already_lost else 1)
    assert files.deleted == ([] if already_lost else ["files/review-1"])
