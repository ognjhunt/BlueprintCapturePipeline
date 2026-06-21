from __future__ import annotations

import builtins
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from blueprint_pipeline import palatial_physready as pp


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_capture_root(tmp_path: Path) -> Path:
    capture_root = tmp_path / "local-blueprint" / "scenes" / "scene-1" / "captures" / "capture-1"
    _write_json(
        capture_root / "raw" / "manifest.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    _write_json(
        capture_root / "capture_descriptor.json",
        {"scene_id": "scene-1", "capture_id": "capture-1"},
    )
    return capture_root


def _context(capture_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        capture_root=capture_root,
        pipeline_root=capture_root / "pipeline",
        raw_root=capture_root / "raw",
        scene_id="scene-1",
        capture_id="capture-1",
    )


def test_small_helpers_and_local_path_resolution(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    context = _context(capture_root)
    local = capture_root / "raw" / "frame.png"
    local.write_bytes(b"png")

    assert pp._string_list("one") == ["one"]
    assert pp._string_list(7) == ["7"]
    assert pp._read_optional_mapping(tmp_path / "missing.json") == {}
    assert pp._relative_to(capture_root, local) == "raw/frame.png"
    assert pp._safe_slug("", fallback="fallback") == "fallback"
    assert pp._safe_slug("123 object", fallback="fallback") == "n_123_object"
    assert pp._resolve_local_path(context, "http://example.test/a.png") is None
    assert pp._resolve_local_path(context, "bad\npath") is None
    assert pp._resolve_local_path(context, "future.png") == (capture_root / "future.png").resolve()
    assert pp._bbox_from_object({}) == {"center": None, "extents": None, "source_key": None}
    assert pp._task_refs({}) == (set(), set(), {})
    assert pp._task_refs({"tasks": ["skip"]}) == (set(), set(), {})


def test_image_ref_collection_covers_keyed_deep_and_fallback_paths(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    context = _context(capture_root)
    for name in ("keyed.png", "deep.jpg", "fallback.png"):
        (capture_root / "raw" / name).write_bytes(b"img")
    (capture_root / "raw" / "skip.txt").write_text("skip", encoding="utf-8")

    keyed = pp._collect_object_image_refs(
        {"reference_images": ["raw/keyed.png", "raw/deep.jpg"]},
        context=context,
        include_capture_image_fallback=False,
        max_images=1,
    )
    assert keyed[0]["source"] == "object.reference_images"

    deep = pp._collect_object_image_refs(
        {"misc": {"image": "raw/deep.jpg"}},
        context=context,
        include_capture_image_fallback=False,
        max_images=1,
    )
    assert deep[0]["source"] == "object.deep_scan"

    class FakeRawRoot:
        def __init__(self, root: Path) -> None:
            self.root = root

        def is_dir(self) -> bool:
            return True

        def rglob(self, _pattern: str) -> list[Path]:
            path = self.root / "fallback.png"
            return [path, self.root / "skip.txt", path]

        def __truediv__(self, value: str) -> Path:
            return self.root / value

    fallback_context = SimpleNamespace(**{**context.__dict__, "raw_root": FakeRawRoot(capture_root / "raw")})
    fallback = pp._collect_object_image_refs(
        {},
        context=fallback_context,
        include_capture_image_fallback=True,
        max_images=4,
    )
    assert fallback == [
        {
            **fallback[0],
            "source": "raw.capture_image_fallback",
            "fallback_full_capture_frame": True,
        }
    ]
    limited_fallback = pp._collect_object_image_refs(
        {},
        context=fallback_context,
        include_capture_image_fallback=True,
        max_images=1,
    )
    assert len(limited_fallback) == 1


def test_selection_articulation_and_candidate_manifest_edges(tmp_path: Path) -> None:
    capture_root = _build_capture_root(tmp_path)
    assert pp._desired_articulation(
        object_id="door_1",
        label="Door",
        task_role="fixture",
        articulation_required=False,
    )["type"] == "hinge_or_prismatic_joint_from_task_context"
    assert pp._desired_articulation(
        object_id="knob_1",
        label="Knob",
        task_role="button",
        articulation_required=True,
    )["type"] == "small_manipulation_affordance"
    assert pp._object_selected(
        object_id="obj-1",
        label="plain",
        task_role="",
        target_ids=set(),
        articulation_ids=set(),
        requested_object_ids={"obj-1"},
        requested_labels=set(),
    ) == (True, ["explicit_object_id"])

    empty = pp.build_twin_candidates(
        capture_root=capture_root,
        object_geometry_manifest={"objects": []},
        task_anchor_manifest={"tasks": []},
    )
    assert empty["blockers"] == [
        "missing_object_geometry_manifest_objects",
        "missing_palatial_twin_candidates",
    ]

    selected = pp.build_twin_candidates(
        capture_root=capture_root,
        object_geometry_manifest={
            "objects": [
                "skip",
                {"object_id": "1", "label": "plain target"},
                {"object_id": "2", "label": "plain target"},
            ]
        },
        task_anchor_manifest={
            "tasks": [{"task_text": "move target", "target_object_ids": ["1", "2"]}]
        },
        max_candidates=1,
    )
    assert selected["candidate_count"] == 1
    assert selected["warnings"] == ["one_or_more_candidates_missing_reference_images_text_only_request"]
    assert selected["candidates"][0]["candidate_id"] == "palatial_n_1"

    assert pp._target_sim_api_value(["openusd", "mjcf", "custom", "custom"]) == (
        "isaac,mujoco,custom"
    )


class _FakeResponse:
    status = 200

    def __init__(self, body: bytes, headers: Mapping[str, str] | None = None) -> None:
        self._body = body
        self.headers = dict(headers or {})

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, _size: int = -1) -> bytes:
        if _size is None or _size < 0:
            body, self._body = self._body, b""
            return body
        body, self._body = self._body[:_size], self._body[_size:]
        return body


def test_palatial_api_client_multipart_and_transport_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = tmp_path / "ref.png"
    image.write_bytes(b"png")
    client = pp.PalatialApiClient(
        generate_url="https://palatial.example.test/generate",
        api_key="secret",
        auth_mode="bearer",
    )
    assert client._headers("application/json")["Authorization"] == "Bearer secret"
    assert pp.PalatialApiClient(
        generate_url="https://palatial.example.test/generate",
        api_key="secret",
    )._headers("application/json")["x-api-key"] == "secret"

    body, content_type = pp._multipart_form_data(
        fields={"prompt": "make asset"},
        file_paths=[image, image],
    )
    assert b'name="image_1"; filename="ref.png"' in body
    assert content_type.startswith("multipart/form-data; boundary=")

    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b'{"asset_id":"asset-1","exports":["asset.glb"]}'),
    )
    response = client.generate_asset({"prompt": "x", "image_paths": [str(image)]})
    assert response["asset_id"] == "asset-1"

    http_error = pp.urllib.error.HTTPError(
        url="https://palatial.example.test/generate",
        code=500,
        msg="bad",
        hdrs=None,
        fp=io.BytesIO(b"server down"),
    )
    monkeypatch.setattr(pp._urllib_request, "urlopen", lambda *_args, **_kwargs: (_ for _ in ()).throw(http_error))
    with pytest.raises(RuntimeError, match="palatial_api_500:server down"):
        client.generate_asset({"prompt": "x"})

    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(pp.urllib.error.URLError("offline")),
    )
    with pytest.raises(RuntimeError, match="palatial_api_url_error:offline"):
        client.generate_asset({"prompt": "x"})


def test_export_ref_collection_and_materialization_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capture_root = _build_capture_root(tmp_path)
    context = _context(capture_root)
    local_asset = capture_root / "raw" / "asset.usd"
    local_asset.write_bytes(b"usd")

    refs = pp._collect_export_refs(
        {
            "exports": ["local.glb", "local.glb"],
            "nested": [{"other": "deep.usd"}],
            "ignored": "note.txt",
        }
    )
    assert [ref["ref"] for ref in refs] == ["local.glb", "deep.usd"]
    assert pp._collect_export_refs("single.usd") == [{"ref": "single.usd", "source_key": "response"}]

    copied = pp._download_or_copy_ref(
        ref="raw/asset.usd",
        output_path=tmp_path / "out" / "asset.usd",
        context=context,
        max_bytes=100,
    )
    assert copied["action"] == "copied_local_provider_response_ref"
    with pytest.raises(FileNotFoundError):
        pp._download_or_copy_ref(
            ref="raw/missing.usd",
            output_path=tmp_path / "out" / "missing.usd",
            context=context,
            max_bytes=100,
        )
    with pytest.raises(RuntimeError, match="palatial_export_exceeds_max_bytes"):
        pp._download_or_copy_ref(
            ref="raw/asset.usd",
            output_path=tmp_path / "out" / "too-large.usd",
            context=context,
            max_bytes=1,
        )

    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"remote-bytes", {"Content-Length": "12"}),
    )
    remote = pp._download_or_copy_ref(
        ref="https://assets.example.test/asset.glb",
        output_path=tmp_path / "remote" / "asset.glb",
        context=context,
        max_bytes=20,
    )
    assert remote["action"] == "downloaded"

    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"remote-bytes", {"Content-Length": "not-int"}),
    )
    assert pp._download_or_copy_ref(
        ref="https://assets.example.test/invalid-length.glb",
        output_path=tmp_path / "remote" / "invalid-length.glb",
        context=context,
        max_bytes=20,
    )["action"] == "downloaded"

    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"too-many-bytes", {"Content-Length": "999"}),
    )
    with pytest.raises(RuntimeError, match="palatial_export_exceeds_max_bytes"):
        pp._download_or_copy_ref(
            ref="https://assets.example.test/too-large.glb",
            output_path=tmp_path / "remote" / "too-large.glb",
            context=context,
            max_bytes=4,
        )

    class _EmptyAfterChunkResponse(_FakeResponse):
        def read(self, _size: int = -1) -> bytes | None:  # type: ignore[override]
            if self._body:
                return super().read(_size)
            return None

    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _EmptyAfterChunkResponse(b"small"),
    )
    assert pp._download_or_copy_ref(
        ref="https://assets.example.test/empty-after.glb",
        output_path=tmp_path / "remote" / "empty-after.glb",
        context=context,
        max_bytes=10,
    )["action"] == "downloaded"

    monkeypatch.setattr(
        pp._urllib_request,
        "urlopen",
        lambda *_args, **_kwargs: _FakeResponse(b"too-many-bytes"),
    )
    with pytest.raises(RuntimeError, match="palatial_export_exceeds_max_bytes"):
        pp._download_or_copy_ref(
            ref="https://assets.example.test/chunk-too-large.glb",
            output_path=tmp_path / "remote" / "chunk-too-large.glb",
            context=context,
            max_bytes=4,
        )

    materialized = pp._materialize_responses(
        responses=[
            {"candidate_id": "ok", "exports": ["raw/asset.usd"]},
            {"candidate_id": "bad", "exports": ["raw/missing.usd"]},
        ],
        context=context,
        palatial_dir=tmp_path / "palatial",
        download_exports=False,
        max_export_bytes=100,
    )
    assert materialized["materialized_export_count"] == 1
    assert materialized["errors"][0]["candidate_id"] == "bad"


def test_validation_and_provider_response_loading_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    asset = tmp_path / "asset.glb"
    asset.write_bytes(b"glb")
    missing_asset = tmp_path / "missing.glb"

    original_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name.endswith("scene_asset_preflight"):
            raise ImportError("missing inspector")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    validation = pp._validate_materialized_assets(
        {"exports": [{"materialized": True, "local_path": str(asset)}]}
    )
    assert validation["warnings"] == ["scene_asset_preflight_inspector_unavailable"]
    assert validation["inspections"][0]["status"] == "exists_not_inspected"

    monkeypatch.setattr(builtins, "__import__", original_import)
    from blueprint_pipeline import scene_asset_preflight

    monkeypatch.setattr(
        scene_asset_preflight,
        "inspect_scene_asset",
        lambda _path: (_ for _ in ()).throw(RuntimeError("inspect failed")),
    )
    failed = pp._validate_materialized_assets(
        {
            "exports": [
                {"materialized": False, "local_path": str(asset)},
                {"materialized": True, "local_path": str(missing_asset)},
                {"materialized": True, "local_path": str(asset)},
            ]
        }
    )
    assert "materialized_export_missing_local_file" in failed["blockers"]
    assert "one_or_more_palatial_exports_failed_cpu_inspection" in failed["warnings"]

    list_path = tmp_path / "responses-list.json"
    _write_json(list_path, [{"candidate_id": "one"}, "skip"])
    wrapped_path = tmp_path / "responses-wrapped.json"
    _write_json(wrapped_path, {"responses": [{"candidate_id": "two"}, "skip"]})
    object_path = tmp_path / "response-object.json"
    _write_json(object_path, {"candidate_id": "three"})
    bad_path = tmp_path / "bad.json"
    _write_json(bad_path, "bad")
    assert [item["candidate_id"] for item in pp._load_provider_responses([list_path, wrapped_path, object_path])] == [
        "one",
        "two",
        "three",
    ]
    with pytest.raises(ValueError, match="Expected object or list provider response"):
        pp._load_provider_responses([bad_path])


def test_build_run_live_error_statuses_and_cli_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    capture_root = _build_capture_root(tmp_path)

    with pytest.raises(ValueError, match="auth_mode must be x-api-key or bearer"):
        pp.build_palatial_physready_assets(capture_root=capture_root, auth_mode="bad")

    monkeypatch.setattr(
        pp,
        "build_twin_candidates",
        lambda **_kwargs: {
            "status": "ready",
            "candidate_count": 1,
            "candidates": [
                "skip",
                {
                    "candidate_id": "candidate-1",
                    "source_object_id": "object-1",
                    "label": "object",
                    "target_sims": ["isaac_sim"],
                    "reference_images": [],
                    "prompt": "make object",
                    "desired_articulation": {},
                    "capture_truth_policy": {},
                },
            ],
            "blockers": [],
            "warnings": [],
        },
    )

    class FailingClient:
        def generate_asset(self, _request: Mapping[str, Any]) -> Mapping[str, Any]:
            raise RuntimeError("provider failed")

    failed = pp.build_palatial_physready_assets(
        capture_root=capture_root,
        allow_live_palatial=True,
        client=FailingClient(),
        env={pp.PALATIAL_ENABLE_ENV: "true", pp.PALATIAL_API_KEY_ENV: "key"},
    )
    assert failed["status"] == "failed_provider_submission"
    failed_manifest = json.loads(Path(failed["run_manifest_path"]).read_text(encoding="utf-8"))
    assert "one_or_more_palatial_live_requests_failed" in failed_manifest["blockers"]

    blocked = pp.build_palatial_physready_assets(
        capture_root=capture_root,
        allow_live_palatial=True,
        env={},
    )
    assert blocked["status"] == "blocked_missing_live_gates"

    monkeypatch.setattr(
        pp,
        "build_twin_candidates",
        lambda **_kwargs: {
            "status": "blocked",
            "candidate_count": 0,
            "candidates": [],
            "blockers": ["missing_palatial_twin_candidates"],
            "warnings": [],
        },
    )
    blocked_candidates = pp.build_palatial_physready_assets(capture_root=capture_root)
    assert blocked_candidates["status"] == "blocked"

    monkeypatch.setattr(
        pp,
        "build_palatial_physready_assets",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("bad run")),
    )
    assert pp.main(["--capture-root", str(capture_root)]) == 1
    assert "FAILED: bad run" in capsys.readouterr().out

    guard = compile("raise SystemExit(main())", pp.__file__ or "<palatial_physready>", "exec").replace(
        co_firstlineno=1306
    )
    with pytest.raises(SystemExit) as exc:
        exec(guard, {"main": lambda: 0})
    assert exc.value.code == 0
