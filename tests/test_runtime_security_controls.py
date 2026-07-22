from __future__ import annotations

import io
import json
import stat
import tarfile
import zipfile
from pathlib import Path
from types import MethodType

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from blueprint_pipeline import (
    privacy_runner_service,
    pubsub_handoff_listener as phl,
    robot_eval_execution as ree,
    robot_eval_job_orchestrator as rejo,
    robot_eval_worker as rew,
    video_to_world_runner_service,
)
from blueprint_pipeline.common import PipelineError
from blueprint_pipeline.core import security_controls as security
from blueprint_pipeline.native_runtime_backend import (
    NativeRuntimeConfig,
    NativeWorldModelRuntimeStore,
)
from blueprint_pipeline.runtime_service_app import (
    create_runtime_app,
    validate_runtime_service_exposure,
)


def _site_world_payload(site_world_id: str = "siteworld-1") -> dict[str, object]:
    return {
        "spec": {
            "scene_id": "scene-1",
            "capture_id": "capture-1",
            "runtime_eligibility": {"default_backend": "native_world_model"},
        },
        "registration": {
            "site_world_id": site_world_id,
            "scene_id": "scene-1",
            "capture_id": "capture-1",
        },
        "health": {"site_world_id": site_world_id, "launchable": True, "blockers": []},
    }


def _runtime_store(tmp_path: Path) -> NativeWorldModelRuntimeStore:
    return NativeWorldModelRuntimeStore(
        NativeRuntimeConfig(
            root_dir=tmp_path / "runtime",
            base_url="http://runtime.test",
            ws_base_url="ws://runtime.test",
        )
    )


def test_runtime_api_requires_auth_enforces_tenants_and_rejects_unsafe_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _runtime_store(tmp_path)
    monkeypatch.setattr(store, "_ensure_cosmos_frames", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(store, "_ensure_initial_rollout_chunk", lambda *_args, **_kwargs: None)
    app = create_runtime_app(
        backend=store,
        title="secure-runtime",
        auth_tokens={"token-a": "tenant-a", "token-b": "tenant-b"},
        require_auth=True,
    )
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
        assert client.get("/v1/runtime").status_code == 401
        assert client.post("/v1/site-worlds", json=_site_world_payload()).status_code == 401

        tenant_a = {"Authorization": "Bearer token-a"}
        tenant_b = {"Authorization": "Bearer token-b"}
        registration = client.post(
            "/v1/site-worlds",
            json=_site_world_payload(),
            headers=tenant_a,
        )
        assert registration.status_code == 200
        assert client.get("/v1/site-worlds/siteworld-1", headers=tenant_b).status_code == 403
        overwrite = client.post(
            "/v1/site-worlds",
            json=_site_world_payload(),
            headers=tenant_b,
        )
        assert overwrite.status_code == 403
        assert client.get("/v1/site-worlds/siteworld-1", headers=tenant_a).status_code == 200

        unsafe = client.post(
            "/v1/site-worlds/siteworld-1/sessions",
            json={
                "robot_profile_id": "robot-1",
                "task_id": "task-1",
                "scenario_id": "scenario-1",
                "start_state_id": "start-1",
                "unsafe_allow_blocked_site_world": True,
            },
            headers=tenant_a,
        )
        assert unsafe.status_code == 422

        session = client.post(
            "/v1/site-worlds/siteworld-1/sessions",
            json={
                "session_id": "session-1",
                "robot_profile_id": "robot-1",
                "task_id": "task-1",
                "scenario_id": "scenario-1",
                "start_state_id": "start-1",
            },
            headers=tenant_a,
        )
        assert session.status_code == 200
        duplicate = client.post(
            "/v1/site-worlds/siteworld-1/sessions",
            json={
                "session_id": "session-1",
                "robot_profile_id": "robot-1",
                "task_id": "task-1",
                "scenario_id": "scenario-1",
                "start_state_id": "start-1",
            },
            headers=tenant_a,
        )
        assert duplicate.status_code == 409
        assert client.get("/v1/sessions/session-1/state", headers=tenant_b).status_code == 403
        assert client.get("/v1/sessions/session-1/state", headers=tenant_a).status_code == 200
        with pytest.raises(WebSocketDisconnect) as anonymous_ws:
            with client.websocket_connect("/v1/sessions/session-1/stream"):
                pass
        assert anonymous_ws.value.code == 4401
        with pytest.raises(WebSocketDisconnect) as cross_tenant_ws:
            with client.websocket_connect(
                "/v1/sessions/session-1/stream",
                headers=tenant_b,
            ):
                pass
        assert cross_tenant_ws.value.code == 4403
        with client.websocket_connect(
            "/v1/sessions/session-1/stream",
            headers=tenant_a,
        ) as websocket:
            assert websocket.receive_json()["type"] == "state"


def test_runtime_backend_rejects_traversal_and_exposed_blank_auth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("BLUEPRINT_RUNTIME_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("BLUEPRINT_RUNTIME_AUTH_TOKENS_JSON", raising=False)
    monkeypatch.delenv("SITE_WORLD_RUNTIME_SERVICE_API_KEY", raising=False)
    store = _runtime_store(tmp_path)
    with pytest.raises(ValueError, match="site_world_id"):
        store.register_site_world_package(
            spec={},
            registration={"site_world_id": "../escape"},
            health={},
        )
    assert not (tmp_path / "escape").exists()

    store.register_site_world_package(
        spec={"scene_id": "scene-1", "capture_id": "capture-1"},
        registration={"site_world_id": "blocked-world"},
        health={"launchable": False, "blockers": ["not_ready"]},
    )
    with pytest.raises(RuntimeError, match="site world is blocked"):
        store.create_session(
            "blocked-world",
            session_id="blocked-session",
            unsafe_allow_blocked_site_world=True,
        )

    with pytest.raises(RuntimeError, match="requires BLUEPRINT_RUNTIME_AUTH_TOKEN"):
        validate_runtime_service_exposure(host="0.0.0.0")

    app = create_runtime_app(
        backend=store,
        title="missing-auth-runtime",
        auth_tokens={},
        require_auth=True,
    )
    with pytest.raises(RuntimeError, match="authentication is required"):
        with TestClient(app):
            pass


class _Blob:
    def __init__(self, name: str, payload: bytes = b"x") -> None:
        self.name = name
        self.payload = payload

    def download_to_filename(self, filename: str) -> None:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)


class _StorageClient:
    def __init__(self, blobs: list[_Blob]) -> None:
        self.blobs = blobs

    def list_blobs(self, _bucket: str, *, prefix: str):
        return [blob for blob in self.blobs if blob.name.startswith(prefix)]


def _handoff() -> phl.HandoffMessage:
    return phl.HandoffMessage(
        bucket="capture-bucket",
        scene_id="scene-1",
        capture_id="capture-1",
        raw_prefix_uri="gs://capture-bucket/scenes/scene-1/captures/capture-1/raw",
        pipeline_handoff_uri=None,
    )


def test_pubsub_staging_rejects_identity_blob_and_request_path_traversal(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="Invalid Pub/Sub handoff identity"):
        phl.parse_handoff_payload(
            {
                "bucket": "capture-bucket",
                "scene_id": "../escape",
                "capture_id": "capture-1",
                "raw_prefix_uri": "gs://capture-bucket/scenes/../escape/captures/capture-1/raw",
            }
        )

    prefix = _handoff().capture_prefix
    client = _StorageClient([_Blob(f"{prefix}/../../outside.txt")])
    with pytest.raises(PipelineError, match="unsafe path"):
        phl.stage_handoff_capture(
            _handoff(),
            storage_root=tmp_path,
            storage_client=client,  # type: ignore[arg-type]
        )
    assert not (tmp_path / "capture-bucket" / "scenes" / "outside.txt").exists()

    capture_root = tmp_path / "capture-bucket" / prefix
    capture_root.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises(PipelineError, match="may not be absolute"):
        phl._resolve_staged_handoff_path(
            str(outside),
            handoff=_handoff(),
            capture_root=capture_root,
            storage_root=tmp_path,
            expect_directory=False,
        )
    with pytest.raises((PipelineError, security.SecurityValidationError), match="escapes"):
        phl._resolve_staged_handoff_path(
            "../../outside.json",
            handoff=_handoff(),
            capture_root=capture_root,
            storage_root=tmp_path,
            expect_directory=False,
        )
    outside_dir = tmp_path / "outside-dir"
    outside_dir.mkdir()
    (outside_dir / "request.json").write_text("{}", encoding="utf-8")
    (capture_root / "linked").symlink_to(outside_dir, target_is_directory=True)
    with pytest.raises((PipelineError, security.SecurityValidationError), match="escapes"):
        phl._resolve_staged_handoff_path(
            "linked/request.json",
            handoff=_handoff(),
            capture_root=capture_root,
            storage_root=tmp_path,
            expect_directory=False,
        )


def test_capture_archive_rejects_file_sources_links_bombs_and_member_floods(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "bundle.zip"
    with zipfile.ZipFile(local, "w") as archive:
        archive.writestr("capture_descriptor.json", "{}")
    with pytest.raises(ValueError, match="file://"):
        rew._extract_capture_root_bundle(local.as_uri(), tmp_path / "file-uri")
    monkeypatch.setenv("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME", "true")
    with pytest.raises(ValueError, match="local capture bundle"):
        rew._extract_capture_root_bundle(str(local), tmp_path / "provider-local")
    monkeypatch.delenv("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME")
    with pytest.raises(security.SecurityValidationError, match="not public"):
        rew._uri_to_local_file(
            "https://127.0.0.1/capture.zip",
            tmp_path / "private-source",
            filename="capture.zip",
        )

    remote_manifest = {
        "schema_version": "robot_eval_worker_manifest.v1",
        "job_id": "remote-local-root",
        "provisioner": "runpod",
        "simulator": "fixture",
        "capture_root": str(tmp_path),
        "job_request": {"job_id": "remote-local-root", "capture_root": str(tmp_path)},
    }

    def fetch_manifest(_url: str, *, output_path: Path, **_kwargs):  # type: ignore[no-untyped-def]
        output_path.write_text(json.dumps(remote_manifest), encoding="utf-8")
        return security.BoundedHttpResponse(
            body=b"",
            status=200,
            content_type="application/json",
            final_url="https://storage.example/worker.json",
        )

    monkeypatch.setenv("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME", "true")
    monkeypatch.setenv(
        "BLUEPRINT_WORKER_ALLOWED_DOWNLOAD_ORIGINS",
        "https://storage.example",
    )
    monkeypatch.setattr(rew, "fetch_bounded_https", fetch_manifest)
    blocked = rew.run_robot_eval_worker(
        manifest_uri="https://storage.example/worker.json",
        work_dir=tmp_path / "remote-local-root",
    )
    assert blocked["blockers"] == ["provider_runtime_capture_root_bundle_required"]
    monkeypatch.delenv("BLUEPRINT_ROBOT_EVAL_PROVIDER_RUNTIME")

    symlink_zip = tmp_path / "symlink.zip"
    with zipfile.ZipFile(symlink_zip, "w") as archive:
        info = zipfile.ZipInfo("capture_descriptor.json")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, "../../outside")
    with pytest.raises(ValueError, match="links or special"):
        rew._extract_capture_root_bundle(str(symlink_zip), tmp_path / "zip-link")

    symlink_tar = tmp_path / "symlink.tar"
    with tarfile.open(symlink_tar, "w") as archive:
        info = tarfile.TarInfo("capture_descriptor.json")
        info.type = tarfile.SYMTYPE
        info.linkname = "../../outside"
        archive.addfile(info)
    with pytest.raises(ValueError, match="links or special"):
        rew._extract_capture_root_bundle(str(symlink_tar), tmp_path / "tar-link")

    bomb = tmp_path / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("capture_descriptor.json", b"0" * (2 * 1024 * 1024))
    with pytest.raises(ValueError, match="compression ratio"):
        rew._extract_capture_root_bundle(str(bomb), tmp_path / "zip-bomb")

    flood = tmp_path / "flood.zip"
    with zipfile.ZipFile(flood, "w") as archive:
        archive.writestr("capture_descriptor.json", "{}")
        archive.writestr("extra.json", "{}")
    monkeypatch.setattr(rew, "CAPTURE_ARCHIVE_MAX_MEMBERS", 1)
    with pytest.raises(ValueError, match="member count"):
        rew._extract_capture_root_bundle(str(flood), tmp_path / "member-flood")


class _FakePeerSocket:
    def __init__(self, peer_ip: str = "93.184.216.34") -> None:
        self.peer_ip = peer_ip
        self.timeouts: list[float] = []

    def getpeername(self) -> tuple[str, int]:
        return self.peer_ip, 443

    def settimeout(self, timeout: float) -> None:
        self.timeouts.append(timeout)


class _FakePinnedResponse:
    reason = "OK"

    def __init__(
        self,
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes | Exception] | None = None,
    ) -> None:
        self.status = status
        self.headers = _Headers(headers or {"Content-Type": "application/json"})
        self._chunks = iter(chunks or [b"{}", b""])
        self.closed = False

    def read(self, _size: int = -1) -> bytes:
        item = next(self._chunks, b"")
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.closed = True


class _FakePinnedConnection:
    def __init__(
        self,
        response: _FakePinnedResponse,
        *,
        peer_ip: str = "93.184.216.34",
    ) -> None:
        self.response = response
        self.sock = _FakePeerSocket(peer_ip)
        self.requests: list[dict[str, object]] = []
        self.closed = False

    def connect(self) -> None:
        return None

    def request(
        self,
        method: str,
        target: str,
        *,
        body: bytes | None,
        headers: dict[str, str],
        encode_chunked: bool,
    ) -> None:
        self.requests.append(
            {
                "method": method,
                "target": target,
                "body": body,
                "headers": dict(headers),
                "encode_chunked": encode_chunked,
            }
        )

    def getresponse(self) -> _FakePinnedResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def _install_fake_connections(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[_FakePinnedResponse],
) -> list[_FakePinnedConnection]:
    pending = iter(responses)
    connections: list[_FakePinnedConnection] = []

    def open_connection(**_kwargs: object) -> _FakePinnedConnection:
        connection = _FakePinnedConnection(next(pending))
        connections.append(connection)
        return connection

    monkeypatch.setattr(security, "_open_pinned_connection", open_connection)
    return connections


def test_policy_endpoint_blocks_ssrf_oversize_redirect_and_http(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("BLUEPRINT_POLICY_ENDPOINT_ALLOWED_ORIGINS", "https://127.0.0.1")
    status, payload, detail = ree._call_policy_api(
        endpoint="https://127.0.0.1/latest/meta-data",
        observation_manifest={"observations": []},
        timeout_seconds=1,
    )
    assert status == "failed"
    assert payload is None
    assert detail["blockers"] == ["policy_api_call_failed"]

    modality_status, missing = rejo._validate_policy_modality(
        modality="policy_api_endpoint",
        payload={"endpoint_url": "http://policy.example/run"},
    )
    assert modality_status == "blocked"
    assert missing == ["policy_package.policy_api_endpoint.endpoint_url"]

    monkeypatch.setattr(
        security,
        "resolve_public_ips",
        lambda *_args, **_kwargs: ("93.184.216.34",),
    )
    _install_fake_connections(
        monkeypatch,
        [
            _FakePinnedResponse(
                headers={"Content-Type": "application/json", "Content-Length": "101"}
            )
        ],
    )
    with pytest.raises(security.SecurityValidationError, match="byte limit"):
        security.fetch_bounded_https(
            "https://policy.example/run",
            max_bytes=100,
            allowed_origins=("https://policy.example",),
            allowed_content_types=("application/json",),
        )

    _install_fake_connections(
        monkeypatch,
        [
            _FakePinnedResponse(
                status=302,
                headers={"Location": "https://169.254.169.254/latest/meta-data"},
                chunks=[b""],
            )
        ],
    )
    with pytest.raises(security.SecurityValidationError, match="origin is not approved|not public"):
        security.fetch_bounded_https(
            "https://policy.example/run",
            max_bytes=100,
            allowed_origins=("https://policy.example",),
            allowed_content_types=("application/json",),
        )

    target = tmp_path / "interrupted-download.json"
    _install_fake_connections(
        monkeypatch,
        [
            _FakePinnedResponse(
                chunks=[b"partial", OSError("stream interrupted")],
            )
        ],
    )
    with pytest.raises(OSError, match="interrupted"):
        security.fetch_bounded_https(
            "https://policy.example/run",
            max_bytes=100,
            allowed_origins=("https://policy.example",),
            allowed_content_types=("application/json",),
            output_path=target,
        )
    assert not target.exists()



def test_dns_rebinding_public_private_public_never_connects_or_requests_private(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    answers = iter(
        [
            ("93.184.216.34",),
            ("10.0.0.7",),
            ("93.184.216.34",),
        ]
    )
    resolver_calls: list[tuple[str, int]] = []

    def resolving(
        host: str,
        port: int,
        *,
        timeout_seconds: float | None = None,
    ) -> tuple[str, ...]:
        assert timeout_seconds is not None and timeout_seconds > 0
        resolver_calls.append((host, port))
        return next(answers)

    connections: list[tuple[str, _FakePinnedConnection]] = []

    def connection_for_ip(
        *,
        validated: security.ValidatedRemoteUrl,
        scheme: str,
        connect_ip: str,
        timeout: float,
    ) -> _FakePinnedConnection:
        assert validated.host == "policy.example"
        assert scheme == "https"
        assert timeout > 0
        connection = _FakePinnedConnection(
            _FakePinnedResponse(
                headers={"Content-Type": "application/json", "Content-Length": "2"}
            ),
            peer_ip=connect_ip,
        )
        connections.append((connect_ip, connection))
        return connection

    monkeypatch.setattr(security, "resolve_public_ips", resolving)
    monkeypatch.setattr(security, "_connection_for_ip", connection_for_ip)

    response = security.fetch_bounded_https(
        "https://policy.example:8443/run?attempt=1",
        method="POST",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        max_bytes=100,
        allowed_origins=("https://policy.example:8443",),
        allowed_content_types=("application/json",),
    )

    assert response.body == b"{}"
    # Only the public answer captured during validation is used. The hostile
    # private answer that a second hostname resolution would return is never
    # consumed, connected to, or sent a request.
    assert resolver_calls == [("policy.example", 8443)]
    assert [connect_ip for connect_ip, _connection in connections] == [
        "93.184.216.34"
    ]
    assert connections[0][1].requests == [
        {
            "method": "POST",
            "target": "/run?attempt=1",
            "body": b"{}",
            "headers": {
                "Content-Type": "application/json",
                "Host": "policy.example:8443",
            },
            "encode_chunked": False,
        }
    ]


def test_pinned_https_uses_numeric_transport_and_original_hostname_for_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RawSocket:
        closed = False

        def close(self) -> None:
            self.closed = True

    class WrappedSocket(_FakePeerSocket):
        def close(self) -> None:
            return None

    raw_socket = RawSocket()
    numeric_calls: list[tuple[str, int, float]] = []
    sni_calls: list[tuple[object, str]] = []

    def numeric_socket(connect_ip: str, port: int, timeout: float) -> RawSocket:
        numeric_calls.append((connect_ip, port, timeout))
        return raw_socket

    class Context:
        minimum_version = security.ssl.TLSVersion.TLSv1

        def wrap_socket(self, sock: object, *, server_hostname: str) -> WrappedSocket:
            sni_calls.append((sock, server_hostname))
            return WrappedSocket()

    monkeypatch.setattr(security, "_numeric_socket", numeric_socket)
    connection = security._PinnedHTTPSConnection(
        "policy.example",
        port=8443,
        connect_ip="93.184.216.34",
        timeout=5.0,
        context=Context(),  # type: ignore[arg-type]
    )

    connection.connect()

    assert numeric_calls == [("93.184.216.34", 8443, 5.0)]
    assert sni_calls == [(raw_socket, "policy.example")]
    assert connection._tls_context.minimum_version == security.ssl.TLSVersion.TLSv1_2


def test_pinned_connection_rejects_unexpected_private_peer_before_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempted = _FakePinnedConnection(
        _FakePinnedResponse(),
        peer_ip="10.0.0.7",
    )

    monkeypatch.setattr(
        security,
        "_connection_for_ip",
        lambda **_kwargs: attempted,
    )
    validated = security.ValidatedRemoteUrl(
        url="https://policy.example/run",
        origin="https://policy.example",
        host="policy.example",
        port=443,
        resolved_ips=("93.184.216.34",),
    )

    with pytest.raises(security.SecurityValidationError, match="pinned IP"):
        security._open_pinned_connection(
            validated=validated,
            scheme="https",
            deadline=security.time.monotonic() + 5,
        )

    assert attempted.requests == []
    assert attempted.closed is True


def test_bounded_dns_resolution_fails_closed_without_live_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = security.threading.Event()
    release = security.threading.Event()
    finished = security.threading.Event()

    def blocked_resolver(*_args: object, **_kwargs: object) -> list[object]:
        started.set()
        release.wait(timeout=1)
        finished.set()
        return []

    monkeypatch.setattr(security.socket, "getaddrinfo", blocked_resolver)
    try:
        with pytest.raises(
            security.SecurityValidationError,
            match="DNS resolution exceeded total time limit",
        ):
            security._bounded_getaddrinfo(
                "policy.example",
                443,
                timeout_seconds=0.05,
            )
        assert started.is_set()
    finally:
        release.set()
    assert finished.wait(timeout=1)


def test_bounded_service_url_allows_only_loopback_plain_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_connections(
        monkeypatch,
        [
            _FakePinnedResponse(
                headers={"Content-Type": "application/json", "Content-Length": "2"}
            )
        ],
    )
    response = security.fetch_bounded_service_url(
        "http://127.0.0.1:8765/health",
        max_bytes=100,
        allowed_origins=(),
        allowed_content_types=("application/json",),
    )
    assert response.body == b"{}"

    for unsafe_url in (
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.2/private",
        "file:///etc/passwd",
        "gopher://example.com/data",
    ):
        with pytest.raises(security.SecurityValidationError):
            security.fetch_bounded_service_url(
                unsafe_url,
                max_bytes=100,
                allowed_origins=(),
            )


class _Headers(dict[str, str]):
    def get(self, key: str, default: str | None = None) -> str | None:
        return super().get(key, default)


def _runner_handler(handler_cls, *, token: str, length: int):
    handler = object.__new__(handler_cls)
    handler.path = "/run"
    handler.headers = _Headers(
        {"Authorization": f"Bearer {token}", "Content-Length": str(length)}
    )
    handler.rfile = io.BytesIO(b"{}")
    handler.wfile = io.BytesIO()
    response: dict[str, object] = {}

    def send_response(self, status: int) -> None:  # type: ignore[no-untyped-def]
        response["status"] = status

    handler.send_response = MethodType(send_response, handler)
    handler.send_header = MethodType(lambda *_args: None, handler)
    handler.end_headers = MethodType(lambda *_args: None, handler)
    return handler, response


@pytest.mark.parametrize(
    ("service", "token_env", "handler_cls"),
    [
        (privacy_runner_service, "PRIVACY_RUNNER_TOKEN", privacy_runner_service._Handler),
        (
            video_to_world_runner_service,
            "VIDEO_TO_WORLD_RUNNER_TOKEN",
            video_to_world_runner_service._Handler,
        ),
    ],
)
def test_runner_services_fail_closed_without_token_and_bound_requests(
    service,
    token_env: str,
    handler_cls,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(token_env, raising=False)
    monkeypatch.delenv("PRIVACY_RUNNER_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="TOKEN must be nonempty"):
        service.main()

    monkeypatch.setenv(token_env, "secret")
    handler, response = _runner_handler(handler_cls, token="secret", length=65 * 1024 * 1024)
    handler.do_POST()
    assert response["status"] == 413
