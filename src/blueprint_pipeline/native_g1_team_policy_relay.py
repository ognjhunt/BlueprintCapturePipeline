"""Private transport for an operator-bound, separately isolated G1 policy.

This module cannot allocate a provider or accept a policy launch command. The
caller owns image/asset admission, isolation, watchdogs, scoring and teardown.
Returned transport receipts contain no observations, credentials or addresses.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import stat
import time
from dataclasses import asdict, dataclass
from typing import Any, Callable

MAX_FRAME_BYTES = 2 * 1024 * 1024
PROTOCOL = "g1_private_policy_relay_v1"
_SHA = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SECRET = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True)
class RelayBinding:
    packet_digest: str
    profile_digest: str
    setup_digest: str
    delivery_mode: str

    def __post_init__(self) -> None:
        if (any(not isinstance(value, str) or not _SHA.fullmatch(value) for value in
                (self.packet_digest, self.profile_digest, self.setup_digest))
                or self.delivery_mode not in {"container", "noncontainer_artifact"}):
            raise ValueError("g1_relay_binding_invalid")


def _timeout(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 600:
        raise ValueError("g1_relay_timeout_invalid")
    return float(value)


def _secret(value: str) -> None:
    if not isinstance(value, str) or not _SECRET.fullmatch(value):
        raise ValueError("g1_relay_secret_invalid")


def _proof(secret: str, challenge: str, binding: RelayBinding) -> str:
    body = json.dumps({"challenge": challenge, "binding": asdict(binding)},
                      sort_keys=True, separators=(",", ":")).encode()
    return hmac.new(bytes.fromhex(secret), body, hashlib.sha256).hexdigest()


def _digest(value: dict[str, Any]) -> str:
    normalized = {key: item for key, item in value.items() if key != "receipt_digest"}
    body = json.dumps(normalized, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("g1_relay_duplicate_json_key")
        value[key] = item
    return value


def _nonfinite(_: str) -> None:
    raise ValueError("g1_relay_nonfinite_json")


class JsonlSocket:
    """Bound a complete frame to one deadline, including trickled partial data."""

    def __init__(self, sock: socket.socket, *, timeout_seconds: float) -> None:
        self.socket = sock
        self.timeout_seconds = _timeout(timeout_seconds)
        self._buffer = bytearray()

    def read(self) -> dict[str, Any]:
        deadline = time.monotonic() + self.timeout_seconds
        while b"\n" not in self._buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("g1_relay_read_timeout")
            self.socket.settimeout(remaining)
            block = self.socket.recv(min(65536, MAX_FRAME_BYTES + 1 - len(self._buffer)))
            if not block:
                if self._buffer:
                    raise ValueError("g1_relay_partial_eof")
                raise EOFError("g1_relay_peer_closed")
            self._buffer.extend(block)
            if len(self._buffer) > MAX_FRAME_BYTES:
                raise ValueError("g1_relay_frame_oversized")
        line, _, remainder = self._buffer.partition(b"\n")
        self._buffer = bytearray(remainder)
        try:
            value = json.loads(line, object_pairs_hook=_object, parse_constant=_nonfinite)
        except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("g1_relay_frame_invalid") from exc
        if not isinstance(value, dict):
            raise ValueError("g1_relay_frame_not_object")
        # JSON exponent overflow also must not become an infinity.
        try:
            _finite_json(value)
        except RecursionError as exc:
            raise ValueError("g1_relay_frame_invalid") from exc
        return value

    def write(self, value: dict[str, Any]) -> None:
        body = (json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if len(body) > MAX_FRAME_BYTES:
            raise ValueError("g1_relay_frame_oversized")
        self.socket.settimeout(self.timeout_seconds)
        self.socket.sendall(body)

    def close(self) -> None:
        self.socket.close()


def _finite_json(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("g1_relay_nonfinite_json")
    if isinstance(value, dict):
        for item in value.values():
            _finite_json(item)
    elif isinstance(value, list):
        for item in value:
            _finite_json(item)


def _private_path(path: Path) -> None:
    if (not isinstance(path, Path) or not path.is_absolute()
            or path.resolve() != path or path.is_symlink()):
        raise ValueError("g1_relay_path_invalid")
    parent = path.parent.stat()
    if not stat.S_ISDIR(parent.st_mode) or parent.st_uid != os.geteuid() or parent.st_mode & 0o077:
        raise ValueError("g1_relay_parent_not_private")


def validate_relay_conformance(value: Any, binding: RelayBinding) -> None:
    """Accept only the actual native synthetic receipt, without extra payloads."""
    fields = {"schema_version", "status", "profile_digest", "source_setup_digest",
              "robot_preset_id", "delivery_mode", "synthetic_policy_query_count",
              "returned_action_count", "site_observation_sent", "site_policy_query_count",
              "task_scored", "runtime_identity_verified", "rights_authorized",
              "paid_launch_authorized", "public_redistribution_authorized", "claim_ceiling",
              "receipt_digest"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema_version"] != "native_g1_team_policy_synthetic_conformance.v1"
            or value["status"] != "synthetic_wire_compatible"
            or value["profile_digest"] != binding.profile_digest
            or value["source_setup_digest"] != binding.setup_digest
            or value["delivery_mode"] != binding.delivery_mode
            or value["robot_preset_id"] != "unitree_g1_dex3_sonic_v1"
            or type(value["synthetic_policy_query_count"]) is not int
            or value["synthetic_policy_query_count"] != 1
            or type(value["returned_action_count"]) is not int
            or not 1 <= value["returned_action_count"] <= 64
            or type(value["site_policy_query_count"]) is not int or value["site_policy_query_count"] != 0
            or any(value[key] is not False for key in {"site_observation_sent", "task_scored",
                "runtime_identity_verified", "rights_authorized", "paid_launch_authorized",
                "public_redistribution_authorized"})
            or value["claim_ceiling"] != "planning_only"
            or value["receipt_digest"] != _digest(value)):
        raise ValueError("g1_relay_conformance_invalid")


def validate_relay_close(value: Any, binding: RelayBinding, *, conformance_digest: str,
                         linked_episode_digest: str | None) -> None:
    fields = {"schema_version", "status", "profile_digest", "delivery_mode",
              "synthetic_conformance_digest", "child_teardown_digest", "child_teardown_required",
              "linked_scored_episode_result_digest", "linked_episode_media_verified_by_session",
              "provider_teardown_verified", "claim_ceiling", "receipt_digest"}
    if (not isinstance(value, dict) or set(value) != fields
            or value["schema_version"] != "native_g1_team_runtime_session.v1"
            or value["status"] != "closed" or value["profile_digest"] != binding.profile_digest
            or value["delivery_mode"] != binding.delivery_mode
            or value["synthetic_conformance_digest"] != conformance_digest
            or value["child_teardown_required"] is not True
            or not isinstance(value["child_teardown_digest"], str)
            or not _SHA.fullmatch(value["child_teardown_digest"])
            or value["linked_scored_episode_result_digest"] != linked_episode_digest
            or value["linked_episode_media_verified_by_session"] is not False
            or value["provider_teardown_verified"] is not False
            or value["claim_ceiling"] != "planning_only"
            or value["receipt_digest"] != _digest(value)):
        raise ValueError("g1_relay_policy_close_unverified")


def _infer_payload(payload: dict[str, Any]) -> None:
    """Validate the exact permitted wire, without importing Isaac or tensor code."""
    try:
        observation = payload["observation"]
        image = observation["images"]["front"]
        state = observation["state"]
        valid = (
            set(payload) == {"observation", "robot_type", "return_chunk", "task"}
            and payload["robot_type"] == "unitree_g1_refpose_v3_1"
            and payload["return_chunk"] is True
            and isinstance(payload["task"], str) and 1 <= len(payload["task"].strip()) <= 512
            and set(observation) == {"images", "state"}
            and set(observation["images"]) == {"front"}
            and set(image) == {"shape", "dtype", "data_b64"}
            and image["shape"] == [480, 640, 3] and image["dtype"] == "uint8"
            and len(base64.b64decode(image["data_b64"], validate=True)) == 480 * 640 * 3
            and isinstance(state, list) and len(state) == 64
            and all(type(item) in (int, float) and math.isfinite(item) for item in state)
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError("g1_relay_infer_payload_invalid")


class G1PolicyRelayServer:
    """Serve one authenticated simulator; never retry or replace its policy."""

    def __init__(self, *, path: Path, binding: RelayBinding, secret: str) -> None:
        _private_path(path)
        _secret(secret)
        if path.exists():
            raise ValueError("g1_relay_path_exists")
        self.path, self.binding, self._secret = path, binding, secret
        parent = path.parent.stat()
        self._parent_inode = (parent.st_dev, parent.st_ino)
        self._inode, self._used = None, False
        self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._listener.bind(str(path))
            os.chmod(path, 0o600)
            info = path.lstat()
            self._inode = (info.st_dev, info.st_ino)
            self._listener.listen(1)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        self._listener.close()
        try:
            parent = self.path.parent.stat()
            if (parent.st_dev, parent.st_ino) != self._parent_inode or self.path.resolve() != self.path:
                return
            info = self.path.lstat()
            if stat.S_ISSOCK(info.st_mode) and (info.st_dev, info.st_ino) == self._inode:
                self.path.unlink()
        except FileNotFoundError:
            pass

    def serve_one(self, *, session_factory: Callable[[], Any], timeout_seconds: float) -> dict[str, Any]:
        timeout = _timeout(timeout_seconds)
        if self._used:
            raise ValueError("g1_relay_server_already_used")
        self._used = True
        session, wire, closure = None, None, None
        conformance_digest = None
        close_attempted = False
        count, index, inference_count = 0, 0, 0
        linked_digest = None
        status, failure = "failed", None
        try:
            self._listener.settimeout(timeout)
            peer, _ = self._listener.accept()
            self._listener.close()
            wire = JsonlSocket(peer, timeout_seconds=timeout)
            challenge = secrets.token_hex(32)
            wire.write({"protocol": PROTOCOL, "challenge": challenge})
            hello = wire.read()
            if (set(hello) != {"protocol", "binding", "proof"}
                    or hello["protocol"] != PROTOCOL or hello["binding"] != asdict(self.binding)
                    or not isinstance(hello["proof"], str)
                    or not hmac.compare_digest(hello["proof"], _proof(self._secret, challenge, self.binding))):
                raise ValueError("g1_relay_authentication_failed")
            session = session_factory()
            conformance = session.conformance
            if (session.profile_digest != self.binding.profile_digest
                    or session.delivery_mode != self.binding.delivery_mode
                    or session.client.profile_digest != self.binding.profile_digest):
                raise ValueError("g1_relay_session_binding_invalid")
            validate_relay_conformance(conformance, self.binding)
            conformance_digest = conformance["receipt_digest"]
            wire.write({"protocol": PROTOCOL, "status": "ready", "binding": asdict(self.binding),
                        "conformance": conformance})
            while True:
                request = wire.read()
                if (set(request) != {"protocol", "request_id", "kind", "payload"}
                        or request["protocol"] != PROTOCOL or type(request["request_id"]) is not int
                        or request["request_id"] != index or not isinstance(request["payload"], dict)):
                    raise ValueError("g1_relay_request_identity_invalid")
                kind, payload = request["kind"], request["payload"]
                if kind == "close" and payload == {}:
                    close_attempted = True
                    closure = self._close_session(session, linked_digest=linked_digest)
                    wire.write(self._reply(index, closure))
                    status = "policy_session_closed"
                    break
                if kind == "link_episode":
                    if (set(payload) != {"episode"} or not isinstance(payload["episode"], dict)
                            or linked_digest is not None or inference_count <= 0
                            or type(payload["episode"].get("policy_query_count")) is not int
                            or payload["episode"]["policy_query_count"] != inference_count):
                        raise ValueError("g1_relay_episode_query_binding_invalid")
                    session.link_scored_episode(payload["episode"])
                    linked_digest = payload["episode"]["result_digest"]
                    wire.write(self._reply(index, {"linked_scored_episode_result_digest": linked_digest}))
                    index += 1
                    continue
                if linked_digest is not None:
                    raise ValueError("g1_relay_episode_already_linked")
                if kind == "reset":
                    if set(payload) != {"seed"} or type(payload["seed"]) is not int or payload["seed"] < 0:
                        raise ValueError("g1_relay_reset_invalid")
                elif kind == "infer":
                    _infer_payload(payload)
                else:
                    raise ValueError("g1_relay_command_invalid")
                response = session.client._exchange(kind, payload)
                if kind == "reset" and response.get("ok") is not True:
                    raise ValueError("g1_relay_reset_ack_invalid")
                if kind == "reset":
                    inference_count = 0
                else:
                    inference_count += 1
                # Backend verified its own sequence/profile before rebinding.
                wire.write(self._reply(index, response))
                count += 1
                index += 1
        except Exception as exc:
            failure = type(exc).__name__  # Never retain an untrusted exception message.
        finally:
            if session is not None and not close_attempted:
                try:
                    closure = self._close_session(session, linked_digest=linked_digest)
                except Exception:
                    failure, status = "PolicyCloseUnverified", "failed"
            if wire is not None:
                wire.close()
        receipt = {"schema_version": "native_g1_private_policy_relay_session.v1",
                "status": status, "binding": asdict(self.binding), "forwarded_request_count": count,
                "inference_query_count": inference_count,
                "failure_type": failure, "policy_session_closed": closure is not None,
                "synthetic_conformance_digest": conformance_digest,
                "policy_session_close_digest": closure["receipt_digest"] if closure else None,
                "provider_teardown_verified": False, "claim_ceiling": "planning_only"}
        receipt["receipt_digest"] = _digest(receipt)
        return receipt

    def _reply(self, request_id: int, payload: dict[str, Any]) -> dict[str, Any]:
        return {"protocol": PROTOCOL, "request_id": request_id,
                "profile_digest": self.binding.profile_digest, "payload": payload}

    def _close_session(self, session: Any, *, linked_digest: str | None) -> dict[str, Any]:
        value = session.close()
        validate_relay_close(value, self.binding, conformance_digest=session.conformance["receipt_digest"],
                             linked_episode_digest=linked_digest)
        return value


class G1PolicyRelayClient:
    def __init__(self, *, path: Path, binding: RelayBinding, secret: str, timeout_seconds: float) -> None:
        _private_path(path)
        _secret(secret)
        timeout = _timeout(timeout_seconds)
        info = path.lstat()
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
            raise ValueError("g1_relay_socket_not_private")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._wire = JsonlSocket(sock, timeout_seconds=timeout)
        self.binding, self.profile_digest = binding, binding.profile_digest
        self._index, self._closed, self._failed = 0, None, False
        self._linked_digest = None
        self.candidate_policy_queried = False
        try:
            sock.settimeout(timeout)
            sock.connect(str(path))
            challenge = self._wire.read()
            if (set(challenge) != {"protocol", "challenge"} or challenge["protocol"] != PROTOCOL
                    or not isinstance(challenge["challenge"], str) or not _SECRET.fullmatch(challenge["challenge"])):
                raise ValueError("g1_relay_challenge_invalid")
            self._wire.write({"protocol": PROTOCOL, "binding": asdict(binding),
                              "proof": _proof(secret, challenge["challenge"], binding)})
            ready = self._wire.read()
            if (set(ready) != {"protocol", "status", "binding", "conformance"}
                    or ready["protocol"] != PROTOCOL or ready["status"] != "ready"
                    or ready["binding"] != asdict(binding)):
                raise ValueError("g1_relay_ready_invalid")
            validate_relay_conformance(ready["conformance"], binding)
            self.conformance = ready["conformance"]
        except BaseException:
            self._wire.close()
            raise

    def _exchange(self, kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self._failed:
            raise ValueError("g1_relay_client_failed")
        if self._closed is not None:
            raise ValueError("g1_relay_client_closed")
        request_id = self._index
        self._index += 1
        try:
            self._wire.write({"protocol": PROTOCOL, "request_id": request_id, "kind": kind, "payload": payload})
            reply = self._wire.read()
            if (set(reply) != {"protocol", "request_id", "profile_digest", "payload"}
                    or reply["protocol"] != PROTOCOL or type(reply["request_id"]) is not int
                    or reply["request_id"] != request_id or reply["profile_digest"] != self.profile_digest
                    or not isinstance(reply["payload"], dict)):
                raise ValueError("g1_relay_response_identity_invalid")
            return reply["payload"]
        except BaseException:
            self._failed = True
            self._wire.close()
            raise

    def reset(self, *, seed: int) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("g1_relay_seed_invalid")
        if self._exchange("reset", {"seed": seed}).get("ok") is not True:
            raise ValueError("g1_relay_reset_ack_invalid")
        self.candidate_policy_queried = False

    def infer_chunk(self, *, front_rgb: Any, observation_state: Any, task: str) -> list[list[float]]:
        from .native_g1_humanoidarena_policy_client import (
            build_semantic_v3_infer_request, validate_semantic_v3_infer_response,
        )
        payload = build_semantic_v3_infer_request(front_rgb=front_rgb, observation_state=observation_state, task=task)
        try:
            actions = validate_semantic_v3_infer_response(self._exchange("infer", payload))
        except BaseException:
            self._failed = True
            self._wire.close()
            raise
        self.candidate_policy_queried = True
        return actions

    def close(self) -> dict[str, Any]:
        if self._closed is not None:
            return self._closed
        try:
            value = self._exchange("close", {})
            validate_relay_close(value, self.binding, conformance_digest=self.conformance["receipt_digest"],
                                 linked_episode_digest=self._linked_digest)
            self._closed = value
            return value
        finally:
            self._wire.close()

    def link_scored_episode(self, episode: dict[str, Any]) -> None:
        if self._linked_digest is not None:
            raise ValueError("g1_relay_episode_already_linked")
        value = self._exchange("link_episode", {"episode": episode})
        if value != {"linked_scored_episode_result_digest": episode["result_digest"]}:
            self._failed = True
            self._wire.close()
            raise ValueError("g1_relay_episode_link_ack_invalid")
        self._linked_digest = episode["result_digest"]
