"""Simulator-side session for a policy isolated on the operator's VM host.

The private config is created at runtime, never included in a sealed bundle.
The host owns the actual policy process. This proxy retains its receipts; it
does not synthesize child cleanup, grade a task or attest provider teardown.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
import os
from pathlib import Path
import stat
from typing import Any

from .native_g1_team_policy_relay import (
    G1PolicyRelayClient, RelayBinding, _object, _private_path, _secret,
    validate_relay_close,
)
from .native_g1_team_runtime_session import (
    CONFORMANCE_FILENAME, SESSION_SCHEMA, NativeG1TeamRuntimeSession,
)
from .team_policy_delivery_profile import validate_team_policy_delivery_profile

CONFIG_SCHEMA = "native_g1_team_private_relay_config.v1"


def write_g1_team_relay_config(
    *, path: Path, binding: RelayBinding, socket_path: Path, secret: str,
) -> None:
    _private_path(path)
    _private_path(socket_path)
    _secret(secret)
    if not isinstance(binding, RelayBinding):
        raise ValueError("g1_relay_config_binding_invalid")
    value = {"schema_version": CONFIG_SCHEMA, "binding": asdict(binding),
             "socket_path": str(socket_path), "secret": secret}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, allow_nan=False, sort_keys=True)
        stream.write("\n")


def read_g1_team_relay_config(
    *, config_path: Path, execution_packet_digest: str,
    profile: Mapping[str, Any], trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str],
) -> tuple[RelayBinding, Path, str]:
    """Reject foreign or unsafe configuration before connecting to the host."""
    bound = validate_team_policy_delivery_profile(
        profile, trusted_setup=trusted_setup, authenticated_owner=authenticated_owner,
    )
    binding = RelayBinding(execution_packet_digest, bound["profile_digest"],
                           bound["source_setup_digest"], bound["delivery"]["mode"])
    _private_path(config_path)
    try:
        descriptor = os.open(config_path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "r", encoding="utf-8") as stream:
            info = os.fstat(stream.fileno())
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or stat.S_IMODE(info.st_mode) != 0o600 or info.st_nlink != 1
                    or not 0 < info.st_size <= 16384):
                raise ValueError("g1_relay_config_not_private")
            value = json.loads(stream.read(16385), object_pairs_hook=_object)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("g1_relay_config_unavailable") from exc
    if (not isinstance(value, dict)
            or set(value) != {"schema_version", "binding", "socket_path", "secret"}
            or value["schema_version"] != CONFIG_SCHEMA
            or value["binding"] != asdict(binding) or not isinstance(value["socket_path"], str)):
        raise ValueError("g1_relay_config_binding_invalid")
    path = Path(value["socket_path"])
    _private_path(path)
    _secret(value["secret"])
    return binding, path, value["secret"]


class NativeG1TeamRelayRuntimeSession(NativeG1TeamRuntimeSession):
    def link_scored_episode(self, result: Mapping[str, Any]) -> None:
        super().link_scored_episode(result)
        self.client.link_scored_episode(dict(result))

    def close(self) -> dict[str, Any]:
        if self._closed is not None:
            return self._closed
        value = self.client.close()
        validate_relay_close(value, self.client.binding,
                             conformance_digest=self.conformance["receipt_digest"],
                             linked_episode_digest=self._linked_episode_digest)
        _write_receipt(self.output_dir / (SESSION_SCHEMA + ".json"), value)
        self._closed = value
        return value


def _write_receipt(path: Path, value: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def open_g1_team_relay_runtime_session(
    *, config_path: Path, execution_packet_digest: str,
    profile: Mapping[str, Any], trusted_setup: Mapping[str, Any],
    authenticated_owner: Mapping[str, str], output_dir: Path,
    timeout_seconds: float = 30,
) -> NativeG1TeamRelayRuntimeSession:
    binding, path, secret = read_g1_team_relay_config(
        config_path=config_path, execution_packet_digest=execution_packet_digest,
        profile=profile, trusted_setup=trusted_setup, authenticated_owner=authenticated_owner,
    )
    if (not isinstance(output_dir, Path) or not output_dir.is_absolute()
            or output_dir.resolve() != output_dir or output_dir.exists() or output_dir.is_symlink()):
        raise ValueError("g1_relay_output_invalid")
    client = G1PolicyRelayClient(path=path, binding=binding, secret=secret, timeout_seconds=timeout_seconds)
    try:
        output_dir.mkdir(mode=0o700)
        _write_receipt(output_dir / CONFORMANCE_FILENAME, client.conformance)
    except BaseException:
        client.close()
        raise
    return NativeG1TeamRelayRuntimeSession(
        client=client, conformance=client.conformance,
        profile_digest=binding.profile_digest, delivery_mode=binding.delivery_mode,
        output_dir=output_dir,
    )
