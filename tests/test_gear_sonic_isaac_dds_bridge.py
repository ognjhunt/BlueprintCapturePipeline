from __future__ import annotations

import hashlib

from blueprint_pipeline import gear_sonic_isaac_dds_bridge as bridge
from blueprint_pipeline.gear_sonic_joint_order_contract import (
    PROTOCOL_V4_BODY_JOINT_NAMES,
)


def test_native_bridge_source_is_bound_to_live_isaac_and_unitree_topics() -> None:
    source = bridge.NATIVE_BRIDGE_SOURCE.decode("utf-8")
    assert hashlib.sha256(bridge.NATIVE_BRIDGE_SOURCE).hexdigest() == (
        bridge.NATIVE_BRIDGE_SOURCE_SHA256
    )
    assert 'ChannelPublisher<LowState> lowstate_publisher("rt/lowstate")' in source
    assert 'ChannelPublisher<ImuState> torso_imu_publisher("rt/secondary_imu")' in source
    assert 'ChannelFactory::Instance()->Init(0, "lo")' in source
    assert '"status", "initializing"' in source
    assert '"heartbeat_at_ns"' in source
    assert '"startup_phase"' in source
    assert "gear_sonic_isaac_dds_bridge_startup_phase=" in source
    phase_markers = (
        "process_started",
        "channel_factory_initializing",
        "channel_factory_initialized",
        "channel_publishers_constructing",
        "channel_publishers_constructed",
        "lowstate_publisher_initializing",
        "lowstate_publisher_initialized",
        "torso_imu_publisher_initializing",
        "dds_publishers_initialized",
    )
    for marker in phase_markers:
        assert f'"{marker}"' in source
    assert source.index('"channel_factory_initializing"') < source.index(
        'ChannelFactory::Instance()->Init(0, "lo")'
    )
    assert source.index('ChannelFactory::Instance()->Init(0, "lo")') < source.index(
        '"channel_factory_initialized"'
    )
    assert source.index('"lowstate_publisher_initializing"') < source.index(
        "lowstate_publisher.InitChannel()"
    )
    assert source.index('"torso_imu_publisher_initializing"') < source.index(
        "torso_imu_publisher.InitChannel()"
    )
    assert "crc32_core" in source
    assert "snapshot_stale_before_first_valid_publish" in source
    assert "source_age_ns > kMaxSourceAgeNs" in source
    assert "holding_last_validated_isaac_state" in source
    assert "holding_last_validated_snapshot_source_stale" in source
    assert '"source_fresh"' in source
    assert 'payload.value("source", "") != "live_isaac_articulation"' in source
    assert 'payload.value("surrogate", true)' in source
    assert "steady_clock::time_point::min()" not in source
    assert (
        "std::chrono::steady_clock::now() - std::chrono::milliseconds(100)"
        in source
    )
    assert "body_q" in source and "body_dq" in source
    for name in PROTOCOL_V4_BODY_JOINT_NAMES:
        assert f'"{name}"' in source


def test_prepare_script_compiles_against_vendored_sdk_and_audits_elf() -> None:
    script = bridge.bridge_prepare_script()
    assert bridge.NATIVE_BRIDGE_SOURCE_SHA256 in script
    assert "/opt/wbc/gear_sonic_deploy/thirdparty/unitree_sdk2" in script
    assert '-I"$SDK_ROOT/include"' in script
    assert '-I"$SDK_ROOT/thirdparty/include/ddscxx"' in script
    assert '"$SDK_LIB/libunitree_sdk2.a"' in script
    assert '-L"$DDS_LIB"' in script
    assert "-lddscxx -lddsc -pthread" in script
    assert "libunitree_sdk2.a" in script
    assert "libddsc.so" in script
    assert "libddscxx.so" in script
    assert "1048576" in script
    assert f"ldd {bridge.BRIDGE_BINARY_PATH}" in script
    assert bridge.BRIDGE_MANIFEST_PATH in script
    assert "raw_secret_values_recorded" in script


def test_start_script_binds_snapshot_heartbeat_pid_and_source() -> None:
    script = bridge.bridge_start_script()
    assert bridge.SNAPSHOT_ENV in script
    assert bridge.SNAPSHOT_DEFAULT_PATH in script
    assert bridge.BRIDGE_REQUIRED_ENV in script
    assert bridge.BRIDGE_PID_ENV in script
    assert bridge.BRIDGE_HEARTBEAT_PATH in script
    assert bridge.BRIDGE_LOG_PATH in script
    assert bridge.NATIVE_BRIDGE_SOURCE_SHA256 in script
