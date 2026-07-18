"""Native Unitree-DDS bridge from one live Isaac G1 state snapshot.

The bridge is compiled inside the sealed worker from the exact Unitree SDK
already vendored with the pinned GEAR-SONIC controller. It publishes fresh,
name-ordered state sampled from the same persistent Isaac articulation. During
one thread-affine Isaac render/update call it may zero-order-hold the last
validated sample so GEAR's 500 ms transport watchdog does not permanently exit;
the heartbeat preserves the true source age and marks that held interval.
It never invents a second robot or uses a MuJoCo sidecar as task state.
"""

from __future__ import annotations

import base64
import gzip
import hashlib

from .gear_sonic_joint_order_contract import PROTOCOL_V4_BODY_JOINT_NAMES

BRIDGE_ROOT = "/workspace/runtime_overlay/gear_sonic_isaac_dds_bridge"
BRIDGE_SOURCE_PATH = f"{BRIDGE_ROOT}/gear_sonic_isaac_dds_bridge.cpp"
BRIDGE_BINARY_PATH = f"{BRIDGE_ROOT}/gear_sonic_isaac_dds_bridge"
BRIDGE_MANIFEST_PATH = (
    "/workspace/closed_loop_out/gear_sonic_isaac_dds_bridge_build.json"
)
BRIDGE_HEARTBEAT_PATH = (
    "/workspace/closed_loop_out/gear_sonic_isaac_dds_bridge_heartbeat.json"
)
BRIDGE_LOG_PATH = "/workspace/gear_sonic_isaac_dds_bridge.log"
SNAPSHOT_ENV = "BLUEPRINT_GEAR_SONIC_ISAAC_STATE_SNAPSHOT"
SNAPSHOT_DEFAULT_PATH = (
    "/workspace/closed_loop_out/gear_sonic_isaac_state_snapshot.json"
)
BRIDGE_REQUIRED_ENV = "BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_REQUIRED"
BRIDGE_PID_ENV = "GEAR_SONIC_ISAAC_BRIDGE_PID"
BRIDGE_MAX_SOURCE_AGE_MS = 500


def _cpp_string_list(values: tuple[str, ...]) -> str:
    return ",\n      ".join(f'"{value}"' for value in values)


NATIVE_BRIDGE_SOURCE = f'''// Generated from a hash-pinned Blueprint source module.
#include <unitree/dds_wrapper/common/crc.h>
#include <unitree/idl/hg/IMUState_.hpp>
#include <unitree/idl/hg/LowState_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>

#include <nlohmann/json.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>

namespace fs = std::filesystem;
using Json = nlohmann::json;
using LowState = unitree_hg::msg::dds_::LowState_;
using ImuState = unitree_hg::msg::dds_::IMUState_;
using unitree::robot::ChannelFactory;
using unitree::robot::ChannelPublisher;

constexpr std::uint64_t kMaxSourceAgeNs = {BRIDGE_MAX_SOURCE_AGE_MS}ULL * 1000000ULL;
constexpr std::size_t kBodyJointCount = 29;
std::atomic<bool> g_running{{true}};

const std::array<std::string, kBodyJointCount> kExpectedJointNames = {{{{
      {_cpp_string_list(PROTOCOL_V4_BODY_JOINT_NAMES)}
}}}};

void Stop(int) {{ g_running.store(false); }}

std::uint64_t NowNs() {{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
}}

std::uint64_t CapturedAtNs(const Json& payload) {{
  const auto& value = payload.at("captured_at_ns");
  if (value.is_string()) return std::stoull(value.get<std::string>());
  if (value.is_number_unsigned()) return value.get<std::uint64_t>();
  if (value.is_number_integer()) {{
    const auto signed_value = value.get<std::int64_t>();
    if (signed_value <= 0) throw std::runtime_error("snapshot_captured_at_ns_invalid");
    return static_cast<std::uint64_t>(signed_value);
  }}
  throw std::runtime_error("snapshot_captured_at_ns_invalid");
}}

template <std::size_t N>
std::array<float, N> FiniteArray(const Json& payload, const char* field) {{
  const auto& value = payload.at(field);
  if (!value.is_array() || value.size() != N) {{
    throw std::runtime_error(std::string("snapshot_dimension_invalid:") + field);
  }}
  std::array<float, N> result{{}};
  for (std::size_t index = 0; index < N; ++index) {{
    const double item = value.at(index).get<double>();
    if (!std::isfinite(item)) {{
      throw std::runtime_error(std::string("snapshot_nonfinite:") + field);
    }}
    result[index] = static_cast<float>(item);
  }}
  return result;
}}

void ValidateIdentity(const Json& payload) {{
  if (payload.value("schema_version", "") != "gear_sonic_isaac_state_snapshot.v1")
    throw std::runtime_error("snapshot_schema_version_invalid");
  if (payload.value("source", "") != "live_isaac_articulation")
    throw std::runtime_error("snapshot_source_invalid");
  if (payload.value("surrogate", true))
    throw std::runtime_error("snapshot_surrogate_forbidden");
  if (payload.value("simulator_session_id", "").empty())
    throw std::runtime_error("snapshot_simulator_session_id_missing");
  if (payload.value("stage_id", "").empty())
    throw std::runtime_error("snapshot_stage_id_missing");
  const auto& names = payload.at("body_joint_names");
  if (!names.is_array() || names.size() != kBodyJointCount)
    throw std::runtime_error("snapshot_body_joint_names_dimension_invalid");
  for (std::size_t index = 0; index < kBodyJointCount; ++index) {{
    if (names.at(index).get<std::string>() != kExpectedJointNames[index])
      throw std::runtime_error("snapshot_body_joint_names_order_mismatch");
  }}
}}

Json ReadSnapshot(const fs::path& path) {{
  std::ifstream stream(path);
  if (!stream) throw std::runtime_error("snapshot_unavailable");
  Json payload;
  stream >> payload;
  ValidateIdentity(payload);
  return payload;
}}

std::array<float, 3> QuaternionToRpy(const std::array<float, 4>& q) {{
  const double w = q[0], x = q[1], y = q[2], z = q[3];
  const double sinr = 2.0 * (w * x + y * z);
  const double cosr = 1.0 - 2.0 * (x * x + y * y);
  const double sinp = 2.0 * (w * y - z * x);
  const double roll = std::atan2(sinr, cosr);
  const double pitch = std::abs(sinp) >= 1.0
      ? std::copysign(1.5707963267948966, sinp) : std::asin(sinp);
  const double siny = 2.0 * (w * z + x * y);
  const double cosy = 1.0 - 2.0 * (y * y + z * z);
  return {{static_cast<float>(roll), static_cast<float>(pitch),
           static_cast<float>(std::atan2(siny, cosy))}};
}}

void AtomicHeartbeat(const fs::path& path, const Json& payload) {{
  const fs::path temporary = path.string() + ".tmp";
  {{
    std::ofstream stream(temporary, std::ios::trunc);
    if (!stream) throw std::runtime_error("bridge_heartbeat_open_failed");
    stream << payload.dump(2) << '\\n';
    stream.flush();
    if (!stream) throw std::runtime_error("bridge_heartbeat_write_failed");
  }}
  fs::rename(temporary, path);
}}

std::string BridgeSourceSha() {{
  const char* source_sha =
      std::getenv("BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_SOURCE_SHA256");
  return source_sha ? source_sha : "";
}}

void StartupPhase(const fs::path& snapshot_path, const fs::path& heartbeat_path,
                  const char* phase) {{
  const std::uint64_t heartbeat_at_ns = NowNs();
  std::cout << "gear_sonic_isaac_dds_bridge_startup_phase=" << phase
            << " heartbeat_at_ns=" << heartbeat_at_ns << std::endl;
  AtomicHeartbeat(heartbeat_path, Json{{
      {{"schema_version", "gear_sonic_isaac_dds_bridge_heartbeat.v1"}},
      {{"status", "initializing"}},
      {{"startup_phase", phase}},
      {{"snapshot_path", snapshot_path.string()}},
      {{"captured_at_ns", "0"}},
      {{"heartbeat_at_ns", std::to_string(heartbeat_at_ns)}},
      {{"source_age_ms", nullptr}},
      {{"simulator_session_id", ""}},
      {{"stage_id", ""}},
      {{"publish_count", 0}},
      {{"source_fresh", false}},
      {{"holding_last_validated_snapshot", false}},
      {{"last_error", ""}},
      {{"surrogate", false}},
      {{"bridge_source_sha256", BridgeSourceSha()}},
  }});
}}

int main(int argc, char** argv) {{
  if (argc != 3) {{
    std::cerr << "usage: gear_sonic_isaac_dds_bridge SNAPSHOT HEARTBEAT\\n";
    return 64;
  }}
  std::signal(SIGTERM, Stop);
  std::signal(SIGINT, Stop);
  const fs::path snapshot_path(argv[1]);
  const fs::path heartbeat_path(argv[2]);
  fs::create_directories(heartbeat_path.parent_path());

  StartupPhase(snapshot_path, heartbeat_path, "process_started");
  StartupPhase(snapshot_path, heartbeat_path, "channel_factory_initializing");
  ChannelFactory::Instance()->Init(0, "lo");
  StartupPhase(snapshot_path, heartbeat_path, "channel_factory_initialized");
  StartupPhase(snapshot_path, heartbeat_path, "channel_publishers_constructing");
  ChannelPublisher<LowState> lowstate_publisher("rt/lowstate");
  ChannelPublisher<ImuState> torso_imu_publisher("rt/secondary_imu");
  StartupPhase(snapshot_path, heartbeat_path, "channel_publishers_constructed");
  StartupPhase(snapshot_path, heartbeat_path, "lowstate_publisher_initializing");
  lowstate_publisher.InitChannel();
  StartupPhase(snapshot_path, heartbeat_path, "lowstate_publisher_initialized");
  StartupPhase(snapshot_path, heartbeat_path, "torso_imu_publisher_initializing");
  torso_imu_publisher.InitChannel();
  StartupPhase(snapshot_path, heartbeat_path, "dds_publishers_initialized");

  std::uint64_t publish_count = 0;
  std::uint32_t tick = 0;
  std::string last_error;
  bool announced_ready = false;
  // Subtracting time_point::min() from ``now`` can overflow the signed
  // steady-clock duration before the first loop iteration.  On the native
  // x86 worker that made the first periodic heartbeat predicate permanently
  // false even though DDS publication itself was succeeding.  Seed the timer
  // one interval in the past so the first loop always emits a ready/waiting
  // heartbeat without relying on an out-of-range duration calculation.
  auto last_heartbeat =
      std::chrono::steady_clock::now() - std::chrono::milliseconds(100);

  while (g_running.load()) {{
    const auto loop_started = std::chrono::steady_clock::now();
    std::uint64_t captured_at_ns = 0;
    std::uint64_t source_age_ns = 0;
    std::string simulator_session_id;
    std::string stage_id;
    bool published = false;
    bool holding_last_validated_snapshot = false;
    try {{
      const Json snapshot = ReadSnapshot(snapshot_path);
      captured_at_ns = CapturedAtNs(snapshot);
      const std::uint64_t now_ns = NowNs();
      if (captured_at_ns > now_ns + 1000000000ULL)
        throw std::runtime_error("snapshot_timestamp_in_future");
      source_age_ns = now_ns > captured_at_ns ? now_ns - captured_at_ns : 0;
      holding_last_validated_snapshot = source_age_ns > kMaxSourceAgeNs;
      if (holding_last_validated_snapshot && publish_count == 0)
        throw std::runtime_error("snapshot_stale_before_first_valid_publish");

      const auto body_q = FiniteArray<29>(snapshot, "body_q");
      const auto body_dq = FiniteArray<29>(snapshot, "body_dq");
      const auto base_quat = FiniteArray<4>(snapshot, "base_quaternion_wxyz");
      const auto base_omega = FiniteArray<3>(snapshot, "base_angular_velocity");
      const auto acceleration = FiniteArray<3>(snapshot, "accelerometer_mps2");
      double q_norm = 0.0;
      for (const float item : base_quat) q_norm += item * item;
      if (!std::isfinite(q_norm) || std::abs(std::sqrt(q_norm) - 1.0) > 0.02)
        throw std::runtime_error("snapshot_base_quaternion_not_normalized");

      ImuState imu;
      imu.quaternion(base_quat);
      imu.gyroscope(base_omega);
      imu.accelerometer(acceleration);
      imu.rpy(QuaternionToRpy(base_quat));

      LowState lowstate;
      lowstate.version()[0] = 1;
      lowstate.mode_pr(0);
      lowstate.mode_machine(0);
      lowstate.tick(++tick);
      lowstate.imu_state(imu);
      for (std::size_t index = 0; index < kBodyJointCount; ++index) {{
        lowstate.motor_state()[index].mode(1);
        lowstate.motor_state()[index].q(body_q[index]);
        lowstate.motor_state()[index].dq(body_dq[index]);
        lowstate.motor_state()[index].motorstate(0);
      }}
      lowstate.crc(crc32_core(reinterpret_cast<std::uint32_t*>(&lowstate),
                              (sizeof(LowState) >> 2) - 1));
      if (!lowstate_publisher.Write(lowstate) || !torso_imu_publisher.Write(imu))
        throw std::runtime_error("dds_publish_failed");

      simulator_session_id = snapshot.at("simulator_session_id").get<std::string>();
      stage_id = snapshot.at("stage_id").get<std::string>();
      ++publish_count;
      published = true;
      last_error = holding_last_validated_snapshot
          ? "holding_last_validated_snapshot_source_stale" : "";
      if (!announced_ready) {{
        std::cout << "gear_sonic_isaac_dds_bridge_ready" << std::endl;
        announced_ready = true;
      }}
    }} catch (const std::exception& error) {{
      last_error = error.what();
    }}

    const auto now_steady = std::chrono::steady_clock::now();
    if (now_steady - last_heartbeat >= std::chrono::milliseconds(100)) {{
      Json heartbeat = {{
          {{"schema_version", "gear_sonic_isaac_dds_bridge_heartbeat.v1"}},
          {{"status", published
              ? (holding_last_validated_snapshot
                    ? "holding_last_validated_isaac_state" : "ready")
              : "waiting_for_fresh_isaac_state"}},
          {{"startup_phase", "publishing_live_isaac_state"}},
          {{"snapshot_path", snapshot_path.string()}},
          {{"captured_at_ns", std::to_string(captured_at_ns)}},
          {{"heartbeat_at_ns", std::to_string(NowNs())}},
          {{"source_age_ms", static_cast<double>(source_age_ns) / 1000000.0}},
          {{"simulator_session_id", simulator_session_id}},
          {{"stage_id", stage_id}},
          {{"publish_count", publish_count}},
          {{"source_fresh", published && !holding_last_validated_snapshot}},
          {{"holding_last_validated_snapshot", holding_last_validated_snapshot}},
          {{"last_error", last_error}},
          {{"surrogate", false}},
          {{"bridge_source_sha256", BridgeSourceSha()}},
      }};
      AtomicHeartbeat(heartbeat_path, heartbeat);
      last_heartbeat = now_steady;
    }}
    std::this_thread::sleep_until(loop_started + std::chrono::milliseconds(5));
  }}
  ChannelFactory::Instance()->Release();
  return 0;
}}
'''.encode("utf-8")

NATIVE_BRIDGE_SOURCE_SHA256 = hashlib.sha256(NATIVE_BRIDGE_SOURCE).hexdigest()


def bridge_prepare_script() -> str:
    """Return a fail-closed worker fragment that materializes and links bridge."""

    payload = base64.b64encode(
        gzip.compress(NATIVE_BRIDGE_SOURCE, compresslevel=9, mtime=0)
    ).decode("ascii")
    return f'''mkdir -p {BRIDGE_ROOT} /workspace/closed_loop_out
python3 - <<'PY'
import base64, gzip, hashlib
from pathlib import Path

payload = base64.b64decode({payload!r}, validate=True)
source = gzip.decompress(payload)
expected = {NATIVE_BRIDGE_SOURCE_SHA256!r}
if hashlib.sha256(source).hexdigest() != expected:
    raise SystemExit("gear_sonic_isaac_dds_bridge_embedded_sha256_mismatch")
path = Path({BRIDGE_SOURCE_PATH!r})
path.write_bytes(source)
PY
SDK_ROOT=/opt/wbc/gear_sonic_deploy/thirdparty/unitree_sdk2
SDK_LIB="$SDK_ROOT/lib/x86_64"
DDS_LIB="$SDK_ROOT/thirdparty/lib/x86_64"
for SDK_FILE in "$SDK_LIB/libunitree_sdk2.a" "$DDS_LIB/libddsc.so" "$DDS_LIB/libddscxx.so"; do
  if [ ! -f "$SDK_FILE" ] || [ "$(wc -c < "$SDK_FILE")" -lt 1048576 ]; then
    echo "gear_sonic_isaac_dds_bridge_sdk_library_missing_or_lfs_pointer:$SDK_FILE" >&2
    exit 46
  fi
done
g++ -std=c++17 -O2 -pthread -I"$SDK_ROOT/include" \
  -I"$SDK_ROOT/thirdparty/include" -I"$SDK_ROOT/thirdparty/include/ddscxx" \
  {BRIDGE_SOURCE_PATH} -o {BRIDGE_BINARY_PATH} \
  "$SDK_LIB/libunitree_sdk2.a" -L"$DDS_LIB" \
  -Wl,-rpath,"$DDS_LIB" -lddscxx -lddsc -pthread
if ldd {BRIDGE_BINARY_PATH} | grep -F 'not found'; then
  echo "gear_sonic_isaac_dds_bridge_elf_dependency_missing" >&2
  exit 47
fi
BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_SOURCE_SHA256={NATIVE_BRIDGE_SOURCE_SHA256} \
python3 - <<'PY'
import hashlib, json, os
from pathlib import Path

root = Path("/opt/wbc/gear_sonic_deploy/thirdparty/unitree_sdk2")
binary = Path({BRIDGE_BINARY_PATH!r})
source = Path({BRIDGE_SOURCE_PATH!r})
libraries = [root / "lib/x86_64/libunitree_sdk2.a", root / "thirdparty/lib/x86_64/libddsc.so", root / "thirdparty/lib/x86_64/libddscxx.so"]
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {{
    "schema_version": "gear_sonic_isaac_dds_bridge_build.v1",
    "status": "compiled_and_elf_audited",
    "source_path": str(source),
    "source_sha256": sha(source),
    "expected_source_sha256": os.environ["BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_SOURCE_SHA256"],
    "binary_path": str(binary),
    "binary_sha256": sha(binary),
    "sdk_libraries": [{{"path": str(path), "size_bytes": path.stat().st_size, "sha256": sha(path)}} for path in libraries],
    "network_interface": "lo",
    "topics": ["rt/lowstate", "rt/secondary_imu"],
    "raw_secret_values_recorded": False,
    "claim_boundary": {{"bridge_build_is_not_controller_readiness": True, "bridge_build_is_not_task_success": True}},
}}
Path({BRIDGE_MANIFEST_PATH!r}).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\\n", encoding="utf-8")
if manifest["source_sha256"] != manifest["expected_source_sha256"]:
    raise SystemExit("gear_sonic_isaac_dds_bridge_materialized_sha256_mismatch")
PY
'''


def bridge_start_script() -> str:
    """Return the process launch fragment inserted beside GEAR and Isaac."""

    return f'''export {SNAPSHOT_ENV}="${{{SNAPSHOT_ENV}:-{SNAPSHOT_DEFAULT_PATH}}}"
export {BRIDGE_REQUIRED_ENV}=true
export BLUEPRINT_GEAR_SONIC_ISAAC_DDS_BRIDGE_SOURCE_SHA256={NATIVE_BRIDGE_SOURCE_SHA256}
{BRIDGE_BINARY_PATH} "${{{SNAPSHOT_ENV}}}" {BRIDGE_HEARTBEAT_PATH} > {BRIDGE_LOG_PATH} 2>&1 &
{BRIDGE_PID_ENV}=$!
export {BRIDGE_PID_ENV}
'''
