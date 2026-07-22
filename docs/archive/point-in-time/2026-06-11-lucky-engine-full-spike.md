# Lucky Engine Full Spike - 2026-06-11

> Archived investigation snapshot. Not a current implementation plan.

## Decision

Do not adopt `lucky_engine` as a supported Blueprint simulator backend yet.

Keep Lucky Engine as an experimental external-runner candidate only. It should not
be added to `SIMULATOR_FRAMEWORKS`, `SIMULATORS`, first-GPU defaults, production
handoff readiness, or live closure gates until a real Lucky run produces accepted
Blueprint proof artifacts.

## Scope Run

This spike tested the public Lucky Engine 2026.1 macOS arm64 release and the
`luckyrobots==0.3.0` Python client package on this host.

Artifacts:

- `output/lucky-engine-spike-2026-06-11/lucky_engine_spike_manifest.json`
- `output/lucky-engine-spike-2026-06-11/runtime-smoke/runtime_probe_manifest.json`
- `output/lucky-engine-spike-2026-06-11/runtime-smoke-copy-headless/runtime_probe_manifest.json`
- `output/lucky-engine-spike-2026-06-11/runtime-smoke-copy-windowed/runtime_probe_manifest.json`
- `output/lucky-engine-spike-2026-06-11/runtime-smoke-copy-bin-headless/runtime_probe_manifest.json`
- `output/lucky-engine-spike-2026-06-11/runtime-smoke-copy-bin-windowed/runtime_probe_manifest.json`
- `output/lucky-engine-spike-2026-06-11/HAZEL.log`

## What Worked

- Downloaded the official macOS arm64 DMG:
  `https://downloads.luckyrobots.com/releases/2026/1/0/LuckyEngine-2026.1-macOS-arm64.dmg`
- Mounted the DMG and inspected the signed app bundle.
- Installed `luckyrobots==0.3.0` in an isolated venv under `/tmp/blueprint-lucky-spike/.venv`.
- Confirmed the SDK exposes useful client surfaces including `Session`,
  `LuckyEngineClient`, `MujocoScene`, `LuckyEnv`, `PolicyEnv`, and
  `RobotController`.
- Confirmed the app bundle includes the `RobotSandbox` project, `Welcome` scene,
  gRPC proto files, and an offline Panda robot pack.

## What Failed

No local Lucky simulator execution was proven.

Five launch variants failed to expose the gRPC server on `127.0.0.1:50051`:

- Mounted app, headless: `LuckyEditor ... --auto-play=1 -Headless`
- Copied writable app, headless: `LuckyEditor ... --auto-play=1 -Headless`
- Copied writable app, windowed: `LuckyEditor ... --auto-play=1`
- Copied writable app direct binary, headless:
  `LuckyEditor-bin ... --auto-play=1 -Headless`
- Copied writable app direct binary, windowed:
  `LuckyEditor-bin ... --auto-play=1`

The wrapper launches entered Lucky's crash reporter before gRPC became
reachable. The captured stderr pattern was:

```text
[crash-reporter] child signalled crash, showing dialog immediately
[crash-reporter] ShowDialog: building layer
[crash-reporter] ShowDialog: MakeDialogSpec
[crash-reporter] ShowDialog: constructing DialogApplication
[crash-reporter] ShowDialog: app->Run()
```

The direct `LuckyEditor-bin` launches exposed the underlying fatal error:

```text
[Lucky Engine - Fatal Error] Verify Failed (src/Hazel/Platform/Vulkan/VulkanDeviceManager.cpp:97) : glfwGetRequiredInstanceExtensions returned NULL.
GLFW error (65548): Cocoa: Regular windows do not have icons on macOS

This usually means no Vulkan-capable GPU driver is installed.
The Vulkan loader (vulkan-1.dll) may be present but no ICD (Installable Client Driver) was found.
```

The copied `HAZEL.log` records the same renderer failure.

The downloaded app bundle did not include offline Unitree G1 or Go2 assets. The
public docs describe Unitree examples as Content Vault examples, but the local
bundle inspection only found Panda assets without signing in or installing
additional vault content.

The `luckyrobots` console script is client tooling only. It exposes `inspect`
and `sysid`, but does not start the engine. The `inspect` command also failed
before useful server inspection because its implementation constructs
`Session.connect()` without a robot name.

## Blueprint Fit

Lucky Engine still looks interesting for Blueprint, but not as a current default
or replacement for Isaac Sim / Isaac Lab.

Potential fit after proof:

- Optional MuJoCo-backed external runner for fast local or cloud smoke tests.
- A gRPC-driven robot-control surface that could map to Blueprint owner proof
  traces.
- A possible synthetic-data and policy/IK co-control lane if Content Vault
  assets and recording outputs can be automated.

Current blockers:

- No local simulator run completed.
- No gRPC step loop was proven.
- No robot POV, policy trace, scene load trace, spawn trace, or owner artifact
  manifest was produced by Lucky.
- The macOS runtime failed before renderer initialization with
  `glfwGetRequiredInstanceExtensions returned NULL`.
- No included offline Unitree humanoid/quadruped pack was available in the DMG.
- The runtime currently requires more vendor/runtime debugging before it can be
  trusted as a Blueprint runner.

## Required Evidence Before Adoption

Do not add `lucky_engine` to the supported simulator framework lists until a
runner can provide:

- `gpu_owner_system_proof.json` or an equivalent owner-system proof input.
- Simulator stdout/stderr and a zero exit code.
- Scene load trace.
- Spawn pose trace.
- Action or policy trace.
- Robot POV frame/video manifest.
- Lucky Engine version, SDK version, host OS, hardware, and launch command.
- Clear robot asset identity, including whether Unitree G1/Go2 came from the
  Content Vault, a bundled pack, or an owner-supplied asset.
- A proof boundary that keeps generated-world rank fidelity, safety/contact validation, and
  public-claim upgrades false unless separately proven.

## Next Practical Test

Run Lucky Engine on the vendor's strongest supported path before any repo
contract change:

1. Use a Linux or Windows machine with Vulkan 1.3 support, or debug the macOS
   crash with Lucky Robots.
2. Launch `RobotSandbox` and start gRPC on `127.0.0.1:50051`.
3. Install a Unitree G1 or Go2 Content Vault example if available.
4. Run a 100 to 1,000 step deterministic gRPC episode with zero or simple
   walk-to-target actions.
5. Emit the same proof artifacts expected by `blueprint-run-owner-gpu-proof`.
6. Only then consider adding `lucky_engine` as an experimental, blocked-by-default
   simulator backend.
