# Native appearance cache for configured-scene controls

This supports ADP-009's development-only fixed-arm rehearsal and the day-14/day-21
native-controls prerequisite. A configured NuRec asset remains the appearance
source; the native ParticleField is a derived representation of the same learned
Gaussian arrays. Structural conversion does not qualify camera fidelity or policy
observations. Those still require the native render and controls gates.

Provision the CPU import closure once with the canonical control-plane interpreter:

```bash
PYTHONPATH=src .venv/bin/python scripts/install_native_appearance_transcode.py
```

The installer pins NVIDIA 3DGRUT at `a37ef721012dea0f29c0fcfff2d525023b4e854a`
and installs its transcode import dependencies under an isolated system-runtime
root, without altering the base venv. The base interpreter must already provide
torch, usd-core, and the pipeline's normal dependencies. The installer verifies the
real import closure and records dependency file hashes. Repeated installation
validates the existing immutable runtime. It does not download model weights.

The episode compiler automatically converts an uncached NuRec larger than 32 MiB
through that runtime, up to 128 MiB. A digest-specific lock prevents duplicate
conversion. The child has a 180-second deadline, and the compilation service has
a 4 GiB memory ceiling. Only a validated, digest-bound upstream artifact and
receipt can be atomically published to the cache. Later destination, construction,
and controls compiles reuse that exact cache entry. Failed builds retain diagnostic
logs and never create an admissible cache entry. Larger uncached sources refuse
explicitly; the size guard is not disabled.

The R26 retained-input replay exposed two missing production seams before native
GPU allocation: absent automatic cache creation and loss of the destination USD
format suffix when reading content-addressed references. Format suffixes now come
from the sealed USD bytes. This works for any admitted passive rigid destination;
there is no object-name or tray-specific branch.

The controls catalog's `runtime_source_payload_dir` must contain the verified
`native_task_runtime_source_packet.v1.json` and its `native_task_runtime_sources.zip`,
rather than expanded IsaacLab/Arena checkout directories. Provisioning verifies
the same runtime packet contract the episode consumer reads before publishing a
wrapper. Use `external_layer_bucket` for the canonical object-store bucket when
binding large runtime packets; the existing external-layer store reuses immutable
bytes by hardlink and publishes a small wrapper. Update the persistent operator
bootstrap through `build_bootstrap` and the normal controls-autoprovision installer.

Native activation now waits for its exact successful episode compilation and,
when the production disk ledger is configured, sufficient unreserved activation
capacity. Its release window is minted after those checks. This prevents the
activation worker from racing compilation for disk space or consuming its window
while the CPU packet is still being built. The worker's independent disk and
paid-resource gates remain mandatory.

The runtime adapter now hashes immutable external layers before linking them
into its member cache. It reuses the existing inode when possible; writable
inputs and unsupported link operations retain the verified copy path. This
avoids another physical 4 GiB copy while preserving every content digest and
retained path. A prepared activation also blocks unused-attempt retirement.
