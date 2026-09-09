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
