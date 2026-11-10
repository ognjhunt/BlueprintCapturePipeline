# Systemd security baseline

The live Pipeline control-plane, production and isolated-staging intake,
Pub/Sub listener, and GPU spend guard run as the dedicated `blueprint` user and
group. The installers create writable state and handoff roots with mode `0750`;
environment files are `root:blueprint` mode `0640`. The repository and operating
system remain read-only to the units.

Every production unit must retain these controls:

- `NoNewPrivileges`, empty capability and ambient-capability sets;
- strict system/home/device/kernel/control-group protection;
- private temporary storage, `UMask=0077`, and explicit `ReadWritePaths`;
- bounded address families, system-call architecture/filter, tasks, file
  descriptors, memory, and CPU.

Three resource rules sit beside those controls. `tests/test_deploy_systemd_contract.py`
holds each list and `scripts/verify_systemd_ci.py` checks the unit files.

- **Start timeout.** Every `Type=oneshot` unit sets a bounded `TimeoutStartSec`.
  A oneshot has none by default, and a timer that waits for its unit to go
  inactive never fires again after one hung run. Units that keep a GPU
  watchdog child alive use `KillMode=process`, so a timeout stops the worker
  only.
- **Batch memory slice.** The batch units name
  `Slice=system-blueprintbatch.slice`, which bounds them together
  (`MemoryHigh=4.5G`, `MemoryMax=5G`) below what the 8.33 GB host has free at
  idle (6.19 GB). No batch unit's own `MemoryMax` is above the slice's. The
  always-on, money-safety and housekeeping units stay outside the slice. Five
  money-safety units keep `MemoryMax=8G`, which is above the host and never
  acts; each file says so.
- **Failure notifier.** The units whose failure is abnormal and matters name
  `OnFailure=blueprint-unit-failure@%n.service`. The notifier writes one record
  for each failed unit and posts on the capacity alert route. A unit whose
  non-zero exit is an ordinary result does not name it.

The slice and the notifier are installed first, by the deploy tool and by the
installer. Nothing enables or starts them: systemd reads them at the
`daemon-reload` that precedes every restart.

On the target Linux image, validate the installed units after every change:

```bash
sudo systemd-analyze verify \
  /etc/systemd/system/blueprint-*.service

for unit in \
  blueprint-pipeline-intake.service \
  blueprint-pubsub-handoff-listener.service \
  blueprint-gpu-spend-guard.service
do
  # systemd's CLI uses an integer percentage: 40 corresponds to exposure 4.0.
  sudo systemd-analyze security --no-pager --threshold=40 "$unit"
done
```

The release threshold is an exposure score of **4.0 or lower for every unit**.
A missing analysis, higher score, root execution, mode regression, or disabled
control blocks the live launch evidence. Passing this static hardening check is
not proof of provider reachability, application correctness, or runtime IAM.
