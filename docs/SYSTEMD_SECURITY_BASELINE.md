# Systemd security baseline

The live Pipeline control-plane, intake, Pub/Sub listener, and GPU spend guard
run as the dedicated `blueprint` user and group. The installer creates writable
state and handoff roots with mode `0750`; the environment file is `root:blueprint`
mode `0640`. The repository and operating system remain read-only to the units.

Every production unit must retain these controls:

- `NoNewPrivileges`, empty capability and ambient-capability sets;
- strict system/home/device/kernel/control-group protection;
- private temporary storage, `UMask=0077`, and explicit `ReadWritePaths`;
- bounded address families, system-call architecture/filter, tasks, file
  descriptors, memory, and CPU.

On the target Linux image, validate the installed units after every change:

```bash
sudo systemd-analyze verify \
  /etc/systemd/system/blueprint-*.service

for unit in \
  blueprint-pipeline-control-plane.service \
  blueprint-pipeline-intake.service \
  blueprint-pubsub-handoff-listener.service \
  blueprint-gpu-spend-guard.service
do
  sudo systemd-analyze security --no-pager --threshold=4.0 "$unit"
done
```

The release threshold is an exposure score of **4.0 or lower for every unit**.
A missing analysis, higher score, root execution, mode regression, or disabled
control blocks the live launch evidence. Passing this static hardening check is
not proof of provider reachability, application correctness, or runtime IAM.
