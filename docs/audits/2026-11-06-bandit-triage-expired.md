# Bandit triage re-review: 36 entries expired on 2026-11-05

Status: **owner**. This page lists what needs review. It renews nothing and
makes no judgment about whether an entry is still right.

## Why it matters

Since 2026-11-05, `main`'s CI fails one job, "Bandit high and reviewed-medium
gate". That skips "SBOM, license, and provenance contract", so no `main` commit
after `9dd4e0dc4` gets a source attestation, and the operator door deploys only
attested commits. Two deploys have been blocked on it
(`deploy_scene_retirement_runtime_unproven`, `ci_finished_without_attestation`):
`5e826deba` at 11-06 03:57Z and `a5b47bd11` at 11-06 15:00Z. Until the gate
passes, #3411 and #3416 cannot deploy.

What the gate finds, reproduced locally on `692f79ca8` with the pinned Bandit
1.8.6 and `scripts/verify_bandit_policy.py`, the same as CI on `3d1cd4dd2`:

- 36 blockers, all `triage_expired`. Nothing else blocks.
- 0 high findings, 96 medium. All 96 medium findings are triaged and no new
  finding is untriaged.
- Every one of the 96 entries still matches a current finding by fingerprint,
  so the code at each site is the code that was reviewed. Line numbers below are
  where each finding is today.

The 36 expired entries were all reviewed on 2026-10-06 with a 30-day window,
and their `owner` field reads `Codex /root/independent_review`. There are 26
`mitigated_by_control` and 10 `false_positive`.

The other 60 entries expire on **2026-11-12**: 58 reviewed 2026-08-14 by
`@ognjhunt`, and 2 reviewed 2026-10-28 by `codex/real-video-e2e-20261028`. If
only the 36 are renewed, the gate fails again on 11-12. They are listed at the
end.

## How an entry is renewed

`scripts/verify_bandit_policy.py` enforces these rules for each entry in
`docs/bandit_triage.json`:

- `reviewed_on` is the review date, not after today;
- `expires_on` is no earlier than `reviewed_on` and at most 90 days after it;
- `owner` names who reviewed it (3 characters or more);
- `reason` says why (20 characters or more);
- `disposition` is `accepted_risk`, `false_positive` or `mitigated_by_control`;
- only medium findings can be triaged, never high.

An entry that no longer holds can't be renewed. Either change the code (a code
edit changes the fingerprint, so the new finding needs its own review), or
record why with a different disposition.

To check a renewal before pushing:

```bash
uv run bandit -r src/blueprint_pipeline scripts --format json --output bandit.json --quiet
uv run python scripts/verify_bandit_policy.py --bandit-report bandit.json --output bandit-policy-gate.json
```

## The 36 expired entries

Each item gives the current location, the disposition on file, the review date,
the first 12 hex digits of the fingerprint (to find the entry in
`docs/bandit_triage.json`), the flagged line, and the reason on file.

### B108 `hardcoded_tmp_directory` (18)

1. `src/blueprint_pipeline/asset_authoring_sandbox.py:174` · false_positive · reviewed 2026-10-06 · `438b6b7c15c6`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `'--perms', '0755', '--tmpfs', '/tmp']`
   - Reason on file: Both literal /tmp uses construct namespace-local paths: bwrap installs a fresh --tmpfs /tmp and unshares namespaces, or the privileged fallback unshares pid/ipc/uts/net then drops credentials/groups and sets no-new-privileges. System/admitted roots are read-only, only the admitted attempt root is writable, and environment/credentials are cleared. These strings do not create predictable files in host /tmp.
2. `src/blueprint_pipeline/asset_authoring_sandbox.py:181` · false_positive · reviewed 2026-10-06 · `ec620f3375a6`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `for parent in target.parents if str(parent) not in {'/', '/tmp', '/proc', '/dev'}}`
   - Reason on file: Both literal /tmp uses construct namespace-local paths: bwrap installs a fresh --tmpfs /tmp and unshares namespaces, or the privileged fallback unshares pid/ipc/uts/net then drops credentials/groups and sets no-new-privileges. System/admitted roots are read-only, only the admitted attempt root is writable, and environment/credentials are cleared. These strings do not create predictable files in host /tmp.
3. `src/blueprint_pipeline/company_policy_sandbox_executor.py:398` · false_positive · reviewed 2026-10-06 · `7f4dcbd86489`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `f'/tmp:rw,noexec,nosuid,nodev,size={resources["tmpfs_mib"]}m',`
   - Reason on file: The literal /tmp is a private container tmpfs with rw,noexec,nosuid,nodev and an admitted size cap. The executor requires exact validated plan equality, digest-bound image, runsc, read-only root, dropped capabilities, no-new-privileges, admitted UID/GID and bounded CPU/memory/PIDs with seccomp/AppArmor; host namespaces/mounts and open network are refused.
4. `src/blueprint_pipeline/company_policy_sandbox_v2.py:238` · false_positive · reviewed 2026-10-06 · `574819e9806f`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `f'/tmp:rw,noexec,nosuid,nodev,size={resources["tmpfs_mib"]}m',`
   - Reason on file: The plan encodes a private container /tmp tmpfs with noexec,nosuid,nodev and an admitted size bound, plus runsc/read-only root/capability drop/no-new-privileges/resource and namespace restrictions. Executor equality checks prevent widening this plan to a host mount.
5. `src/blueprint_pipeline/native_g1_team_archive_gpu.py:198` · mitigated_by_control · reviewed 2026-10-06 · `f07b0549c928`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `env={"PATH": "/usr/bin:/bin", "HOME": "/tmp", "LANG": "C.UTF-8"})`
   - Reason on file: The host environment is restricted to trusted bwrap launcher PATH/HOME/XDG/LANG, without inherited credentials or Python path overrides. Canonical constructor clears that environment and sets HOME/XDG inside a private tmpfs, with fixed UID/GID/capability/network/device/fd bindings checked by the GPU probe.
6. `src/blueprint_pipeline/native_g1_team_artifact_runtime.py:154` · false_positive · reviewed 2026-10-06 · `12613b5d7106`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `"--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",`
   - Reason on file: Literal /tmp paths are inside bwrap --tmpfs /tmp with --unshare-all, --clearenv, UID/GID 65534 and all capabilities dropped. The artifact is bound by SHA256, held read-only descriptors and bounded link-free unpacking; only minimal runtime paths are read-only bound and GPU access is explicit character-device binding. HOME/XDG are set in the private tmpfs after clearenv; the host wrapper environment does not reach artifact execution.
7. `src/blueprint_pipeline/native_g1_team_artifact_runtime.py:167` · false_positive · reviewed 2026-10-06 · `47ed176bc7b2`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `"--chdir", "/work", "--setenv", "HOME", "/tmp",`
   - Reason on file: Literal /tmp paths are inside bwrap --tmpfs /tmp with --unshare-all, --clearenv, UID/GID 65534 and all capabilities dropped. The artifact is bound by SHA256, held read-only descriptors and bounded link-free unpacking; only minimal runtime paths are read-only bound and GPU access is explicit character-device binding. HOME/XDG are set in the private tmpfs after clearenv; the host wrapper environment does not reach artifact execution.
8. `src/blueprint_pipeline/native_g1_team_artifact_runtime.py:168` · false_positive · reviewed 2026-10-06 · `9d24c5c7d20d`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `"--setenv", "XDG_CACHE_HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",`
   - Reason on file: Literal /tmp paths are inside bwrap --tmpfs /tmp with --unshare-all, --clearenv, UID/GID 65534 and all capabilities dropped. The artifact is bound by SHA256, held read-only descriptors and bounded link-free unpacking; only minimal runtime paths are read-only bound and GPU access is explicit character-device binding. HOME/XDG are set in the private tmpfs after clearenv; the host wrapper environment does not reach artifact execution.
9. `src/blueprint_pipeline/native_g1_team_artifact_runtime.py:291` · mitigated_by_control · reviewed 2026-10-06 · `8b9145e06fb6`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `"PATH": "/usr/bin:/bin", "HOME": "/tmp",`
   - Reason on file: The literals HOME=/tmp and XDG_CACHE_HOME=/tmp belong to the host launcher environment for the checked canonical bwrap command, not an already active namespace. Only fixed PATH/HOME/XDG_CACHE_HOME/LANG values are passed; inherited credentials and Python override variables are excluded. That canonical command then uses --unshare-all, --clearenv and a private tmpfs for HOME/XDG, fixed UID/GID65534, dropped capabilities, admitted network configuration and held read-only artifact descriptor and explicit revalidated character-device binding when GPU is admitted. SHA256/link-free bounded artifact admission precedes execution. The control excludes arbitrary direct commands.
10. `src/blueprint_pipeline/native_g1_team_artifact_runtime.py:292` · mitigated_by_control · reviewed 2026-10-06 · `a242cabbbed5`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `"XDG_CACHE_HOME": "/tmp", "LANG": "C.UTF-8",`
   - Reason on file: The literals HOME=/tmp and XDG_CACHE_HOME=/tmp belong to the host launcher environment for the checked canonical bwrap command, not an already active namespace. Only fixed PATH/HOME/XDG_CACHE_HOME/LANG values are passed; inherited credentials and Python override variables are excluded. That canonical command then uses --unshare-all, --clearenv and a private tmpfs for HOME/XDG, fixed UID/GID65534, dropped capabilities, admitted network configuration and held read-only artifact descriptor and explicit revalidated character-device binding when GPU is admitted. SHA256/link-free bounded artifact admission precedes execution. The control excludes arbitrary direct commands.
11. `src/blueprint_pipeline/native_g1_team_container_runtime.py:53` · false_positive · reviewed 2026-10-06 · `bc5069f20c02`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `"--tmpfs", "/tmp:rw,nosuid,nodev,size=1024m",`
   - Reason on file: The /tmp string is a private Docker tmpfs of 1024 MiB with nosuid,nodev, inside a read-only root and network-none container with all capabilities dropped, UID 65534, bounded PIDs/CPU/memory and an exact already-local pinned image.
12. `src/blueprint_pipeline/native_g1_team_vm_host.py:229` · mitigated_by_control · reviewed 2026-10-06 · `d14acd4adaf1`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `with tempfile.TemporaryDirectory(prefix="g1-vm-relay-", dir=str(Path("/tmp").resolve())) as private:`
   - Reason on file: TemporaryDirectory creates unique mode-0700 scratch with owned lifecycle cleanup. Relay _private_path validates canonical private owner/mode ancestry, refuses an existing socket, binds mode-0600 config/socket and compares socket identity before unlink so a replacement is preserved. Relay authority uses HMAC challenge/profile bindings, refuses replay, bounds frame/time and tears down watchdog/container; the relay mount is narrowly read-only.
13. `src/blueprint_pipeline/native_g1_team_vm_system_installation.py:105` · false_positive · reviewed 2026-10-06 · `71b84abafc11`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `"--dev", "/dev", "--tmpfs", "/tmp", "--dir", "/etc", "--ro-bind-fd", str(fd), "/work",`
   - Reason on file: Offline exact-host installation probes use bwrap --tmpfs /tmp with all namespaces unshared, cleared environment, UID/GID 65534 and a read-only held archive fd. The fixed Python -I -B -S probe and observed namespace checks require expected UID/GID, zero effective capabilities and only loopback interfaces.
14. `src/blueprint_pipeline/remote_cpu_worker.py:75` · mitigated_by_control · reviewed 2026-10-06 · `eca543542594`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `RELEASE_PARENT = "/tmp/blueprint-release"`
   - Reason on file: Canonical Cloud Run jobs have per-instance isolated filesystem, one pinned-image container, IaC task_count/parallelism one and max_retries zero. /var/lib uses a private instance-local volume; umask 077 is set. Runtime and output cache are not shared host /tmp. Authenticated transport binds archive generation/SHA256/size/Git header, extraction refuses links/traversal and preexisting release, handlers are fixed/registered and fresh subprocess stage scratch uses mkdtemp mode 0700 with bounded kill-session/owned cleanup.
15. `src/blueprint_pipeline/remote_cpu_worker.py:76` · mitigated_by_control · reviewed 2026-10-06 · `76c98b037fe2`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `PYCACHE_ROOT = "/tmp/blueprint-pycache"`
   - Reason on file: Canonical Cloud Run jobs have per-instance isolated filesystem, one pinned-image container, IaC task_count/parallelism one and max_retries zero. /var/lib uses a private instance-local volume; umask 077 is set. Runtime and output cache are not shared host /tmp. Authenticated transport binds archive generation/SHA256/size/Git header, extraction refuses links/traversal and preexisting release, handlers are fixed/registered and fresh subprocess stage scratch uses mkdtemp mode 0700 with bounded kill-session/owned cleanup.
16. `src/blueprint_pipeline/remote_cpu_worker.py:698` · mitigated_by_control · reviewed 2026-10-06 · `62cc89b3d0b2`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `runtime.local("/tmp").mkdir(parents=True, exist_ok=True)`
   - Reason on file: Canonical Cloud Run jobs have per-instance isolated filesystem, one pinned-image container, IaC task_count/parallelism one and max_retries zero. /var/lib uses a private instance-local volume; umask 077 is set. Runtime and output cache are not shared host /tmp. Authenticated transport binds archive generation/SHA256/size/Git header, extraction refuses links/traversal and preexisting release, handlers are fixed/registered and fresh subprocess stage scratch uses mkdtemp mode 0700 with bounded kill-session/owned cleanup.
17. `src/blueprint_pipeline/remote_cpu_worker.py:699` · mitigated_by_control · reviewed 2026-10-06 · `7b2684c56d28`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `scratch = Path(tempfile.mkdtemp(prefix="remote-cpu-stage-", dir=runtime.local("/tmp")))`
   - Reason on file: Canonical Cloud Run jobs have per-instance isolated filesystem, one pinned-image container, IaC task_count/parallelism one and max_retries zero. /var/lib uses a private instance-local volume; umask 077 is set. Runtime and output cache are not shared host /tmp. Authenticated transport binds archive generation/SHA256/size/Git header, extraction refuses links/traversal and preexisting release, handlers are fixed/registered and fresh subprocess stage scratch uses mkdtemp mode 0700 with bounded kill-session/owned cleanup.
18. `src/blueprint_pipeline/vast_provider_transfer_upload.py:13` · mitigated_by_control · reviewed 2026-10-06 · `0d169ea5ad6b`
   - Finding: Probable insecure usage of temp file/directory.
   - Code: `def provider_output_upload_shell_fragment(*, scratch_root: str = "/tmp") -> str:`
   - Reason on file: The default /tmp is only a scratch parent. Every attempt obtains a fresh mode0700 mktemp directory. Noclobber file creation atomically holds mode0600 status/body/empty receipt on original FD11-13 before transport. Receipt authority is successful bounded fixed curl return plus held source directory/file identity, never mutable receipt contents or later path open. Canonical readback validates identity and emits no file bytes. Cleanup closes only owned descriptors and clears active state, never path unlinks. Repeated calls release descriptors and use fresh roots. Same capability URL/file SHA256 binding, byte ceiling, three transient transport attempts,1200s/earlier parent deadline remain; no provider workload replay or new authority. Normally at most3status bytes and two empty files plus private directory/inode overhead retained per call until existing finite worker/session filesystem disposal. No isolation against malicious sameUID actor asserted. Namespace replacements are refused without reading/deleting foreign contents. SIGKILL cannot run shell cleanup but OS closes process descriptors. Existing upload_backend independent broader97+41 test review remains distinct evidence; these15 tests are a narrow independent aggregate recheck. No human approval, risk acceptance, actual host/deployment proof or waiver.

### B103 `set_bad_file_permissions` (11)

19. `src/blueprint_pipeline/control_plane_break_glass.py:200` · mitigated_by_control · reviewed 2026-10-06 · `ef15cc76d88f`
   - Finding: Chmod setting a permissive mask 0o755 on file (directory).
   - Code: `os.fchmod(directory, 0o755)`
   - Reason on file: scripts/install_live_pipeline_control_plane.sh installs the canonical STATE_DIR owner/group blueprint with mode 0750; child 0755/0644 does not grant other-user traversal through that ancestor. Break-glass uses no-follow receipt roots and exclusive no-follow files; GC publishes through a held no-follow directory with exclusive scratch and named-inode revalidation. Capacity systemd unit is root-owned with UMask=0077; full latest/history files rely on that private creation umask, while the summary is an explicit reduced projection. Scope: scoped to canonical protected state layout. Actual deployed state ancestry and blueprint group membership were not read; this is a source/deployment-contract review. Break-glass reason/operator fields, full GC paths, and full capacity budget metadata are not universally secret-free. Capacity direct CLI callers outside the unit do not get a forced 0600 full-report mode; arbitrary report roots need separate protection.
20. `src/blueprint_pipeline/control_plane_capacity_controller.py:871` · mitigated_by_control · reviewed 2026-10-06 · `7f5472946962`
   - Finding: Chmod setting a permissive mask 0o755 on file (report_root).
   - Code: `os.chmod(report_root, 0o755)`
   - Reason on file: scripts/install_live_pipeline_control_plane.sh installs the canonical STATE_DIR owner/group blueprint with mode 0750; child 0755/0644 does not grant other-user traversal through that ancestor. Break-glass uses no-follow receipt roots and exclusive no-follow files; GC publishes through a held no-follow directory with exclusive scratch and named-inode revalidation. Capacity systemd unit is root-owned with UMask=0077; full latest/history files rely on that private creation umask, while the summary is an explicit reduced projection. Scope: scoped to canonical protected state layout. Actual deployed state ancestry and blueprint group membership were not read; this is a source/deployment-contract review. Break-glass reason/operator fields, full GC paths, and full capacity budget metadata are not universally secret-free. Capacity direct CLI callers outside the unit do not get a forced 0600 full-report mode; arbitrary report roots need separate protection.
21. `src/blueprint_pipeline/control_plane_disk_footprints.py:106` · mitigated_by_control · reviewed 2026-10-06 · `a2a54108737e`
   - Finding: Chmod setting a permissive mask 0o2770 on file (history).
   - Code: `os.fchmod(history, 0o2770)`
   - Reason on file: Disk history opens its directory relative to a held no-follow ledger descriptor and files with no-follow, regular-only admission. The history directory inherits the canonical ledger group and setgid02770 for runtime-group coordination; mode repair uses its owned descriptor. Root/runtime group write serves explicit coordination roles. Actual deployed OS group membership was not observed; this is not a public write grant or a universal filesystem privacy claim.
22. `src/blueprint_pipeline/control_plane_disk_footprints.py:122` · mitigated_by_control · reviewed 2026-10-06 · `906cfa012cd5`
   - Finding: Chmod setting a permissive mask 0o660 on file (descriptor).
   - Code: `os.fchmod(descriptor, 0o660)`
   - Reason on file: Disk history opens its directory relative to a held no-follow ledger descriptor and files with no-follow, regular-only admission. The history directory inherits the canonical ledger group and setgid02770 for runtime-group coordination; mode repair uses its owned descriptor. Root/runtime group write serves explicit coordination roles. Actual deployed OS group membership was not observed; this is not a public write grant or a universal filesystem privacy claim.
23. `src/blueprint_pipeline/control_plane_disk_ledger.py:106` · mitigated_by_control · reviewed 2026-10-06 · `f38ba061ac85`
   - Finding: Chmod setting a permissive mask 0o2770 on file (descriptor).
   - Code: `os.fchmod(descriptor, 0o2770)`
   - Reason on file: The ledger refuses symlink locks and nonregular files. Mode repair uses the held descriptor only for the object owner; require_mode refuses locks which cannot meet0660. The runtime group write grant is intentional ledger coordination under the canonical role layout. Root/runtime group write serves explicit coordination roles. Actual deployed OS group membership was not observed; this is not a public write grant or a universal filesystem privacy claim.
24. `src/blueprint_pipeline/control_plane_registered_checkpoint_cache.py:1262` · mitigated_by_control · reviewed 2026-10-06 · `602bc3bd2621`
   - Finding: Chmod setting a permissive mask 0o750 on file (stage).
   - Code: `os.fchmod(stage, 0o750)`
   - Reason on file: The registered checkpoint cache resolves the canonical blueprint account/group and grants only exact root-authorized private reader authority. Initial staging is root-only; held/named identity and grant binding are verified through chown/chmod before publishing root:blueprint directories0750 and files0640. This is the explicit blueprint-reader cache role, with no extension of HTTP authentication into OS group authority. Actual deployed reader membership was not observed.
25. `src/blueprint_pipeline/control_plane_storage_gc.py:1715` · mitigated_by_control · reviewed 2026-10-06 · `f6edee8ff734`
   - Finding: Chmod setting a permissive mask 0o755 on file (parent_fd).
   - Code: `os.fchmod(parent_fd, 0o755)`
   - Reason on file: scripts/install_live_pipeline_control_plane.sh installs the canonical STATE_DIR owner/group blueprint with mode 0750; child 0755/0644 does not grant other-user traversal through that ancestor. Break-glass uses no-follow receipt roots and exclusive no-follow files; GC publishes through a held no-follow directory with exclusive scratch and named-inode revalidation. Capacity systemd unit is root-owned with UMask=0077; full latest/history files rely on that private creation umask, while the summary is an explicit reduced projection. Scope: scoped to canonical protected state layout. Actual deployed state ancestry and blueprint group membership were not read; this is a source/deployment-contract review. Break-glass reason/operator fields, full GC paths, and full capacity budget metadata are not universally secret-free. Capacity direct CLI callers outside the unit do not get a forced 0600 full-report mode; arbitrary report roots need separate protection.
26. `src/blueprint_pipeline/control_plane_storage_gc.py:1720` · mitigated_by_control · reviewed 2026-10-06 · `a45e845a74a2`
   - Finding: Chmod setting a permissive mask 0o755 on file (parent_fd).
   - Code: `os.fchmod(parent_fd, 0o755)`
   - Reason on file: scripts/install_live_pipeline_control_plane.sh installs the canonical STATE_DIR owner/group blueprint with mode 0750; child 0755/0644 does not grant other-user traversal through that ancestor. Break-glass uses no-follow receipt roots and exclusive no-follow files; GC publishes through a held no-follow directory with exclusive scratch and named-inode revalidation. Capacity systemd unit is root-owned with UMask=0077; full latest/history files rely on that private creation umask, while the summary is an explicit reduced projection. Scope: scoped to canonical protected state layout. Actual deployed state ancestry and blueprint group membership were not read; this is a source/deployment-contract review. Break-glass reason/operator fields, full GC paths, and full capacity budget metadata are not universally secret-free. Capacity direct CLI callers outside the unit do not get a forced 0600 full-report mode; arbitrary report roots need separate protection.
27. `src/blueprint_pipeline/control_plane_workspace_lock.py:128` · mitigated_by_control · reviewed 2026-10-06 · `a6ad6c24f220`
   - Finding: Chmod setting a permissive mask 0o770 on file (locks).
   - Code: `os.fchmod(locks, 0o770)`
   - Reason on file: Intentional runtime-group coordination. Held no-follow directory and regular single-link lock FDs; root/parent/effectiveUID or runtime-group database ownership admission, sameGID modes770/660, named/held identities rechecked around shared producer and nonblocking exclusive GC flock. Existing unsafe locks are refused, reclaim creates no new proof, stable coordination inode never unlinked. This is not a claim that every ancestor is root-only or non-group-writable. No observed host group membership, enrollment or exact operating-system deployment state claim. No claim that manager ingress/firewall/IP allowlist has been deployed or independently observed. No accepted risk, policy waiver or human approval. Source mitigations do not prove CI or live runtime deployment.
28. `src/blueprint_pipeline/control_plane_workspace_lock.py:151` · mitigated_by_control · reviewed 2026-10-06 · `6a5cc082222e`
   - Finding: Chmod setting a permissive mask 0o660 on file (fd).
   - Code: `os.fchmod(fd, 0o660)`
   - Reason on file: Intentional runtime-group coordination. Held no-follow directory and regular single-link lock FDs; root/parent/effectiveUID or runtime-group database ownership admission, sameGID modes770/660, named/held identities rechecked around shared producer and nonblocking exclusive GC flock. Existing unsafe locks are refused, reclaim creates no new proof, stable coordination inode never unlinked. This is not a claim that every ancestor is root-only or non-group-writable. No observed host group membership, enrollment or exact operating-system deployment state claim. No claim that manager ingress/firewall/IP allowlist has been deployed or independently observed. No accepted risk, policy waiver or human approval. Source mitigations do not prove CI or live runtime deployment.
29. `src/blueprint_pipeline/task_evaluation_result_artifact_store.py:214` · mitigated_by_control · reviewed 2026-10-06 · `2924bd09212a`
   - Finding: Chmod setting a permissive mask 0o660 on file (descriptor).
   - Code: `os.fchmod(descriptor, 0o660)`
   - Reason on file: Artifact-store lease admission binds the registered root and safe artifact path. Shared download leases prevent simultaneous exclusive eviction; existing lease permissions are repaired only by their owner. The runtime-group write grant coordinates admitted producers, consumers and reclaimers. Root/runtime group write serves explicit coordination roles. Actual deployed OS group membership was not observed; this is not a public write grant or a universal filesystem privacy claim.

### B102 `exec_used` (4)

30. `scripts/scene_retirement_continuous_bootstrap.py:248` · mitigated_by_control · reviewed 2026-10-06 · `df02c152bd07`
   - Finding: Use of exec detected.
   - Code: `exec(compile(raw, str(source), 'exec', dont_inherit=True), module.__dict__)`
   - Reason on file: Fixed seven-member _CORE source cohort; fixed CURRENT roots; root-owned ancestry without group/world write; regular single-link bounded files read through O_NOFOLLOW descriptors with identity/version rechecks; source bytes retained before compile/exec; native isolated -I -S startup and no preloaded package substitution. The call executes protected installed source rather than caller-provided text.
31. `src/blueprint_pipeline/control_plane_lane_experiment_retirement.py:73` · mitigated_by_control · reviewed 2026-10-06 · `e8e9f6f2f8db`
   - Finding: Use of exec detected.
   - Code: `exec(compile(source, module.__file__, "exec"), module.__dict__)`
   - Reason on file: The executable configuration path is fixed under /opt/blueprint/operator-door, with no repository fallback. The shared _Files.read(protected=True) validates root-owned non-writable ancestry, regular bounded single-link files, O_NOFOLLOW held descriptors and identity/version proof before and after compile/exec; policy, enablement and budget are parsed from the protected exact cohort.
32. `src/blueprint_pipeline/control_plane_lane_owner_consents.py:466` · mitigated_by_control · reviewed 2026-10-06 · `7e5bb5e9393e`
   - Finding: Use of exec detected.
   - Code: `exec(compile(source, module.__file__, 'exec'), module.__dict__)`
   - Reason on file: Fixed protected operator-door config path; root-owned non-writable ancestry, regular single-link bounded source, O_NOFOLLOW descriptors and before/after identity/version proof through shared _Files; no repository fallback. Source execution supplies protected installed policy, while consent payloads remain parsed data and exact owner/budget/enablement validation remains mandatory.
33. `src/blueprint_pipeline/task_evaluation_scene_retirement_supervisor.py:949` · mitigated_by_control · reviewed 2026-10-06 · `827f8a9783a4`
   - Finding: Use of exec detected.
   - Code: `exec(compile(self.row['raw'], self.row['path'], 'exec', dont_inherit=True), module.__dict__)`
   - Reason on file: A fixed catalogue is recovered using AST literals from protected engine source; exact consumer cohort and SHA256 policy rows are bounded in count, per-file and total bytes and read from held protected roots. _BoundSourceFinder binds protected dependency source/extensions/namespaces, refuses stale bytecode and preloaded-core substitution, and retains source under startup fencing and credential drop.

### B104 `hardcoded_bind_all_interfaces` (2)

34. `scripts/serve_company_policy_sandbox_bridge.py:121` · mitigated_by_control · reviewed 2026-10-06 · `0fa80e68cff9`
   - Finding: Possible binding to all interfaces.
   - Code: `parser.add_argument("--bind-host", default="0.0.0.0")`
   - Reason on file: Intentional listener secured by TLS>=1.2 and unchanged constant-time bearer/closed-route/job/source/qualification checks. Accept thread no longer blocks on TLS; 8 total TLS+handler slots,5s handshake limit, one absolute10s header/body read deadline, cancellation and slot cleanup, no progress renewal. No observed host group membership, enrollment or exact operating-system deployment state claim. No claim that manager ingress/firewall/IP allowlist has been deployed or independently observed. No accepted risk, policy waiver or human approval. Source mitigations do not prove CI or live runtime deployment.
35. `src/blueprint_pipeline/company_policy_sandbox_manager.py:374` · mitigated_by_control · reviewed 2026-10-06 · `720fa7aa1567`
   - Finding: Possible binding to all interfaces.
   - Code: `server = BoundedPolicyHTTPSServer((str(settings.get("bind_host", "0.0.0.0")),`
   - Reason on file: Intentional listener secured by TLS>=1.2 and unchanged constant-time bearer/closed-route/job/source/qualification checks. Accept thread no longer blocks on TLS; 8 total TLS+handler slots,5s handshake limit, one absolute10s header/body read deadline, cancellation and slot cleanup, no progress renewal. No observed host group membership, enrollment or exact operating-system deployment state claim. No claim that manager ingress/firewall/IP allowlist has been deployed or independently observed. No accepted risk, policy waiver or human approval. Source mitigations do not prove CI or live runtime deployment.

### B608 `hardcoded_sql_expressions` (1)

36. `src/blueprint_pipeline/agent_execution/journal.py:205` · false_positive · reviewed 2026-10-06 · `5378e079354f`
   - Finding: Possible SQL injection vector through string-based query construction.
   - Code: `f"SELECT task_id FROM tasks {where} ORDER BY updated_at,task_id LIMIT ?",`
   - Reason on file: AgentJournal.tasks interpolates only one of two closed constant WHERE fragments. The LIMIT is bound with a SQLite parameter after 1..1000 validation; external strings never become SQL fragments. Scope: supported. No dedicated repository test of every LIMIT boundary was found; the included hermetic SQLite probe supplies this review evidence, without claiming nonexistent coverage.

## The 60 entries that expire on 2026-11-12

| Location | Test | Disposition | Reviewed |
|---|---|---|---|
| `src/blueprint_pipeline/production_preflight_summary.py:250` | B103 | mitigated_by_control | 2026-10-28 |
| `src/blueprint_pipeline/production_preflight_summary.py:254` | B103 | mitigated_by_control | 2026-10-28 |
| `src/blueprint_pipeline/privacy_runner_service.py:297` | B104 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/video_to_world_runner_service.py:270` | B104 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/mujoco_g1_simulator_command.py:690` | B108 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/object_index_stage.py:641` | B108 | false_positive | 2026-08-14 |
| `src/blueprint_pipeline/object_index_stage.py:642` | B108 | false_positive | 2026-08-14 |
| `src/blueprint_pipeline/object_index_stage.py:643` | B108 | false_positive | 2026-08-14 |
| `src/blueprint_pipeline/robot_eval_worker.py:1572` | B108 | accepted_risk | 2026-08-14 |
| `scripts/run_beta_intake_soak_test.py:57` | B310 | accepted_risk | 2026-08-14 |
| `scripts/run_sim_only_beta_deployment_parity_proof.py:112` | B310 | accepted_risk | 2026-08-14 |
| `scripts/run_sim_only_beta_local_gate.py:215` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/capture_enrichment_llm.py:338` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/city_launch_autonomy_harness.py:905` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/consent_takedown.py:622` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/g1_microwave_finetune_provider_job.py:420` | B310 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/g1_microwave_finetune_provider_job.py:468` | B310 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/g1_microwave_finetune_provider_job.py:751` | B310 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/g1_microwave_finetune_provider_job.py:818` | B310 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/isaac_particlefield_render_job.py:551` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/isaac_particlefield_render_job.py:621` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/live_pipeline_setup.py:454` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/mujoco_g1_wam_vla_policy_endpoint_eval.py:936` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/mujoco_g1_wam_vla_policy_endpoint_eval.py:6823` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/palatial_physready.py:729` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/palatial_physready.py:829` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/privacy_processing.py:152` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/privacy_service_runtime.py:178` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/provider_preview.py:52` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/provider_preview.py:78` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/provider_preview.py:107` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/provider_worker_policy_command_adapter.py:180` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/robot_eval_worker.py:974` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/runpod_live_execution_proof.py:190` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/single_g1_kitchen_episode_runpod.py:587` | B310 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/single_g1_kitchen_episode_runpod.py:632` | B310 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/single_g1_kitchen_episode_runpod.py:647` | B310 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/site_world_runtime_service_client.py:67` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/unitree_unifolm_runpod_server.py:665` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/unitree_unifolm_vla_server_bridge.py:213` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/vast_provider_adapter.py:3621` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/vast_provider_adapter.py:3649` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/vast_provider_adapter.py:3762` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/video_to_world_client.py:76` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/wam_compute_providers.py:338` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/wam_compute_providers.py:351` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/webapp_sync.py:662` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/webapp_sync.py:898` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/worldlabs_asset_materialization.py:202` | B310 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/mujoco_g1_wam_vla_policy_endpoint_eval.py:7428` | B614 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/official_g1_policy_handoff.py:1945` | B614 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/unitree_g1_policy_execution.py:104` | B614 | accepted_risk | 2026-08-14 |
| `src/blueprint_pipeline/openvla_policy_command_adapter.py:617` | B615 | false_positive | 2026-08-14 |
| `src/blueprint_pipeline/openvla_policy_command_adapter.py:631` | B615 | false_positive | 2026-08-14 |
| `src/blueprint_pipeline/oscar_runtime_asset_contract.py:636` | B615 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/oscar_runtime_asset_contract.py:646` | B615 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/oscar_runtime_asset_contract.py:794` | B615 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/wam_real_provider_validation_probe.py:536` | B615 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/wam_real_provider_validation_probe.py:537` | B615 | mitigated_by_control | 2026-08-14 |
| `src/blueprint_pipeline/wam_real_provider_validation_probe.py:913` | B615 | mitigated_by_control | 2026-08-14 |
