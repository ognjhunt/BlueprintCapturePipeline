# CAD execution in provider containers

This is the ADP-009B replacement-creation boundary needed for ADP-009D's day-14
policy rehearsal and day-21 sealed results. Instance 51019374 failed its early
CAD preflight because ordinary container root lacks the namespace privileges
required by the original bubblewrap path. Training had not started.

`SandboxedAssetRunner.preflight()` first tries the existing OS backend. On
Linux, an unavailable namespace sandbox can select Landlock plus seccomp.
Both paths must successfully execute the no-op probe before any generated code
or CAD model reservation. There is no unsandboxed fallback. macOS continues to use
`sandbox-exec`.

The container backend requires Linux Landlock and `libseccomp.so.2`; the provider
bootstrap explicitly supplies system Python and libseccomp. A fresh isolated
stdlib launcher applies no-new-privileges, clears capabilities, installs the
filesystem and syscall policies, then executes the requested program with only
the existing approved environment. No threaded-parent `preexec_fn` is used.

- File contents are readable only from approved runtime/source roots and narrow
  public system paths. Persistent writes are limited to the authoring directory.
  Root inode identities are rechecked when rules are installed.
- Seccomp defaults to denial. Network sockets, connection/binding operations,
  other-process memory/FD inspection, outside-process signals, namespace changes, and process-
  group escape are excluded. Only private connected Unix stream socketpairs
  are permitted for local IPC.
- Filesystem metadata mutations such as chmod, ownership, xattrs, and timestamps
  are denied, including inside the output directory. Metadata queries are not
  equivalent to file-content access and are not universally hidden, including
  existence metadata for known /proc paths. PID-taking query syscalls are
  restricted to the current process.
- Linux 5.15 / Landlock ABI1 is supported: kernel rules deny cross-directory
  reparenting; syscall restrictions block path truncation, openat2, and
  read-only opens with O_TRUNC. Writable opens remain governed by WRITE_FILE.
  Newer ABIs also enforce native REFER/TRUNCATE rights.
- Only null stdin and output pipes reach candidates. Output forwarding happens
  in the trusted parent. Descendants inherit restrictions and are cleaned up
  by process group on completion or timeout. Shared global `/dev/shm` is not
  writable; libraries may choose serial operation.

Preflight records backend outcomes and stderr outside the candidate-writable
root. The early production check also imports the real CAD closure, constructs
an OpenCascade box, and saves/renders a small Blender scene inside the sandbox.
These are synthetic runtime probes, never replacement assets or physical proof.

Verification includes real Docker tests with default capabilities/seccomp,
ABI1 compatibility rules, denied file/network/process operations, and actual
STEP export plus the sealed Blender 5.2.1 save/render. A new provider still has
to pass its live preflight; unsupported kernels refuse before training.

The [Linux Landlock API documentation](https://docs.kernel.org/userspace-api/landlock.html)
defines versioned filesystem rights and their inheritance. Seccomp compensates
for the older ABI's truncation gap; unsupported features are not silently
accepted as protection.

A second live preflight on instance 51035345 identified a distinct DAC boundary:
clearing capabilities made `/isaac-sim/kit/python` inaccessible. Trusted setup
now prepares the explicitly declared runtime-code roots before isolation. It adds
only the read/search bits the existing UID needs without DAC capabilities;
contents, ownership and write permissions stay unchanged. Root paths are canonicalized; symlinks within those trees are not
followed, broad system roots are not recursively broadened, and changes are
recorded in `runtime_code_access.json`. Candidate execution still has zero
capabilities and the same Landlock/seccomp restrictions. The Docker regression
first reproduces refusal on a foreign-owned private Python tree, then proves
file access and executable startup while outside reads remain denied.

The already-authorized long-running rehearsal also exhausted its original 32
source-attempt slots. The append-only owner budget-extension API permits an
explicit grant up to 64 attempts; initial intake still permits at most 32.
Historical attempts and exposure are never refunded, per-action limits are
unchanged, and the cumulative dollar ceiling remains $1,000. No extra attempts
are granted merely by deploying the code.
