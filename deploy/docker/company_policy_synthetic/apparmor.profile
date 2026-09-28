# Synthetic worker confinement; all writable paths remain Docker tmpfs/IPC.
#include <tunables/global>
profile blueprint-policy-apparmor-v2 flags=(attach_disconnected,mediate_deleted) {
  #include <abstractions/base>
  network,
  capability,
  file,
  umount,
  deny mount,
  deny /proc/sys/** w,
  deny /proc/sysrq-trigger rwklx,
  deny /sys/** wklx,
}
