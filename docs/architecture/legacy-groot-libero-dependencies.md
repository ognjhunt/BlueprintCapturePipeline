# Historical GR00T LIBERO GPU dependencies

The unused `groot-libero` project extra was removed from the product resolver in
October 2026 because LeRobot 0.4.4's `groot` extra requires `Pillow>=10.0.0,<13.0.0`.
The published fixes for the current Pillow advisories require Pillow 13.0.0,
which also raises the product's Python floor to 3.11. Production remains on 3.12.

For compatibility history, the removed definition was:

```toml
groot-libero = [
    "torch>=2.2.0",
    "torchvision>=0.17.0",
    "lerobot[groot]",
]
```

This historical GPU environment is security-blocked and is not an admitted
installation path. Reintroducing it requires an upstream dependency release
that supports the patched image decoder and a separately frozen, audited
runtime closure. Do not override the LeRobot metadata or weaken the dependency
audit to install it.

The used `groot-libero-cpu` extra retains base LeRobot 0.4.4 and its existing
CPU dependency declarations. Frozen DROID policy identities, worker locks,
weights, runtime admissions and the two-candidate ADP protocol are unchanged.

Sources: [Pillow 13 release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html),
[LeRobot 0.4.4 metadata](https://pypi.org/pypi/lerobot/0.4.4/json), and
[Certifi security advisory](https://github.com/certifi/python-certifi/security/advisories/GHSA-6q48-6c7w-9gw2).
