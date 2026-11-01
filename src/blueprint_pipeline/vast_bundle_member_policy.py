"""Which provider bundle members the Vast preflight parses as Blueprint JSON."""

from __future__ import annotations

TOOLCHAIN_COMPONENTS_PREFIX = "provider_runtime/toolchain/components/"


def vendored_toolchain_runtime_member(member: str) -> bool:
    """A third-party file inside a toolchain component's vendored runtime tree.

    Those bytes (Blender's USD ``plugInfo.json`` files carry ``#`` comments,
    site-packages ship fixtures) are not Blueprint JSON: the component's own
    inventory pins every byte, and the provider runtime verifies it there. The
    component's manifests at ``package/`` and ``package/runtime/`` are still
    parsed (capture-447ae4f4, 2026-11-01).
    """
    if not member.startswith(TOOLCHAIN_COMPONENTS_PREFIX):
        return False
    parts = member[len(TOOLCHAIN_COMPONENTS_PREFIX):].split("/")
    return len(parts) > 4 and parts[1] == "package" and parts[2] == "runtime"


__all__ = ["TOOLCHAIN_COMPONENTS_PREFIX", "vendored_toolchain_runtime_member"]
