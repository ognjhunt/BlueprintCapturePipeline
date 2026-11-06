"""Import a policy client's pinned packages only from its isolated install.

Inside Isaac's Kit app, putting the pinned install first on ``sys.path`` does
not decide which package imports: Kit resolves its bundled Python packages with
a fast importer that does not consult ``sys.path`` (Kit
``/app/extensions/fastImporter/enabled``, on by default), and the Kit worker
kept Isaac's older websockets while the same install passed in the bare
``python.sh`` provisioning process. This finder runs before every other hook
and is the only source for the pinned top-level names; a module missing from
the install fails to import instead of falling back to the host's copy.
"""
from __future__ import annotations

import importlib
from importlib.machinery import ModuleSpec, PathFinder
from pathlib import Path
import sys
from typing import Iterable, Sequence


class PinnedClientFinder:
    def __init__(self, root: Path, names: Iterable[str]):
        self.root = Path(root).resolve()
        self.names = frozenset(names)

    def find_spec(self, fullname: str, path: Sequence[str] | None = None,
                  target: object = None) -> ModuleSpec | None:
        if fullname.partition(".")[0] not in self.names:
            return None
        search = [str(self.root)] if "." not in fullname else [str(entry) for entry in path or ()]
        spec = None
        if search and all(Path(entry).resolve().is_relative_to(self.root) for entry in search):
            spec = PathFinder.find_spec(fullname, search)
        if spec is None:
            raise ModuleNotFoundError(f"No module named {fullname!r} in the pinned client at {self.root}",
                                      name=fullname)
        return spec


def activate_pinned_packages(root: Path, names: Iterable[str]) -> None:
    """Make ``root`` the only source of ``names`` in this process, then forget cached copies."""
    finder = PinnedClientFinder(root, names)
    sys.meta_path[:] = [hook for hook in sys.meta_path
                        if not (isinstance(hook, PinnedClientFinder) and hook.names & finder.names)]
    sys.meta_path.insert(0, finder)
    entry = str(root)
    sys.path[:] = [item for item in sys.path if item != entry]
    sys.path.insert(0, entry)
    for name in list(sys.modules):
        if name.partition(".")[0] in finder.names:
            del sys.modules[name]
    importlib.invalidate_caches()
