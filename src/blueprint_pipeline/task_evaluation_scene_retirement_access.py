"""Compatibility admission facade and authentic producer birth delegation.

The producer-independent leaf owns all lifetime state. Attribute overrides at
this existing API reach that same state, including descriptor fault hooks;
module metadata stays here for exact protected-source verification.
"""
from __future__ import annotations

import sys
from types import ModuleType

from . import task_evaluation_scene_retirement_lifetime as _lifetime

# Retain the names even during a delete/restore fault injection. No copied
# policy globals or function implementations can diverge from the leaf.
_LIFETIME_NAMES = frozenset(name for name in vars(_lifetime) if not name.startswith('__'))


class _AdmissionFacade(ModuleType):
    def __getattr__(self, name):
        if name in _LIFETIME_NAMES:
            return getattr(_lifetime, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name in _LIFETIME_NAMES:
            setattr(_lifetime, name, value)
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in _LIFETIME_NAMES:
            delattr(_lifetime, name)
        else:
            super().__delattr__(name)

    def __dir__(self):
        return sorted(set(super().__dir__()) | _LIFETIME_NAMES)


sys.modules[__name__].__class__ = _AdmissionFacade


def birth_scene_member(path, *, owner_intent_id, owner_raw_ref, birth_request_raw_ref, now=None):
    from .task_evaluation_scene_retirement_generations import birth_member
    return birth_member(path, owner_intent_id=owner_intent_id, owner_raw_ref=owner_raw_ref,
                        birth_request_raw_ref=birth_request_raw_ref, now=now)
