"""Compatibility admission facade and authentic producer birth delegation.

The producer-independent leaf owns all lifetime state. Attribute overrides at
this existing API reach that same state, including descriptor fault hooks, and
an override's undo restores it; module metadata stays here for exact
protected-source verification.
"""
from __future__ import annotations

import sys
from types import ModuleType

from . import task_evaluation_scene_retirement_lifetime as _lifetime

# Retain the names even during a delete/restore fault injection. No copied
# policy globals or function implementations can diverge from the leaf.
_LIFETIME_NAMES = frozenset(name for name in vars(_lifetime) if not name.startswith('__'))


class _Forwarded:
    """One lifetime name on the facade's type: read, written and deleted in the leaf.

    A patcher that records a module's own ``__dict__`` entry (pytest 9.2's
    monkeypatch) found none here and undid an override by deleting the leaf's
    value. Through a data descriptor it records the leaf's value and restores it.
    """

    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name

    def __get__(self, module, owner=None):
        return self if module is None else getattr(_lifetime, self.name)

    def __set__(self, module, value):
        setattr(_lifetime, self.name, value)

    def __delete__(self, module):
        delattr(_lifetime, self.name)


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


def _forward_lifetime_names():
    # Once every name of this module is bound: a name it defines itself stays its own.
    for name in _LIFETIME_NAMES - set(vars(sys.modules[__name__])):
        setattr(_AdmissionFacade, name, _Forwarded(name))


_forward_lifetime_names()
