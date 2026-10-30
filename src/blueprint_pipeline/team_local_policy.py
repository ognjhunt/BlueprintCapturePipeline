"""Explicit local-team runtime dispatch; no dynamic module or route loading."""
from __future__ import annotations


def runtime_for_interface(interface: str):
    if interface == "openwam_piper_eef20.v1":
        from . import openwam_piper_runtime
        return openwam_piper_runtime
    if interface == "molmoact2_yam_joint14.v1":
        from . import molmoact2_yam_runtime
        return molmoact2_yam_runtime
    raise ValueError("team_local_policy_interface_unsupported")
