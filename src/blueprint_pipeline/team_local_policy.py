"""Explicit local-team runtime dispatch; no dynamic module or route loading."""
from __future__ import annotations


def runtime_for_interface(interface: str):
    if interface == "pi05_trlc_joint14.v1":
        from . import trlc_pi05_runtime
        return trlc_pi05_runtime
    if interface == "openwam_franka_eef10.v1":
        from . import openwam_franka_runtime
        return openwam_franka_runtime
    if interface == "openwam_piper_eef20.v1":
        from . import openwam_piper_runtime
        return openwam_piper_runtime
    if interface == "molmoact2_yam_joint14.v1":
        from . import molmoact2_yam_runtime
        return molmoact2_yam_runtime
    raise ValueError("team_local_policy_interface_unsupported")


def openwam_bundle_for_interface(interface: str):
    if interface == "openwam_piper_eef20.v1":
        from . import openwam_piper_bundle
        return openwam_piper_bundle
    if interface == "openwam_franka_eef10.v1":
        from . import openwam_franka_bundle
        return openwam_franka_bundle
    raise ValueError("openwam_policy_interface_unsupported")
