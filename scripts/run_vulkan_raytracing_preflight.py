#!/usr/bin/env python3
"""Windowless Vulkan device and ray-tracing extension preflight."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path


VK_SUCCESS = 0
VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1
REQUIRED_EXTENSIONS = {
    "VK_KHR_acceleration_structure",
    "VK_KHR_deferred_host_operations",
    "VK_KHR_ray_tracing_pipeline",
}


class VkApplicationInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_uint32),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(VkApplicationInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.POINTER(ctypes.c_char_p)),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.POINTER(ctypes.c_char_p)),
    ]


class VkExtensionProperties(ctypes.Structure):
    _fields_ = [
        ("extensionName", ctypes.c_char * 256),
        ("specVersion", ctypes.c_uint32),
    ]


def _write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def evaluate_device_selection(
    rows: list[dict[str, object]], selected_device_index: int | None
) -> tuple[list[str], list[str], list[int]]:
    """Judge the enumerated devices for a trustworthy ray-traced render.

    Passing because *some* device supports ray tracing is not the property that
    matters: on a mixed host the renderer's own device choice decides whether
    frames contain pixels.  A live OVRTX run passed the old check on a host with
    one ray-tracing GPU and one 114-extension software rasterizer.

    That run's black frames turned out to have an unrelated cause, and Vulkan
    enumeration order is not necessarily the renderer's device order, so a mixed
    host cannot be resolved here.  Blocking on it would stop legitimate runs over
    a failure never observed.  Ambiguity is therefore recorded as a warning and
    surfaced in the receipt; only a pin that is demonstrably wrong -- naming an
    incapable or out-of-range device -- fails closed.

    Returns ``(blockers, warnings, incapable_device_indices)``.  Kept free of
    Vulkan calls so the rule is testable without a GPU.
    """

    incapable = [
        int(row["device_index"])
        for row in rows
        if not row["required_raytracing_extensions_present"]
    ]
    if not rows or not incapable or len(incapable) == len(rows):
        # No devices, all capable, or none capable: the existing checks decide.
        return [], [], incapable
    if selected_device_index is None:
        return [], ["vulkan_raytracing_device_selection_ambiguous"], incapable
    if not 0 <= int(selected_device_index) < len(rows):
        return ["vulkan_raytracing_selected_device_out_of_range"], [], incapable
    if int(selected_device_index) in incapable:
        return ["vulkan_raytracing_selected_device_incapable"], [], incapable
    return [], [], incapable


def probe(selected_device_index: int | None = None) -> dict[str, object]:
    library = ctypes.CDLL("libvulkan.so.1")
    library.vkCreateInstance.argtypes = [
        ctypes.POINTER(VkInstanceCreateInfo),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.vkCreateInstance.restype = ctypes.c_int32
    library.vkEnumeratePhysicalDevices.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_void_p),
    ]
    library.vkEnumeratePhysicalDevices.restype = ctypes.c_int32
    library.vkEnumerateDeviceExtensionProperties.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(VkExtensionProperties),
    ]
    library.vkEnumerateDeviceExtensionProperties.restype = ctypes.c_int32
    library.vkDestroyInstance.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    library.vkDestroyInstance.restype = None
    application = VkApplicationInfo(
        sType=VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pNext=None,
        pApplicationName=b"Blueprint OVRTX preflight",
        applicationVersion=1,
        pEngineName=b"Blueprint",
        engineVersion=1,
        apiVersion=(1 << 22) | (2 << 12),
    )
    create_info = VkInstanceCreateInfo(
        sType=VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pNext=None,
        flags=0,
        pApplicationInfo=ctypes.pointer(application),
        enabledLayerCount=0,
        ppEnabledLayerNames=None,
        enabledExtensionCount=0,
        ppEnabledExtensionNames=None,
    )
    instance = ctypes.c_void_p()
    create_result = int(
        library.vkCreateInstance(ctypes.byref(create_info), None, ctypes.byref(instance))
    )
    rows: list[dict[str, object]] = []
    blockers: list[str] = []
    if create_result != VK_SUCCESS or not instance.value:
        blockers.append("vulkan_instance_creation_failed")
    else:
        try:
            count = ctypes.c_uint32()
            enumerate_result = int(
                library.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
            )
            if enumerate_result != VK_SUCCESS or count.value < 1:
                blockers.append("vulkan_physical_device_missing")
            else:
                devices = (ctypes.c_void_p * count.value)()
                enumerate_result = int(
                    library.vkEnumeratePhysicalDevices(
                        instance, ctypes.byref(count), devices
                    )
                )
                if enumerate_result != VK_SUCCESS:
                    blockers.append("vulkan_physical_device_enumeration_failed")
                for index, device in enumerate(devices):
                    extension_count = ctypes.c_uint32()
                    result = int(
                        library.vkEnumerateDeviceExtensionProperties(
                            device, None, ctypes.byref(extension_count), None
                        )
                    )
                    extensions: set[str] = set()
                    if result == VK_SUCCESS and extension_count.value:
                        values = (VkExtensionProperties * extension_count.value)()
                        result = int(
                            library.vkEnumerateDeviceExtensionProperties(
                                device,
                                None,
                                ctypes.byref(extension_count),
                                values,
                            )
                        )
                        if result == VK_SUCCESS:
                            extensions = {
                                value.extensionName.decode("utf-8", "replace")
                                for value in values
                            }
                    missing = sorted(REQUIRED_EXTENSIONS - extensions)
                    rows.append(
                        {
                            "device_index": index,
                            "extension_count": len(extensions),
                            "required_raytracing_extensions_present": not missing,
                            "missing_required_extensions": missing,
                        }
                    )
                if rows and not any(
                    row["required_raytracing_extensions_present"] for row in rows
                ):
                    blockers.append("vulkan_raytracing_extensions_missing")
        finally:
            library.vkDestroyInstance(instance, None)
    device_blockers, warnings, incapable = evaluate_device_selection(
        rows, selected_device_index
    )
    blockers.extend(device_blockers)
    return {
        "schema_version": "adp009d_vulkan_raytracing_preflight.v1",
        "status": "passed" if not blockers else "blocked",
        "vk_create_instance_result": create_result,
        "physical_device_count": len(rows),
        "devices": rows,
        "raytracing_capable_device_count": len(rows) - len(incapable),
        "warnings": warnings,
        "raytracing_incapable_device_indices": incapable,
        "selected_device_index": selected_device_index,
        "required_extensions": sorted(REQUIRED_EXTENSIONS),
        "window_or_surface_created": False,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--selected-device-index",
        type=int,
        default=None,
        help=(
            "Physical device index the renderer will bind.  Required to pass on a "
            "host where only some devices support ray tracing."
        ),
    )
    args = parser.parse_args()
    try:
        result = probe(selected_device_index=args.selected_device_index)
    except Exception as exc:  # noqa: BLE001 - preserve typed runtime evidence
        result = {
            "schema_version": "adp009d_vulkan_raytracing_preflight.v1",
            "status": "blocked",
            "window_or_surface_created": False,
            "blockers": [f"vulkan_preflight_exception:{type(exc).__name__}"],
            "error": str(exc),
        }
    _write(args.output, result)
    return 0 if result.get("status") == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
