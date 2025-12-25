"""Point cloud utilities for 3D reconstruction.

This module provides:
- PLY file I/O
- COLMAP data loading
- Point cloud processing utilities
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


@dataclass
class PointCloud:
    """Point cloud with optional colors and normals."""
    points: np.ndarray  # [N, 3]
    colors: Optional[np.ndarray] = None  # [N, 3]
    normals: Optional[np.ndarray] = None  # [N, 3]

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    def __len__(self) -> int:
        return self.num_points


def load_ply(path: str | Path) -> Dict[str, np.ndarray]:
    """Load PLY file and return data as dictionary.

    Supports both ASCII and binary PLY formats.
    Handles 3DGS PLY files with spherical harmonics.

    Returns:
        Dictionary with keys like 'xyz', 'colors', 'f_dc', 'f_rest', 'opacity', etc.
    """
    path = Path(path)

    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("utf-8").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        # Parse header info
        is_binary = False
        is_little_endian = True
        num_vertices = 0
        properties = []

        for line in header_lines:
            if line.startswith("format"):
                if "binary_little_endian" in line:
                    is_binary = True
                    is_little_endian = True
                elif "binary_big_endian" in line:
                    is_binary = True
                    is_little_endian = False
                else:
                    is_binary = False

            elif line.startswith("element vertex"):
                num_vertices = int(line.split()[-1])

            elif line.startswith("property"):
                parts = line.split()
                prop_type = parts[1]
                prop_name = parts[2]
                properties.append((prop_name, prop_type))

        # Build numpy dtype
        type_map = {
            "float": np.float32,
            "float32": np.float32,
            "double": np.float64,
            "float64": np.float64,
            "int": np.int32,
            "int32": np.int32,
            "uint": np.uint32,
            "uint32": np.uint32,
            "uchar": np.uint8,
            "uint8": np.uint8,
            "char": np.int8,
            "int8": np.int8,
            "short": np.int16,
            "int16": np.int16,
            "ushort": np.uint16,
            "uint16": np.uint16,
        }

        dtype_list = []
        for prop_name, prop_type in properties:
            np_type = type_map.get(prop_type, np.float32)
            dtype_list.append((prop_name, np_type))

        dtype = np.dtype(dtype_list)

        # Read data
        if is_binary:
            if not is_little_endian:
                dtype = dtype.newbyteorder(">")
            data = np.frombuffer(f.read(num_vertices * dtype.itemsize), dtype=dtype)
        else:
            # ASCII format
            data = np.loadtxt(f, dtype=dtype, max_rows=num_vertices)

    # Organize output
    result = {}

    # Positions
    if all(p in data.dtype.names for p in ["x", "y", "z"]):
        result["xyz"] = np.column_stack([data["x"], data["y"], data["z"]])

    # Colors (0-1 range)
    if all(p in data.dtype.names for p in ["red", "green", "blue"]):
        colors = np.column_stack([data["red"], data["green"], data["blue"]])
        # Normalize if 0-255 range
        if colors.max() > 1.0:
            colors = colors / 255.0
        result["colors"] = colors.astype(np.float32)

    # Normals
    if all(p in data.dtype.names for p in ["nx", "ny", "nz"]):
        result["normals"] = np.column_stack([data["nx"], data["ny"], data["nz"]])

    # 3DGS specific: SH coefficients
    if all(p in data.dtype.names for p in ["f_dc_0", "f_dc_1", "f_dc_2"]):
        result["f_dc"] = np.column_stack([
            data["f_dc_0"], data["f_dc_1"], data["f_dc_2"]
        ])

    # Rest of SH coefficients
    f_rest_names = [n for n in data.dtype.names if n.startswith("f_rest_")]
    if f_rest_names:
        f_rest_names = sorted(f_rest_names, key=lambda x: int(x.split("_")[-1]))
        f_rest = np.column_stack([data[n] for n in f_rest_names])
        # Reshape to [N, num_sh_per_channel, 3]
        num_sh = len(f_rest_names) // 3
        if num_sh > 0:
            result["f_rest"] = f_rest.reshape(-1, num_sh, 3)

    # Opacity
    if "opacity" in data.dtype.names:
        result["opacity"] = data["opacity"].astype(np.float32)

    # Scales
    if all(p in data.dtype.names for p in ["scale_0", "scale_1", "scale_2"]):
        result["scales"] = np.column_stack([
            data["scale_0"], data["scale_1"], data["scale_2"]
        ]).astype(np.float32)

    # Rotations (quaternion)
    if all(p in data.dtype.names for p in ["rot_0", "rot_1", "rot_2", "rot_3"]):
        result["rotations"] = np.column_stack([
            data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"]
        ]).astype(np.float32)

    return result


def save_ply(
    path: str | Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    normals: Optional[np.ndarray] = None,
    binary: bool = True,
) -> None:
    """Save point cloud to PLY file.

    Args:
        path: Output file path
        points: Point positions [N, 3]
        colors: Optional RGB colors [N, 3] in 0-1 or 0-255 range
        normals: Optional normals [N, 3]
        binary: Whether to use binary format
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    num_points = points.shape[0]

    # Normalize colors to 0-255 range
    if colors is not None:
        colors = np.asarray(colors)
        if colors.max() <= 1.0:
            colors = (colors * 255).astype(np.uint8)
        else:
            colors = colors.astype(np.uint8)

    # Build header
    header = "ply\n"
    if binary:
        header += "format binary_little_endian 1.0\n"
    else:
        header += "format ascii 1.0\n"

    header += f"element vertex {num_points}\n"
    header += "property float x\n"
    header += "property float y\n"
    header += "property float z\n"

    if normals is not None:
        header += "property float nx\n"
        header += "property float ny\n"
        header += "property float nz\n"

    if colors is not None:
        header += "property uchar red\n"
        header += "property uchar green\n"
        header += "property uchar blue\n"

    header += "end_header\n"

    # Write file
    with open(path, "wb") as f:
        f.write(header.encode("utf-8"))

        if binary:
            for i in range(num_points):
                f.write(struct.pack("<fff", *points[i]))
                if normals is not None:
                    f.write(struct.pack("<fff", *normals[i]))
                if colors is not None:
                    f.write(struct.pack("<BBB", *colors[i]))
        else:
            for i in range(num_points):
                line = f"{points[i, 0]:.6f} {points[i, 1]:.6f} {points[i, 2]:.6f}"
                if normals is not None:
                    line += f" {normals[i, 0]:.6f} {normals[i, 1]:.6f} {normals[i, 2]:.6f}"
                if colors is not None:
                    line += f" {colors[i, 0]} {colors[i, 1]} {colors[i, 2]}"
                line += "\n"
                f.write(line.encode("utf-8"))


# =============================================================================
# COLMAP Data Loading
# =============================================================================

def read_cameras_text(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP cameras.txt file."""
    cameras = {}

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            camera_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(p) for p in parts[4:]]

            # Parse camera model
            if model == "SIMPLE_PINHOLE":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "PINHOLE":
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            elif model == "SIMPLE_RADIAL":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
                # Ignore distortion for now
            elif model == "RADIAL":
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model == "OPENCV":
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                # Default to simple parsing
                fx = fy = params[0] if len(params) > 0 else width
                cx = params[1] if len(params) > 1 else width / 2
                cy = params[2] if len(params) > 2 else height / 2

            cameras[camera_id] = {
                "model": model,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "params": params,
            }

    return cameras


def read_cameras_binary(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP cameras.bin file."""
    cameras = {}

    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_cameras):
            camera_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]

            # Model-specific parameter counts
            model_params = {
                0: 3,  # SIMPLE_PINHOLE
                1: 4,  # PINHOLE
                2: 4,  # SIMPLE_RADIAL
                3: 5,  # RADIAL
                4: 8,  # OPENCV
            }
            num_params = model_params.get(model_id, 4)
            params = struct.unpack("<" + "d" * num_params, f.read(8 * num_params))

            # Parse based on model
            if model_id == 0:  # SIMPLE_PINHOLE
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model_id == 1:  # PINHOLE
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                fx = fy = params[0]
                cx, cy = params[1] if len(params) > 1 else width / 2, params[2] if len(params) > 2 else height / 2

            cameras[camera_id] = {
                "model_id": model_id,
                "width": width,
                "height": height,
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "params": params,
            }

    return cameras


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])


def read_images_text(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP images.txt file."""
    images = {}

    with open(path, "r") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 10:
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        camera_id = int(parts[8])
        name = parts[9]

        # Skip point observations line
        if i < len(lines):
            i += 1

        qvec = np.array([qw, qx, qy, qz])
        R = qvec2rotmat(qvec)
        t = np.array([tx, ty, tz])

        images[image_id] = {
            "camera_id": camera_id,
            "name": name,
            "qvec": qvec,
            "rotation": R,
            "translation": t,
        }

    return images


def read_images_binary(path: Path) -> Dict[int, Dict[str, Any]]:
    """Read COLMAP images.bin file."""
    images = {}

    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_images):
            image_id = struct.unpack("<I", f.read(4))[0]
            qvec = np.array(struct.unpack("<dddd", f.read(32)))
            tvec = np.array(struct.unpack("<ddd", f.read(24)))
            camera_id = struct.unpack("<I", f.read(4))[0]

            # Read name (null-terminated)
            name = b""
            while True:
                char = f.read(1)
                if char == b"\x00":
                    break
                name += char
            name = name.decode("utf-8")

            # Skip 2D points
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(24 * num_points2D)  # Skip x, y, point3D_id

            R = qvec2rotmat(qvec)

            images[image_id] = {
                "camera_id": camera_id,
                "name": name,
                "qvec": qvec,
                "rotation": R,
                "translation": tvec,
            }

    return images


def read_points3D_text(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.txt file."""
    points = []
    colors = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            r, g, b = int(parts[4]), int(parts[5]), int(parts[6])

            points.append([x, y, z])
            colors.append([r / 255.0, g / 255.0, b / 255.0])

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def read_points3D_binary(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Read COLMAP points3D.bin file."""
    points = []
    colors = []

    with open(path, "rb") as f:
        num_points = struct.unpack("<Q", f.read(8))[0]

        for _ in range(num_points):
            point_id = struct.unpack("<Q", f.read(8))[0]
            xyz = struct.unpack("<ddd", f.read(24))
            rgb = struct.unpack("<BBB", f.read(3))
            error = struct.unpack("<d", f.read(8))[0]

            # Skip track
            track_length = struct.unpack("<Q", f.read(8))[0]
            f.read(8 * track_length)  # Skip image_id, point2D_idx pairs

            points.append(xyz)
            colors.append([c / 255.0 for c in rgb])

    return np.array(points, dtype=np.float32), np.array(colors, dtype=np.float32)


def initialize_from_colmap(
    colmap_path: str | Path
) -> Tuple[np.ndarray, np.ndarray, Dict, Dict]:
    """Initialize point cloud and cameras from COLMAP output.

    Args:
        colmap_path: Path to COLMAP sparse reconstruction (containing cameras.bin/txt, etc.)

    Returns:
        Tuple of (points, colors, cameras, images)
    """
    colmap_path = Path(colmap_path)

    # Find the sparse model directory
    if (colmap_path / "sparse" / "0").exists():
        model_path = colmap_path / "sparse" / "0"
    elif (colmap_path / "sparse").exists():
        model_path = colmap_path / "sparse"
    elif (colmap_path / "0").exists():
        model_path = colmap_path / "0"
    else:
        model_path = colmap_path

    # Load cameras
    if (model_path / "cameras.bin").exists():
        cameras = read_cameras_binary(model_path / "cameras.bin")
    elif (model_path / "cameras.txt").exists():
        cameras = read_cameras_text(model_path / "cameras.txt")
    else:
        raise FileNotFoundError(f"No cameras file found in {model_path}")

    # Load images
    if (model_path / "images.bin").exists():
        images = read_images_binary(model_path / "images.bin")
    elif (model_path / "images.txt").exists():
        images = read_images_text(model_path / "images.txt")
    else:
        raise FileNotFoundError(f"No images file found in {model_path}")

    # Load points
    if (model_path / "points3D.bin").exists():
        points, colors = read_points3D_binary(model_path / "points3D.bin")
    elif (model_path / "points3D.txt").exists():
        points, colors = read_points3D_text(model_path / "points3D.txt")
    else:
        # Create empty point cloud
        points = np.zeros((0, 3), dtype=np.float32)
        colors = np.zeros((0, 3), dtype=np.float32)

    return points, colors, cameras, images


def random_point_cloud(
    num_points: int,
    bounds_min: Tuple[float, float, float] = (-1, -1, -1),
    bounds_max: Tuple[float, float, float] = (1, 1, 1),
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate random point cloud for testing.

    Args:
        num_points: Number of points
        bounds_min: Minimum bounds
        bounds_max: Maximum bounds

    Returns:
        Tuple of (points, colors)
    """
    points = np.random.uniform(
        low=bounds_min,
        high=bounds_max,
        size=(num_points, 3)
    ).astype(np.float32)

    colors = np.random.uniform(0, 1, size=(num_points, 3)).astype(np.float32)

    return points, colors
