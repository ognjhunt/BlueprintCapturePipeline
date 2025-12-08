"""File I/O utilities for pipeline jobs."""
from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np


def ensure_local_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to create.

    Returns:
        The same path, for chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def temp_workspace(prefix: str = "blueprint_", cleanup: bool = True) -> Iterator[Path]:
    """Create a temporary workspace directory.

    Args:
        prefix: Prefix for the temp directory name.
        cleanup: Whether to delete the directory on exit.

    Yields:
        Path to the temporary directory.
    """
    workspace = Path(tempfile.mkdtemp(prefix=prefix))
    try:
        yield workspace
    finally:
        if cleanup and workspace.exists():
            shutil.rmtree(workspace)


def compute_checksum(path: Path, algorithm: str = "sha256") -> str:
    """Compute checksum of a file.

    Args:
        path: File path.
        algorithm: Hash algorithm ('md5', 'sha256', etc.).

    Returns:
        Hex-encoded checksum string.
    """
    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_json(data: Union[Dict, List], path: Path, indent: int = 2) -> Path:
    """Save data as JSON file.

    Args:
        data: Data to serialize.
        path: Output path.
        indent: JSON indentation (0 for compact).

    Returns:
        Path to saved file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent if indent > 0 else None, default=_json_serializer)
    return path


def load_json(path: Path) -> Union[Dict, List]:
    """Load data from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Loaded data.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for special types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def save_image(
    image: "np.ndarray",
    path: Path,
    quality: int = 95,
) -> Path:
    """Save an image array to file.

    Args:
        image: Image array (H, W, C) in RGB or (H, W) for grayscale.
        path: Output path (format inferred from extension).
        quality: JPEG quality (1-100).

    Returns:
        Path to saved image.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("Pillow is required for image I/O. Install with: pip install Pillow")

    path.parent.mkdir(parents=True, exist_ok=True)

    # Handle different array formats
    if image.dtype == np.float32 or image.dtype == np.float64:
        # Assume 0-1 range for float
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    elif image.dtype != np.uint8:
        image = image.astype(np.uint8)

    pil_image = PILImage.fromarray(image)

    # Save with appropriate settings
    save_kwargs = {}
    suffix = path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        save_kwargs["quality"] = quality
        save_kwargs["optimize"] = True
    elif suffix == ".png":
        save_kwargs["compress_level"] = 6

    pil_image.save(path, **save_kwargs)
    return path


def load_image(
    path: Path,
    mode: str = "RGB",
) -> "np.ndarray":
    """Load an image file as numpy array.

    Args:
        path: Path to image file.
        mode: PIL mode ('RGB', 'RGBA', 'L' for grayscale).

    Returns:
        Image array (H, W, C) or (H, W) for grayscale.
    """
    try:
        from PIL import Image as PILImage
    except ImportError:
        raise ImportError("Pillow is required for image I/O. Install with: pip install Pillow")

    with PILImage.open(path) as img:
        if img.mode != mode:
            img = img.convert(mode)
        return np.array(img)


def save_numpy(array: "np.ndarray", path: Path, compressed: bool = True) -> Path:
    """Save numpy array to file.

    Args:
        array: Array to save.
        path: Output path (.npy or .npz).
        compressed: Use compression for .npz files.

    Returns:
        Path to saved file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.suffix == ".npz":
        if compressed:
            np.savez_compressed(path, data=array)
        else:
            np.savez(path, data=array)
    else:
        np.save(path, array)

    return path


def load_numpy(path: Path) -> "np.ndarray":
    """Load numpy array from file.

    Args:
        path: Path to .npy or .npz file.

    Returns:
        Loaded array.
    """
    if path.suffix == ".npz":
        with np.load(path) as data:
            return data["data"]
    return np.load(path)


def iter_frames_from_directory(
    directory: Path,
    pattern: str = "*.png",
    sort: bool = True,
) -> Iterator[tuple[int, Path]]:
    """Iterate over frame files in a directory.

    Args:
        directory: Directory containing frames.
        pattern: Glob pattern for frame files.
        sort: Sort frames by name.

    Yields:
        Tuples of (frame_index, frame_path).
    """
    files = list(directory.glob(pattern))
    if sort:
        files = sorted(files, key=lambda p: p.stem)

    for idx, path in enumerate(files):
        yield idx, path


def get_frame_count(directory: Path, pattern: str = "*.png") -> int:
    """Count frames in a directory.

    Args:
        directory: Directory containing frames.
        pattern: Glob pattern for frame files.

    Returns:
        Number of frames.
    """
    return len(list(directory.glob(pattern)))


class FrameWriter:
    """Helper for writing numbered frames to a directory."""

    def __init__(
        self,
        output_dir: Path,
        prefix: str = "frame",
        extension: str = "png",
        digits: int = 6,
    ):
        """Initialize frame writer.

        Args:
            output_dir: Directory to write frames.
            prefix: Filename prefix.
            extension: File extension.
            digits: Number of digits for frame numbering.
        """
        self.output_dir = ensure_local_dir(output_dir)
        self.prefix = prefix
        self.extension = extension.lstrip(".")
        self.digits = digits
        self._counter = 0

    def write(self, image: "np.ndarray", index: Optional[int] = None) -> Path:
        """Write a frame image.

        Args:
            image: Image array.
            index: Frame index (auto-increments if not specified).

        Returns:
            Path to written frame.
        """
        if index is None:
            index = self._counter
            self._counter += 1

        filename = f"{self.prefix}_{index:0{self.digits}d}.{self.extension}"
        path = self.output_dir / filename
        return save_image(image, path)

    def reset_counter(self, value: int = 0):
        """Reset the frame counter."""
        self._counter = value
