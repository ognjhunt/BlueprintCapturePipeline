"""Build context generator for the reusable OSCAR WAM GPU image."""

from __future__ import annotations

import argparse
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .common import ensure_dir, utc_now_iso, write_json
from .oscar_wam_provider_bundle import DEFAULT_OSCAR_SOURCE_URL


OSCAR_WAM_GPU_IMAGE_SCHEMA_VERSION = "oscar_wam_gpu_image_context.v1"
DEFAULT_BASE_IMAGE = "nvidia/cuda:12.8.0-devel-ubuntu22.04"
DEFAULT_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
DEFAULT_TORCH_VERSION = "2.10.0"
DEFAULT_TORCHVISION_VERSION = "0.25.0"
DEFAULT_CUDNN_PACKAGE = "nvidia-cudnn-cu12>=9.10"
DEFAULT_OSCAR_SOURCE_REF = "main"
DEFAULT_TRANSFORMER_ENGINE_MODE = "shim"
DEFAULT_PLATFORM = "linux/amd64"
TRANSFORMER_ENGINE_MODES = ("shim", "real")
IMAGE_REF_ENV = "BLUEPRINT_OSCAR_WAM_GPU_IMAGE_REF"
LEGACY_IMAGE_REF_ENV = "BLUEPRINT_WAM_PROVIDER_IMAGE_REF"
DEFAULT_CONTEXT_FILENAME = "Dockerfile.oscar-wam-gpu"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _image_ref_is_versioned(image_ref: str) -> bool:
    if not image_ref or image_ref.endswith(":latest"):
        return False
    last = image_ref.rsplit("/", maxsplit=1)[-1]
    return ":" in last or "@" in last


def _secret_file_status(env_name: str, default_path: str) -> dict[str, Any]:
    configured = _string(os.getenv(env_name))
    path = Path(configured or default_path).expanduser()
    mode = oct(path.stat().st_mode & 0o777) if path.exists() else None
    return {
        "env_name": env_name,
        "path": str(path),
        "configured_by_env": bool(configured),
        "present": path.is_file(),
        "mode": mode,
        "mode_is_0600": mode == "0o600",
        "raw_secret_value_recorded": False,
        "secret_hash_recorded": False,
    }


def requirements_text() -> str:
    packages = [
        "accelerate>=0.30.0",
        "attrs>=23.0.0",
        "av>=15",
        "boto3>=1.34.0",
        "botocore>=1.34.0",
        "decord==0.6.0",
        "diffusers==0.35.2",
        "einops==0.8.1",
        "ffmpegcv>=0.3.15",
        "ftfy>=6.2.0",
        "fvcore>=0.1.5.post20221221",
        "hydra-core>=1.3.2",
        "huggingface_hub>=0.23.0",
        "imageio>=2.34.0",
        "imageio-ffmpeg>=0.5.1",
        "loguru>=0.7.2",
        "matplotlib>=3.8.0",
        "megatron-core>=0.14.0",
        "numpy>=1.24.0,<2.3",
        "omegaconf>=2.3.0",
        "onnx>=1.16.0",
        "onnxscript>=0.1.0",
        "opencv-python-headless>=4.10.0",
        "nvidia-ml-py>=12.560.30",
        "pandas>=2.0.0",
        "peft>=0.11.0",
        "Pillow>=10.0.0",
        "pytest>=8.0.0",
        "pytz>=2024.1",
        "qwen-vl-utils>=0.0.8",
        "safetensors>=0.4.3",
        "setuptools>=70.0.0",
        "termcolor>=2.4.0",
        "timm>=1.0.0",
        "torchmetrics>=1.4.0",
        "transformers>=4.45,<5",
        "wandb>=0.17.0",
        "webdataset>=0.2.86",
    ]
    return "\n".join(packages) + "\n"


def filter_requirements_script_text() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
skip = {
    "torch",
    "torchvision",
    "torchaudio",
    "transformer-engine",
    "transformer_engine",
}
lines: list[str] = []
if source.is_file():
    for line in source.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            lines.append(line)
            continue
        match = re.match(r"([A-Za-z0-9_.-]+)", stripped)
        package = (match.group(1) if match else "").lower().replace("_", "-")
        if package in skip:
            continue
        lines.append(line)
target.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
'''


def transformer_engine_shim_script_text() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import textwrap
import sys
from pathlib import Path

source_root = Path(sys.argv[1])
shim_root = source_root / "transformer_engine"
files = {
    shim_root / "__init__.py": """
from . import common
from . import pytorch

BLUEPRINT_COMPAT_SHIM = True
""",
    shim_root / "common" / "__init__.py": """
from . import recipe

BLUEPRINT_COMPAT_SHIM = True
""",
    shim_root / "common" / "recipe.py": """
from __future__ import annotations

import enum
from typing import Any

BLUEPRINT_COMPAT_SHIM = True


class Format(enum.Enum):
    E4M3 = "e4m3"
    HYBRID = "hybrid"


class _Recipe:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs


class DelayedScaling(_Recipe):
    pass


class Float8CurrentScaling(_Recipe):
    pass


class Float8BlockScaling(_Recipe):
    pass


class MXFP8BlockScaling(_Recipe):
    pass


class NVFP4BlockScaling(_Recipe):
    pass


class CustomRecipe(_Recipe):
    pass
""",
    shim_root / "pytorch" / "__init__.py": """
import torch

from . import distributed, ops
from .attention import DotProductAttention, apply_rotary_pos_emb
from .fp8 import FP8GlobalStateManager, fp8_autocast, fp8_model_init, quantized_model_init
from .float8_tensor import Float8Tensor
from .tensor import QuantizedTensor

BLUEPRINT_COMPAT_SHIM = True
RMSNorm = torch.nn.RMSNorm


class Linear(torch.nn.Linear):
    def __init__(self, in_features: int, out_features: int, *args, bias: bool = True, return_bias: bool = False, **kwargs) -> None:
        del args
        init_method = kwargs.pop("init_method", None)
        self.return_bias = bool(return_bias)
        self.use_bias = bool(bias)
        self.parallel_mode = kwargs.pop("parallel_mode", None)
        self.tp_size = int(kwargs.pop("tp_size", 1) or 1)
        kwargs.pop("sequence_parallel", None)
        kwargs.pop("fuse_wgrad_accumulation", None)
        kwargs.pop("tp_group", None)
        kwargs.pop("get_rng_state_tracker", None)
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        if callable(init_method):
            init_method(self.weight)

    def forward(self, input, *args, **kwargs):
        del args, kwargs
        out = super().forward(input)
        return (out, self.bias) if self.return_bias else out

    def set_tensor_parallel_group(self, *args, **kwargs) -> None:
        del args, kwargs
        return None

    def backward_dw(self) -> None:
        return None


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, hidden_size=None, normalized_shape=None, *args, eps: float = 1e-5, **kwargs) -> None:
        del args, kwargs
        super().__init__(normalized_shape or hidden_size, eps=eps)


class LayerNormLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *args, eps: float = 1e-5, bias: bool = True, return_bias: bool = False, **kwargs) -> None:
        super().__init__()
        del args
        init_method = kwargs.pop("init_method", None)
        kwargs.pop("sequence_parallel", None)
        kwargs.pop("fuse_wgrad_accumulation", None)
        kwargs.pop("tp_group", None)
        kwargs.pop("tp_size", None)
        kwargs.pop("get_rng_state_tracker", None)
        kwargs.pop("parallel_mode", None)
        kwargs.pop("return_layernorm_output", None)
        kwargs.pop("zero_centered_gamma", None)
        kwargs.pop("normalization", None)
        self.return_bias = bool(return_bias)
        self.use_bias = bool(bias)
        self.in_features = in_features
        self.out_features = out_features
        self.tp_size = 1
        self.layer_norm = torch.nn.LayerNorm(in_features, eps=eps)
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)
        self.weight = self.linear.weight
        self.bias = self.linear.bias
        if callable(init_method):
            init_method(self.linear.weight)

    def forward(self, input, *args, **kwargs):
        del args, kwargs
        out = self.linear(self.layer_norm(input))
        return (out, self.bias) if self.return_bias else out

    def set_tensor_parallel_group(self, *args, **kwargs) -> None:
        del args, kwargs
        return None

    def backward_dw(self) -> None:
        return None


class GroupedLinear(Linear):
    pass
""",
    shim_root / "pytorch" / "ops" / "__init__.py": """
from __future__ import annotations

from typing import Any

import torch

BLUEPRINT_COMPAT_SHIM = True


class FusibleOperation(torch.nn.Module):
    def forward(self, x, *args: Any, **kwargs: Any):
        del args, kwargs
        return x


class Sequential(torch.nn.Sequential):
    pass


class _Activation(FusibleOperation):
    fn = staticmethod(lambda x: x)

    def forward(self, x, *args: Any, **kwargs: Any):
        del args, kwargs
        return self.fn(x)


class GELU(_Activation):
    fn = staticmethod(torch.nn.functional.gelu)


class ReLU(_Activation):
    fn = staticmethod(torch.nn.functional.relu)


class SiLU(_Activation):
    fn = staticmethod(torch.nn.functional.silu)


class SwiGLU(FusibleOperation):
    pass


class GEGLU(FusibleOperation):
    pass


class ReGLU(FusibleOperation):
    pass


class LayerNorm(torch.nn.LayerNorm):
    def __init__(self, norm_shape, *args: Any, eps: float = 1e-5, **kwargs: Any) -> None:
        del args, kwargs
        super().__init__(norm_shape, eps=eps)


class RMSNorm(torch.nn.RMSNorm):
    pass


class BasicLinear(torch.nn.Linear):
    pass


class Bias(FusibleOperation):
    def __init__(self, size: int, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        device = kwargs.get("device", None)
        dtype = kwargs.get("dtype", None)
        self.bias = torch.nn.Parameter(torch.zeros(size, device=device, dtype=dtype))

    def forward(self, x, *args: Any, **kwargs: Any):
        del args, kwargs
        return x + self.bias


class ReduceScatter(FusibleOperation):
    pass


class AllReduce(FusibleOperation):
    pass
""",
    shim_root / "pytorch" / "distributed" / "__init__.py": """
from __future__ import annotations

from typing import Any

BLUEPRINT_COMPAT_SHIM = True


def activation_recompute_forward(*args: Any, **kwargs: Any):
    del kwargs
    if args and callable(args[0]):
        return args[0](*args[1:])
    return None


def get_all_rng_states(*args: Any, **kwargs: Any) -> dict:
    del args, kwargs
    return {}


def checkpoint(function, *args: Any, **kwargs: Any):
    return function(*args, **kwargs)


class CudaRNGStatesTracker:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def add(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    def fork(self, *args: Any, **kwargs: Any):
        del args, kwargs
        from contextlib import nullcontext

        return nullcontext()
""",
    shim_root / "pytorch" / "module" / "__init__.py": """
BLUEPRINT_COMPAT_SHIM = True
""",
    shim_root / "pytorch" / "module" / "base.py": """
from __future__ import annotations

import torch

BLUEPRINT_COMPAT_SHIM = True


class TransformerEngineBaseModule(torch.nn.Module):
    pass


def get_dummy_wgrad(*args, **kwargs):
    del args, kwargs
    return None


def get_workspace(*args, **kwargs):
    del args, kwargs
    return None
""",
    shim_root / "pytorch" / "fp8.py": """
from __future__ import annotations

from contextlib import nullcontext
from typing import Any

BLUEPRINT_COMPAT_SHIM = True


def fp8_autocast(*args: Any, **kwargs: Any):
    del args, kwargs
    return nullcontext()


def fp8_model_init(*args: Any, **kwargs: Any):
    del args, kwargs
    return nullcontext()


def quantized_model_init(*args: Any, **kwargs: Any):
    del args, kwargs
    return nullcontext()


class FP8GlobalStateManager:
    @staticmethod
    def is_fp8_enabled() -> bool:
        return False

    @staticmethod
    def get_fp8_recipe() -> None:
        return None

    @staticmethod
    def get_fp8_group() -> None:
        return None

    @staticmethod
    def is_first_fp8_module() -> bool:
        return False

    @staticmethod
    def add_fp8_tensors_to_global_buffer(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    @staticmethod
    def reduce_and_update_fp8_tensors(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None

    @staticmethod
    def set_skip_fp8_weight_update_tensor(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        return None
""",
    shim_root / "pytorch" / "float8_tensor.py": """
from .tensor.float8_tensor import Float8Tensor

BLUEPRINT_COMPAT_SHIM = True
""",
    shim_root / "pytorch" / "tensor" / "__init__.py": """
from __future__ import annotations

from typing import Any

import torch

BLUEPRINT_COMPAT_SHIM = True


class QuantizedTensor:
    def __init__(self, data: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        self._data = data

    def dequantize(self) -> torch.Tensor:
        if self._data is None:
            raise RuntimeError("QuantizedTensor shim has no backing tensor")
        return self._data

    def from_float8(self) -> torch.Tensor:
        return self.dequantize()


from .float8_tensor import Float8Tensor
from .mxfp8_tensor import MXFP8Tensor

__all__ = ["Float8Tensor", "MXFP8Tensor", "QuantizedTensor"]
""",
    shim_root / "pytorch" / "tensor" / "float8_tensor.py": """
from __future__ import annotations

from typing import Any

import torch

from . import QuantizedTensor

BLUEPRINT_COMPAT_SHIM = True


class Float8Tensor(QuantizedTensor):
    @classmethod
    def make_like(cls, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        return tensor
""",
    shim_root / "pytorch" / "tensor" / "float8_blockwise_tensor.py": """
from .float8_tensor import Float8Tensor

BLUEPRINT_COMPAT_SHIM = True


class Float8BlockwiseQTensor(Float8Tensor):
    pass
""",
    shim_root / "pytorch" / "tensor" / "mxfp8_tensor.py": """
from .float8_tensor import Float8Tensor

BLUEPRINT_COMPAT_SHIM = True


class MXFP8Tensor(Float8Tensor):
    pass
""",
    shim_root / "pytorch" / "tensor" / "utils.py": """
from __future__ import annotations

from typing import Any

import torch

BLUEPRINT_COMPAT_SHIM = True


def replace_raw_data(fp8_tensor: Any, new_raw_data: torch.Tensor) -> None:
    if hasattr(fp8_tensor, "_data"):
        fp8_tensor._data = new_raw_data


def cast_master_weights_to_fp8(*args: Any, **kwargs: Any) -> None:
    del args, kwargs
    return None


def post_all_gather_processing(tensor: Any, *args: Any, **kwargs: Any) -> Any:
    del args, kwargs
    return tensor
""",
    shim_root / "pytorch" / "attention" / "__init__.py": """
from __future__ import annotations

from typing import Any

import torch
from torch import nn

BLUEPRINT_COMPAT_SHIM = True


def _flatten_bshd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[2] * tensor.shape[3])


class DotProductAttention(nn.Module):
    def __init__(
        self,
        num_attention_heads: int | None = None,
        kv_channels: int | None = None,
        *,
        attention_dropout: float = 0.0,
        qkv_format: str = "bshd",
        **_: Any,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.kv_channels = kv_channels
        self.attention_dropout = float(attention_dropout or 0.0)
        self.qkv_format = qkv_format

    def set_context_parallel_group(self, *_: Any, **__: Any) -> None:
        return None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
            raise ValueError("DotProductAttention shim expects q/k/v rank-4 tensors")
        if self.qkv_format == "sbhd":
            q_bhsd = query.permute(1, 2, 0, 3)
            k_bhsd = key.permute(1, 2, 0, 3)
            v_bhsd = value.permute(1, 2, 0, 3)
            out = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd,
                k_bhsd,
                v_bhsd,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=False,
            )
            return out.permute(2, 0, 1, 3).reshape(query.shape[0], query.shape[1], -1)
        q_bhsd = query.permute(0, 2, 1, 3)
        k_bhsd = key.permute(0, 2, 1, 3)
        v_bhsd = value.permute(0, 2, 1, 3)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_bhsd,
            k_bhsd,
            v_bhsd,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        return _flatten_bshd(out.permute(0, 2, 1, 3).contiguous())


def apply_rotary_pos_emb(
    tensor: torch.Tensor,
    freqs: torch.Tensor,
    *,
    tensor_format: str = "bshd",
    fused: bool = True,
) -> torch.Tensor:
    del fused
    if tensor.dim() != 4:
        raise ValueError("apply_rotary_pos_emb shim expects a rank-4 tensor")
    half = tensor.shape[-1] // 2
    freqs = freqs.to(device=tensor.device, dtype=torch.float32)
    if freqs.shape[-1] >= tensor.shape[-1]:
        freqs = freqs[..., :half]
    if freqs.shape[-1] != half:
        raise ValueError(f"rotary freqs last dim {freqs.shape[-1]} does not match half head dim {half}")
    while freqs.dim() > 2 and freqs.shape[1] == 1:
        freqs = freqs.squeeze(1)
    while freqs.dim() > 2 and freqs.shape[-2] == 1:
        freqs = freqs.squeeze(-2)
    if freqs.dim() != 2:
        freqs = freqs.reshape(freqs.shape[0], half)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    if tensor_format == "sbhd":
        cos = cos[:, None, None, :]
        sin = sin[:, None, None, :]
    else:
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    even = tensor[..., 0::2].to(torch.float32)
    odd = tensor[..., 1::2].to(torch.float32)
    rotated_even = even * cos - odd * sin
    rotated_odd = even * sin + odd * cos
    return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2).to(tensor.dtype)
""",
    shim_root / "pytorch" / "attention" / "rope.py": """
from . import apply_rotary_pos_emb

BLUEPRINT_COMPAT_SHIM = True
""",
}
for path, text in files.items():
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(text).lstrip(), encoding="utf-8")
print("BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_SHIM_WRITTEN")
'''


def image_healthcheck_text() -> str:
    return r'''#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import site
import sys
from pathlib import Path
from typing import Any


def _candidate_cudnn_headers() -> list[Path]:
    paths: list[Path] = []
    for env_name in ("CUDNN_PATH", "CUDNN_HOME"):
        configured = os.environ.get(env_name)
        if configured:
            paths.append(Path(configured) / "include" / "cudnn.h")
            paths.append(Path(configured) / "cudnn.h")
    for base in site.getsitepackages():
        paths.append(Path(base) / "nvidia" / "cudnn" / "include" / "cudnn.h")
    paths.append(Path("/usr/include/cudnn.h"))
    paths.append(Path("/usr/local/cuda/include/cudnn.h"))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--build-time", action="store_true")
    args = parser.parse_args()
    blockers: list[str] = []
    payload: dict[str, Any] = {
        "schema_version": "oscar_wam_gpu_image_healthcheck.v1",
        "python": sys.version.split()[0],
        "build_time": args.build_time,
        "require_cuda": args.require_cuda,
        "raw_secret_values_recorded": False,
    }
    try:
        import torch

        payload["torch_importable"] = True
        payload["torch_version"] = torch.__version__
        payload["torch_cuda_version"] = getattr(torch.version, "cuda", None)
        payload["torch_cuda_available"] = bool(torch.cuda.is_available())
        payload["cuda_device_count"] = torch.cuda.device_count()
        if not torch.__version__.startswith("2.10.0"):
            blockers.append("torch_version_not_2_10_0")
        if "+cu128" not in torch.__version__:
            blockers.append("torch_not_built_for_cu128")
        if args.require_cuda and not torch.cuda.is_available():
            blockers.append("torch_cuda_unavailable")
    except Exception as exc:
        payload["torch_importable"] = False
        payload["torch_error_type"] = type(exc).__name__
        blockers.append("torch_import_failed")

    cudnn_headers = [path for path in _candidate_cudnn_headers() if path.is_file()]
    payload["cudnn_header_visible"] = bool(cudnn_headers)
    payload["cudnn_header_candidates_checked"] = [
        str(path) for path in _candidate_cudnn_headers()
    ]
    payload["cudnn_header_path"] = str(cudnn_headers[0]) if cudnn_headers else None
    if not cudnn_headers:
        blockers.append("cudnn_h_not_visible")

    source_root = Path(os.environ.get("BLUEPRINT_OSCAR_WAM_SOURCE_ROOT", "/opt/oscar-public"))
    payload["oscar_source_root"] = str(source_root)
    payload["oscar_inference_entrypoint_present"] = (
        source_root / "inference" / "inference_oscar.py"
    ).is_file()
    if not payload["oscar_inference_entrypoint_present"]:
        blockers.append("oscar_inference_entrypoint_missing")
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    spec = importlib.util.find_spec("transformer_engine")
    payload["transformer_engine_importable"] = spec is not None
    payload["transformer_engine_origin"] = getattr(spec, "origin", None) if spec else None
    try:
        import transformer_engine as te

        payload["transformer_engine_blueprint_compat_shim"] = bool(
            getattr(te, "BLUEPRINT_COMPAT_SHIM", False)
        )
        payload["transformer_engine_basis"] = (
            "blueprint_torch_sdpa_compat_shim"
            if payload["transformer_engine_blueprint_compat_shim"]
            else "real_transformer_engine_package"
        )
    except Exception as exc:
        payload["transformer_engine_error_type"] = type(exc).__name__
        blockers.append("transformer_engine_import_failed")
    if spec is None:
        blockers.append("transformer_engine_or_shim_not_importable")
    try:
        from transformer_engine.common.recipe import DelayedScaling, Format
        from transformer_engine.pytorch import ops
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_autocast
        from transformer_engine.pytorch.float8_tensor import Float8Tensor
        from transformer_engine.pytorch import Linear, LayerNormLinear
        from transformer_engine.pytorch.tensor import QuantizedTensor

        payload["transformer_engine_tensor_api_importable"] = True
        payload["transformer_engine_tensor_api_classes"] = [
            getattr(QuantizedTensor, "__name__", "QuantizedTensor"),
            getattr(Float8Tensor, "__name__", "Float8Tensor"),
        ]
        payload["transformer_engine_fp8_api_importable"] = True
        payload["transformer_engine_recipe_api_importable"] = True
        payload["transformer_engine_module_api_importable"] = True
        payload["transformer_engine_ops_api_importable"] = True
        payload["transformer_engine_fp8_enabled"] = bool(
            FP8GlobalStateManager.is_fp8_enabled()
        )
        payload["transformer_engine_module_api_classes"] = [
            getattr(Linear, "__name__", "Linear"),
            getattr(LayerNormLinear, "__name__", "LayerNormLinear"),
            getattr(ops.Sequential, "__name__", "Sequential"),
        ]
        payload["transformer_engine_recipe_format_names"] = [
            getattr(Format.E4M3, "name", "E4M3"),
            getattr(Format.HYBRID, "name", "HYBRID"),
            getattr(DelayedScaling, "__name__", "DelayedScaling"),
        ]
    except Exception as exc:
        payload["transformer_engine_tensor_api_importable"] = False
        payload["transformer_engine_fp8_api_importable"] = False
        payload["transformer_engine_recipe_api_importable"] = False
        payload["transformer_engine_module_api_importable"] = False
        payload["transformer_engine_ops_api_importable"] = False
        payload["transformer_engine_tensor_api_error_type"] = type(exc).__name__
        blockers.append("transformer_engine_tensor_api_not_importable")
    pynvml_spec = importlib.util.find_spec("pynvml")
    payload["pynvml_importable"] = pynvml_spec is not None
    payload["pynvml_origin"] = getattr(pynvml_spec, "origin", None) if pynvml_spec else None
    if pynvml_spec is None:
        blockers.append("pynvml_not_importable")
    loguru_spec = importlib.util.find_spec("loguru")
    payload["loguru_importable"] = loguru_spec is not None
    payload["loguru_origin"] = getattr(loguru_spec, "origin", None) if loguru_spec else None
    if loguru_spec is None:
        blockers.append("loguru_not_importable")
    worldsim_runtime_modules = {
        "attrs": "attrs",
        "av": "av",
        "boto3": "boto3",
        "botocore": "botocore",
        "cv2": "cv2",
        "decord": "decord",
        "fvcore": "fvcore",
        "hydra": "hydra",
        "matplotlib": "matplotlib",
        "megatron_core": "megatron.core",
        "omegaconf": "omegaconf",
        "onnx": "onnx",
        "onnxscript": "onnxscript",
        "pandas": "pandas",
        "pytest": "pytest",
        "qwen_vl_utils": "qwen_vl_utils",
        "termcolor": "termcolor",
        "torchmetrics": "torchmetrics",
        "wandb": "wandb",
        "webdataset": "webdataset",
    }
    payload["worldsim_runtime_imports"] = {}
    for label, module in worldsim_runtime_modules.items():
        module_spec = importlib.util.find_spec(module)
        payload["worldsim_runtime_imports"][label] = {
            "module": module,
            "importable": module_spec is not None,
            "origin": getattr(module_spec, "origin", None) if module_spec else None,
        }
        if module_spec is None:
            blockers.append(f"{label}_not_importable")

    payload["status"] = "passed" if not blockers else "blocked"
    payload["blockers"] = blockers
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
'''


def dockerfile_text(
    *,
    base_image: str = DEFAULT_BASE_IMAGE,
    platform: str = DEFAULT_PLATFORM,
    torch_index_url: str = DEFAULT_TORCH_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    torchvision_version: str = DEFAULT_TORCHVISION_VERSION,
    cudnn_package: str = DEFAULT_CUDNN_PACKAGE,
    oscar_source_url: str = DEFAULT_OSCAR_SOURCE_URL,
    oscar_source_ref: str = DEFAULT_OSCAR_SOURCE_REF,
    transformer_engine_mode: str = DEFAULT_TRANSFORMER_ENGINE_MODE,
) -> str:
    if transformer_engine_mode not in TRANSFORMER_ENGINE_MODES:
        raise ValueError(
            f"transformer_engine_mode must be one of {', '.join(TRANSFORMER_ENGINE_MODES)}"
        )
    return f"""# syntax=docker/dockerfile:1
# Blueprint reusable OSCAR WAM provider GPU image.
# This image intentionally excludes raw credentials and model checkpoints.
FROM --platform={platform} {base_image}

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
ARG OSCAR_SOURCE_URL={oscar_source_url}
ARG OSCAR_SOURCE_REF={oscar_source_ref}
ARG BLUEPRINT_TRANSFORMER_ENGINE_MODE={transformer_engine_mode}

ENV PIP_NO_CACHE_DIR=1 \\
    PYTHONUNBUFFERED=1 \\
    BLUEPRINT_WAM_PROVIDER_PYTHON=/usr/bin/python3 \\
    BLUEPRINT_OSCAR_WAM_SOURCE_ROOT=/opt/oscar-public \\
    BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY=torch_sdpa_compat_shim \\
    BLUEPRINT_OSCAR_WAM_ATTEMPT_TRANSFORMER_ENGINE_INSTALL=false \\
    BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL=true \\
    CUDNN_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn \\
    CUDNN_HOME=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn \\
    CPATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/include \\
    LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib

RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    ca-certificates \\
    cmake \\
    curl \\
    ffmpeg \\
    git \\
    libgl1 \\
    libglib2.0-0 \\
    ninja-build \\
    pkg-config \\
    python-is-python3 \\
    python3 \\
    python3-dev \\
    python3-pip \\
    python3-venv \\
    unzip \\
  && rm -rf /var/lib/apt/lists/*

COPY requirements_blueprint_oscar_wam.txt /opt/blueprint/requirements_blueprint_oscar_wam.txt
COPY filter_oscar_requirements.py /opt/blueprint/filter_oscar_requirements.py
COPY install_transformer_engine_shim.py /opt/blueprint/install_transformer_engine_shim.py
COPY oscar_wam_image_healthcheck.py /opt/blueprint/oscar_wam_image_healthcheck.py

RUN python3 -m pip install --upgrade pip setuptools wheel \\
  && python3 -m pip install --index-url {torch_index_url} \\
      torch=={torch_version} torchvision=={torchvision_version} \\
  && python3 -m pip install "{cudnn_package}" \\
  && python3 -m pip install -r /opt/blueprint/requirements_blueprint_oscar_wam.txt

RUN git clone --depth 1 "$OSCAR_SOURCE_URL" "$BLUEPRINT_OSCAR_WAM_SOURCE_ROOT" \\
  && if [[ "$OSCAR_SOURCE_REF" != "main" && "$OSCAR_SOURCE_REF" != "HEAD" ]]; then \\
       cd "$BLUEPRINT_OSCAR_WAM_SOURCE_ROOT" \\
       && git fetch --depth 1 origin "$OSCAR_SOURCE_REF" \\
       && git checkout FETCH_HEAD; \\
     fi

RUN python3 /opt/blueprint/filter_oscar_requirements.py \\
      "$BLUEPRINT_OSCAR_WAM_SOURCE_ROOT/requirements.txt" \\
      /tmp/oscar_requirements_without_torch_or_te.txt \\
  && if [[ -s /tmp/oscar_requirements_without_torch_or_te.txt ]]; then \\
       python3 -m pip install -r /tmp/oscar_requirements_without_torch_or_te.txt; \\
     fi

RUN if [[ "$BLUEPRINT_TRANSFORMER_ENGINE_MODE" == "real" ]]; then \\
       export NVTE_FRAMEWORK=pytorch; \\
       python3 -m pip install --no-build-isolation "transformer_engine[pytorch]"; \\
     else \\
       python3 /opt/blueprint/install_transformer_engine_shim.py "$BLUEPRINT_OSCAR_WAM_SOURCE_ROOT"; \\
     fi

RUN python3 /opt/blueprint/oscar_wam_image_healthcheck.py --build-time

WORKDIR /workspace
CMD ["bash"]
"""


def build_oscar_wam_gpu_image_context(
    *,
    job_dir: Path | None = None,
    image_ref: str | None = None,
    base_image: str = DEFAULT_BASE_IMAGE,
    platform: str = DEFAULT_PLATFORM,
    torch_index_url: str = DEFAULT_TORCH_INDEX_URL,
    torch_version: str = DEFAULT_TORCH_VERSION,
    torchvision_version: str = DEFAULT_TORCHVISION_VERSION,
    cudnn_package: str = DEFAULT_CUDNN_PACKAGE,
    oscar_source_url: str = DEFAULT_OSCAR_SOURCE_URL,
    oscar_source_ref: str = DEFAULT_OSCAR_SOURCE_REF,
    transformer_engine_mode: str = DEFAULT_TRANSFORMER_ENGINE_MODE,
    generated_at: str | None = None,
) -> dict[str, Any]:
    if transformer_engine_mode not in TRANSFORMER_ENGINE_MODES:
        raise ValueError(
            f"transformer_engine_mode must be one of {', '.join(TRANSFORMER_ENGINE_MODES)}"
        )
    root = _repo_root()
    generated = generated_at or utc_now_iso()
    output = Path(
        job_dir or root / "robot_eval_jobs" / f"oscar_wam_gpu_image_{_timestamp()}"
    ).expanduser().resolve()
    ensure_dir(output)
    configured_image_ref = (
        _string(image_ref)
        or _string(os.getenv(IMAGE_REF_ENV))
        or _string(os.getenv(LEGACY_IMAGE_REF_ENV))
    )
    dockerfile_path = output / DEFAULT_CONTEXT_FILENAME
    requirements_path = output / "requirements_blueprint_oscar_wam.txt"
    filter_path = output / "filter_oscar_requirements.py"
    shim_path = output / "install_transformer_engine_shim.py"
    healthcheck_path = output / "oscar_wam_image_healthcheck.py"
    dockerfile_path.write_text(
        dockerfile_text(
            base_image=base_image,
            platform=platform,
            torch_index_url=torch_index_url,
            torch_version=torch_version,
            torchvision_version=torchvision_version,
            cudnn_package=cudnn_package,
            oscar_source_url=oscar_source_url,
            oscar_source_ref=oscar_source_ref,
            transformer_engine_mode=transformer_engine_mode,
        ),
        encoding="utf-8",
    )
    requirements_path.write_text(requirements_text(), encoding="utf-8")
    filter_path.write_text(filter_requirements_script_text(), encoding="utf-8")
    shim_path.write_text(transformer_engine_shim_script_text(), encoding="utf-8")
    healthcheck_path.write_text(image_healthcheck_text(), encoding="utf-8")
    for script_path in (filter_path, shim_path, healthcheck_path):
        script_path.chmod(script_path.stat().st_mode | stat.S_IXUSR)

    build_command_path = output / "build_image.sh"
    push_command_path = output / "push_image.sh"
    run_healthcheck_command_path = output / "run_image_healthcheck.sh"
    build_command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        f"docker build --platform {platform} -f \"$SCRIPT_DIR/{DEFAULT_CONTEXT_FILENAME}\" "
        f"-t \"${{{IMAGE_REF_ENV}}}\" \"$SCRIPT_DIR\"\n"
    )
    push_command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"docker push \"${{{IMAGE_REF_ENV}}}\"\n"
    )
    run_healthcheck_command = (
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        f"docker run --rm --gpus all \"${{{IMAGE_REF_ENV}}}\" "
        "python3 /opt/blueprint/oscar_wam_image_healthcheck.py --require-cuda\n"
    )
    build_command_path.write_text(build_command, encoding="utf-8")
    push_command_path.write_text(push_command, encoding="utf-8")
    run_healthcheck_command_path.write_text(run_healthcheck_command, encoding="utf-8")
    for path in (build_command_path, push_command_path, run_healthcheck_command_path):
        path.chmod(0o755)

    blockers: list[str] = []
    if not configured_image_ref:
        blockers.append(f"missing_env_{IMAGE_REF_ENV}")
    elif not _image_ref_is_versioned(configured_image_ref):
        blockers.append("blocked_oscar_wam_gpu_image_ref_not_versioned")

    docker_auth = {
        "docker_username_file": _secret_file_status(
            "DOCKER_USERNAME_FILE",
            "~/.blueprint-secrets/docker_username",
        ),
        "docker_pat_file": _secret_file_status("DOCKER_PAT_FILE", "~/.blueprint-secrets/docker_pat"),
        "digitalocean_api_token_file": _secret_file_status(
            "DIGITALOCEAN_API_TOKEN_FILE",
            "~/.blueprint-secrets/digitalocean_api_token",
        ),
        "registry_auth_secret_values_written": False,
        "registry_auth_secret_hashes_written": False,
    }
    manifest = {
        "schema_version": OSCAR_WAM_GPU_IMAGE_SCHEMA_VERSION,
        "generated_at": generated,
        "status": "ready_for_image_build" if not blockers else "context_written_blocked",
        "job_dir": str(output),
        "image_ref_env": IMAGE_REF_ENV,
        "legacy_image_ref_env": LEGACY_IMAGE_REF_ENV,
        "configured_image_ref_present": bool(configured_image_ref),
        "configured_image_ref": configured_image_ref or None,
        "configured_image_ref_is_versioned": _image_ref_is_versioned(configured_image_ref),
        "base_image": base_image,
        "platform": platform,
        "torch_index_url": torch_index_url,
        "torch_version": torch_version,
        "torch_cuda_wheel_family": "cu128",
        "torchvision_version": torchvision_version,
        "cudnn_package": cudnn_package,
        "cudnn_header_visibility_required": True,
        "oscar_source_url": oscar_source_url,
        "oscar_source_ref": oscar_source_ref,
        "transformer_engine_mode": transformer_engine_mode,
        "transformer_engine_strategy": (
            "real_transformer_engine_pip_no_build_isolation"
            if transformer_engine_mode == "real"
            else "blueprint_torch_sdpa_compat_shim"
        ),
        "runtime_contract": {
            "sets_BLUEPRINT_OSCAR_WAM_SOURCE_ROOT": "/opt/oscar-public",
            "sets_BLUEPRINT_OSCAR_WAM_SKIP_RUNTIME_PIP_INSTALL": True,
            "sets_BLUEPRINT_OSCAR_WAM_TRANSFORMER_ENGINE_STRATEGY": (
                "torch_sdpa_compat_shim"
            ),
            "model_checkpoint_baked_into_image": False,
            "raw_credentials_baked_into_image": False,
            "provider_bundle_still_supplies_rollout_inputs": True,
        },
        "artifact_paths": {
            "dockerfile": str(dockerfile_path),
            "requirements": str(requirements_path),
            "filter_oscar_requirements": str(filter_path),
            "transformer_engine_shim": str(shim_path),
            "image_healthcheck": str(healthcheck_path),
            "build_command": str(build_command_path),
            "push_command": str(push_command_path),
            "run_healthcheck_command": str(run_healthcheck_command_path),
            "manifest": str(output / "oscar_wam_gpu_image_manifest.json"),
        },
        "commands": {
            "build": f"{build_command_path}",
            "push": f"{push_command_path}",
            "run_gpu_healthcheck": f"{run_healthcheck_command_path}",
            "vast_usage": (
                "blueprint-run-vast-provider-adapter ... "
                f"--provider-bundle-kind wam --public-image \"${{{IMAGE_REF_ENV}}}\""
            ),
        },
        "registry_auth": docker_auth,
        "blockers": blockers,
        "truth_boundary": {
            "image_build_is_not_model_execution": True,
            "image_push_is_not_wam_rollout_generation": True,
            "no_raw_tokens_or_hashes_written": True,
            "model_checkpoint_not_baked": True,
            "generated_world_rank_fidelity_result_proven": False,
            "generated_world_policy_evaluation_scope_proven": False,
        },
        "raw_secret_values_recorded": False,
        "secret_hashes_recorded": False,
    }
    write_json(output / "oscar_wam_gpu_image_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a reusable OSCAR WAM CUDA GPU image build context."
    )
    parser.add_argument("--job-dir")
    parser.add_argument("--image-ref")
    parser.add_argument("--base-image", default=DEFAULT_BASE_IMAGE)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--torch-index-url", default=DEFAULT_TORCH_INDEX_URL)
    parser.add_argument("--torch-version", default=DEFAULT_TORCH_VERSION)
    parser.add_argument("--torchvision-version", default=DEFAULT_TORCHVISION_VERSION)
    parser.add_argument("--cudnn-package", default=DEFAULT_CUDNN_PACKAGE)
    parser.add_argument("--oscar-source-url", default=DEFAULT_OSCAR_SOURCE_URL)
    parser.add_argument("--oscar-source-ref", default=DEFAULT_OSCAR_SOURCE_REF)
    parser.add_argument(
        "--transformer-engine-mode",
        choices=TRANSFORMER_ENGINE_MODES,
        default=DEFAULT_TRANSFORMER_ENGINE_MODE,
    )
    args = parser.parse_args(argv)
    manifest = build_oscar_wam_gpu_image_context(
        job_dir=Path(args.job_dir) if args.job_dir else None,
        image_ref=args.image_ref,
        base_image=args.base_image,
        platform=args.platform,
        torch_index_url=args.torch_index_url,
        torch_version=args.torch_version,
        torchvision_version=args.torchvision_version,
        cudnn_package=args.cudnn_package,
        oscar_source_url=args.oscar_source_url,
        oscar_source_ref=args.oscar_source_ref,
        transformer_engine_mode=args.transformer_engine_mode,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
