"""Precision modes for VLA policies and helpers to describe a loaded model.

SmolVLA checkpoints are natively mixed precision: the vision encoder, language model
and action expert are BF16, while the state/action/time projections and some expert
attention projections are FP32 (D-022). Modes therefore convert by source dtype
instead of casting the whole module.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import torch

_TO_FP32 = ((torch.bfloat16, torch.float32), (torch.float16, torch.float32))


@dataclass(frozen=True)
class Precision:
    name: str
    # Pairs of (stored dtype, target dtype); dtypes not listed are left unchanged.
    convert: tuple[tuple[torch.dtype, torch.dtype], ...]
    tf32: bool
    description: str


PRECISIONS: dict[str, Precision] = {
    "native": Precision("native", (), False, "checkpoint as stored: BF16 backbone, FP32 projections"),
    "fp32": Precision("fp32", _TO_FP32, False, "all tensors FP32, TF32 disabled"),
    "tf32": Precision("tf32", _TO_FP32, True, "all tensors FP32, TF32 tensor-core matmuls"),
    "fp16": Precision("fp16", ((torch.bfloat16, torch.float16),), False, "BF16 tensors converted to FP16, FP32 projections unchanged"),
}


def _target(dtype: torch.dtype, precision: Precision) -> Optional[torch.dtype]:
    for src, dst in precision.convert:
        if dtype == src:
            return dst
    return None


def apply_precision(model: torch.nn.Module, precision: Precision) -> torch.nn.Module:
    """Convert parameters and buffers by their stored dtype, and set the TF32 switches."""
    torch.backends.cuda.matmul.allow_tf32 = precision.tf32
    torch.backends.cudnn.allow_tf32 = precision.tf32
    for module in model.modules():
        for param in module.parameters(recurse=False):
            dst = _target(param.dtype, precision)
            if dst is not None:
                param.data = param.data.to(dst)
        for name, buf in list(module.named_buffers(recurse=False)):
            dst = _target(buf.dtype, precision)
            if dst is not None:
                module._buffers[name] = buf.to(dst)
    return model


def describe_model(model: torch.nn.Module) -> dict:
    params = list(model.parameters())
    dtypes: Counter = Counter()
    for p in params:
        dtypes[str(p.dtype).replace("torch.", "")] += p.numel()
    return {
        "parameters": sum(p.numel() for p in params),
        "parameter_bytes": sum(p.numel() * p.element_size() for p in params),
        "buffer_bytes": sum(b.numel() * b.element_size() for b in model.buffers()),
        "parameters_by_dtype": dict(dtypes),
    }
