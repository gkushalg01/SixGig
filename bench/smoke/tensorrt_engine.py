"""TensorRT: build an FP16 MLP engine with the network API, run it, and check it against PyTorch.

Uses a strongly typed network, so precision comes from the tensor types rather than
builder flags (those are deprecated in recent TensorRT). The ONNX export path is
exercised in Phase 1, where real models are exported.
"""

from __future__ import annotations

import time
from pathlib import Path

from ..harness import Benchmark

BATCH = 64
WIDTH = 1024
LAYERS = 4


def _build_engine(trt, weights, biases) -> tuple[bytes, float]:
    import numpy as np

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    x = network.add_input("x", trt.float16, (BATCH, WIDTH))
    h = x
    for i, (w, bias) in enumerate(zip(weights, biases)):
        w_np = np.ascontiguousarray(w.cpu().numpy())
        b_np = np.ascontiguousarray(bias.reshape(1, WIDTH).cpu().numpy())
        w_c = network.add_constant(w_np.shape, trt.Weights(w_np)).get_output(0)
        b_c = network.add_constant(b_np.shape, trt.Weights(b_np)).get_output(0)
        mm = network.add_matrix_multiply(h, trt.MatrixOperation.NONE, w_c, trt.MatrixOperation.NONE).get_output(0)
        h = network.add_elementwise(mm, b_c, trt.ElementWiseOperation.SUM).get_output(0)
        if i < len(weights) - 1:
            h = network.add_activation(h, trt.ActivationType.RELU).get_output(0)
    h.name = "y"
    network.mark_output(h)

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 << 20)
    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    return bytes(serialized), build_s


def run(quick: bool = False, suite: str = "phase0") -> Path:
    import tensorrt as trt
    import torch

    warmup, iters = (10, 50) if quick else (50, 500)
    params = {"batch": BATCH, "width": WIDTH, "layers": LAYERS, "precision": "fp16", "network": "strongly_typed"}
    with Benchmark("tensorrt_mlp_fp16", suite=suite, params=params) as b:
        b.params["tensorrt"] = trt.__version__
        g = torch.Generator().manual_seed(0)
        weights = [(torch.randn(WIDTH, WIDTH, generator=g) / WIDTH**0.5).half() for _ in range(LAYERS)]
        biases = [(torch.randn(WIDTH, generator=g) * 0.01).half() for _ in range(LAYERS)]

        serialized, build_s = _build_engine(trt, weights, biases)
        b.metrics["engine_build_s"] = build_s
        b.metrics["engine_size_mib"] = len(serialized) / 2**20

        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        engine = runtime.deserialize_cuda_engine(serialized)
        context = engine.create_execution_context()

        x = torch.randn(BATCH, WIDTH, generator=g).half().cuda()
        y = torch.empty(BATCH, WIDTH, dtype=torch.float16, device="cuda")
        context.set_tensor_address("x", x.data_ptr())
        context.set_tensor_address("y", y.data_ptr())
        stream = torch.cuda.Stream()

        def infer():
            if not context.execute_async_v3(stream.cuda_stream):
                raise RuntimeError("execute_async_v3 failed")

        samples = b.measure(infer, warmup=warmup, iters=iters, sync=stream.synchronize)

        ref = x
        for i, (w, bias) in enumerate(zip(weights, biases)):
            ref = ref @ w.cuda() + bias.cuda()
            if i < LAYERS - 1:
                ref = torch.relu(ref)
        max_err = (y.float() - ref.float()).abs().max().item()
        scale = ref.float().abs().max().item()
        b.metrics["max_abs_error_vs_torch"] = max_err
        b.metrics["max_abs_reference"] = scale
        if max_err > 0.05 * max(scale, 1.0):
            raise RuntimeError(f"TensorRT output mismatch: max abs error {max_err} vs reference scale {scale}")
        p50_s = sorted(samples)[len(samples) // 2] / 1e3
        b.metrics["tflops_p50"] = 2 * BATCH * WIDTH * WIDTH * LAYERS / p50_s / 1e12
    return b.output_path
