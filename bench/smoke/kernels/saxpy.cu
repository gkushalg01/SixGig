// Phase 0 smoke test: a SAXPY kernel built with the environment's nvcc for sm_89.
// Exposed through a C ABI so Python can load it with ctypes and time it with CUDA events.

#include <cuda_runtime.h>

#include <cmath>
#include <vector>

namespace {

__global__ void saxpy(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        y[i] = a * x[i] + y[i];
    }
}

}  // namespace

extern "C" {

// Runs `warmup` untimed launches and `iters` timed launches of y = a*x + y.
// Writes each timed launch's duration in milliseconds to times_ms[0..iters).
// Verifies one extra launch against a host reference and writes the max abs error.
// Returns a cudaError_t value (0 on success).
int sixgig_saxpy_bench(int n, int warmup, int iters, float* times_ms, float* max_abs_err) {
    const float a = 2.0f;
    std::vector<float> hx(n), hy(n);
    for (int i = 0; i < n; ++i) {
        hx[i] = static_cast<float>(i % 1024) * 0.001f;
        hy[i] = 1.0f;
    }

    float *dx = nullptr, *dy = nullptr;
    cudaEvent_t start = nullptr, stop = nullptr;
    cudaError_t err = cudaSuccess;

    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    const int block = 256;
    const int grid = sms > 0 ? sms * 8 : 1024;

#define CHECK(call)               \
    do {                          \
        err = (call);             \
        if (err != cudaSuccess) { \
            goto cleanup;         \
        }                         \
    } while (0)

    CHECK(cudaMalloc(&dx, sizeof(float) * n));
    CHECK(cudaMalloc(&dy, sizeof(float) * n));
    CHECK(cudaMemcpy(dx, hx.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < warmup; ++i) {
        saxpy<<<grid, block>>>(n, a, dx, dy);
    }
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    for (int i = 0; i < iters; ++i) {
        CHECK(cudaEventRecord(start));
        saxpy<<<grid, block>>>(n, a, dx, dy);
        CHECK(cudaEventRecord(stop));
        CHECK(cudaEventSynchronize(stop));
        CHECK(cudaEventElapsedTime(&times_ms[i], start, stop));
    }
    CHECK(cudaGetLastError());

    // Correctness: reset y, run once, compare with the host result.
    CHECK(cudaMemcpy(dy, hy.data(), sizeof(float) * n, cudaMemcpyHostToDevice));
    saxpy<<<grid, block>>>(n, a, dx, dy);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(hy.data(), dy, sizeof(float) * n, cudaMemcpyDeviceToHost));
    {
        float worst = 0.0f;
        for (int i = 0; i < n; ++i) {
            const float expected = a * (static_cast<float>(i % 1024) * 0.001f) + 1.0f;
            worst = fmaxf(worst, fabsf(hy[i] - expected));
        }
        *max_abs_err = worst;
    }

cleanup:
    if (start) cudaEventDestroy(start);
    if (stop) cudaEventDestroy(stop);
    if (dx) cudaFree(dx);
    if (dy) cudaFree(dy);
#undef CHECK
    return static_cast<int>(err);
}

}  // extern "C"
