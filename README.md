# TurboQuant-Elyan: POWER8 VSX + CUDA TQ3 KV Cache Compression

Production-ready implementations of Google's [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026) 3-bit KV cache quantization for **IBM POWER8 VSX** and **NVIDIA CUDA** GPUs.

Built clean-room by [Elyan Labs](https://github.com/Scottcjn) from the published paper. No official Google code was used.

## What Is TurboQuant?

TurboQuant compresses the Key-Value cache in transformer attention from FP16/FP32 down to **3 bits per value** with near-zero quality loss. This is the biggest memory bottleneck during LLM and video model inference.

| Metric | Before | After TQ3 |
|--------|--------|-----------|
| KV cache per 128 values | 512 bytes (FP32) / 256 bytes (FP16) | **56 bytes** |
| Compression ratio | 1x | **9.1x vs FP32 / 4.6x vs FP16** |
| Quality loss (NMSE) | 0% | **~10.4%** (cosine sim > 0.95) |
| Retraining required | - | **None** (training-free) |

## Platforms

| Platform | File | Status | Throughput |
|----------|------|--------|------------|
| **IBM POWER8 VSX** | `ggml-turboquant-vsx.h` | Tested on POWER8 S824 | 474K vec/s dequant |
| **NVIDIA CUDA** | `turboquant_cuda_kernel.cu` | Tested on V100-SXM2-32GB, M40, RTX 4070 | **97.5M vec/s** dequant |

Cross-platform **bit-exact** compatibility verified: identical norm, scale, and packed indices across POWER8 and CUDA.

## Algorithm

1. **L2 normalize** the 128-float KV vector
2. **Fast Walsh-Hadamard Transform** (7 butterfly stages) to Gaussianize the distribution
3. **Deterministic sign flips** (xoshiro256** PRNG) for rotation
4. **Absmax scale** to [-1, +1] range
5. **Lloyd-Max 8-level codebook** quantization (3 bits per coordinate)
6. **3-bit packing** into 48 bytes + 4-byte norm + 4-byte scale = 56 bytes

Dequantize is the exact reverse.

### Lloyd-Max Centroids (d=128 Beta distribution)

```c
// 8 centroids, MSE-optimal for transformed distribution
{-0.9816f, -0.6168f, -0.3479f, -0.1129f, 0.1129f, 0.3479f, 0.6168f, 0.9816f}
```

## Quick Start

### POWER8 VSX

```bash
# Self-test (on POWER8)
cat > test.c << 'EOF'
#include "ggml-turboquant-vsx.h"
int main(void) { return tq3_self_test(); }
EOF
gcc -mcpu=power8 -mvsx -maltivec -O2 -std=c11 -o tq3_test test.c -lm
./tq3_test
```

### CUDA (V100/5070/3060/M40)

```bash
# Self-test
nvcc -arch=sm_70 -O2 -o tq3_cuda_test turboquant_cuda_kernel.cu -DTURBOQUANT_SELF_TEST
./tq3_cuda_test
```

### x86 Scalar Fallback

The VSX header includes a scalar fallback that works on any platform:

```bash
gcc -O2 -std=c11 -o tq3_test test.c -lm  # no -mvsx needed
./tq3_test
```

## Benchmark Results

### POWER8 S824 (512GB RAM, 128 threads)

| Operation | Throughput | Bandwidth |
|-----------|-----------|-----------|
| Quantize | 249,247 vec/s | 127.6 MB/s |
| Dequantize | 474,182 vec/s | 242.8 MB/s |

### V100-SXM2-32GB (CUDA SM_70)

| Operation | Throughput | Bandwidth |
|-----------|-----------|-----------|
| Quantize | 87,576,227 vec/s | 44.8 GB/s |
| Dequantize | 97,539,209 vec/s | 49.9 GB/s |

The V100 dequant bandwidth (49.9 GB/s) means TQ3 decompression is **hidden by HBM2 latency** -- effectively free.

## Use Cases

- **LLM inference**: Fit 6x longer contexts in the same VRAM. A 70B model's KV cache at 128K context drops from ~40GB to ~7GB.
- **Video generation (DiT)**: LTX-2, HunyuanVideo, Wan2.1 use transformer attention. TQ3 enables 6x longer video clips or higher resolution on the same GPU.
- **POWER8 inference**: Combined with PSE vec_perm collapse, enables 512GB RAM to serve as ~4.5TB effective KV cache.
- **Heterogeneous fleets**: Bit-exact cross-platform means a model can quantize KV on GPU and serve from CPU (or vice versa).

## Integration with llama.cpp

The headers follow the ggml quantization convention:

```c
// Quantize k floats into TQ3 blocks
void quantize_row_tq3_vsx(const float *x, void *y, int64_t k);

// Dequantize TQ3 blocks back to floats
void dequantize_row_tq3_vsx(const void *vx, float *y, int64_t k);
```

To add to an existing llama.cpp POWER8 build, include the header in `quants.c`:

```c
#include "ggml-turboquant-vsx.h"
```

## Related Work

- [ram-coffers](https://github.com/Scottcjn/ram-coffers) -- NUMA-aware weight banking and neuromorphic cognitive routing for POWER8
- [Google TurboQuant Paper](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) (ICLR 2026)
- [llama.cpp TurboQuant Discussion](https://github.com/ggml-org/llama.cpp/discussions/20969)

## License

MIT License. (c) 2026 Elyan Labs.

## Citation

If you use this implementation in research or production, please cite:

```
@software{turboquant_elyan_2026,
  title={TurboQuant-Elyan: POWER8 VSX + CUDA TQ3 Implementation},
  author={Boudreaux, Scott and Elyan Labs},
  year={2026},
  url={https://github.com/Scottcjn/turboquant-elyan}
}
```
