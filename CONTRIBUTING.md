# Contributing to TurboQuant-Elyan

Thanks for helping improve TurboQuant-Elyan. This repository contains a
clean-room TQ3 KV cache compression implementation for POWER8 VSX and CUDA, so
changes should preserve bit compatibility, numerical behavior, and platform
portability unless the PR clearly explains why behavior is changing.

## Getting Started

1. Fork the repository and create a topic branch.
2. Review `README.md` to understand the POWER8 VSX, CUDA, and scalar fallback
   paths.
3. Keep changes focused on one implementation area at a time:
   `ggml-turboquant-vsx.h`, `turboquant_cuda_kernel.cu`, documentation, or
   benchmarks.
4. Build and run the relevant self-test for every platform your change touches.

## Validation

For POWER8 VSX changes, validate on POWER8 hardware when available:

```bash
cat > test.c << 'EOF'
#include "ggml-turboquant-vsx.h"
int main(void) { return tq3_self_test(); }
EOF
gcc -mcpu=power8 -mvsx -maltivec -O2 -std=c11 -o tq3_test test.c -lm
./tq3_test
```

For scalar fallback changes, also build without VSX flags:

```bash
gcc -O2 -std=c11 -o tq3_test test.c -lm
./tq3_test
```

For CUDA changes, build with an architecture that matches your test GPU:

```bash
nvcc -arch=sm_70 -O2 -o tq3_cuda_test turboquant_cuda_kernel.cu -DTURBOQUANT_SELF_TEST
./tq3_cuda_test
```

If you cannot run hardware-specific validation, state that clearly in the PR and
include the compile or static checks you did run.

## Code Style

- Use portable C/C++ and CUDA constructs unless a block is explicitly guarded by
  a platform macro.
- Keep POWER8 VSX code behind the existing VSX/POWER8 guards and preserve the
  scalar fallback path.
- Avoid hidden allocations, global side effects, or nondeterministic behavior in
  quantize/dequantize paths.
- Keep constants, block layouts, centroids, boundaries, packing order, and seed
  handling synchronized between VSX and CUDA implementations.
- Prefer small, readable helper functions for math, packing, and validation
  logic.

## Pull Requests

Every PR should include:

- A short summary of the behavior or documentation changed.
- The platform paths affected: POWER8 VSX, CUDA, scalar fallback, docs, or
  benchmarks.
- Validation commands and results, including hardware and compiler details when
  relevant.
- Benchmark results for performance-sensitive changes, or a note explaining why
  benchmarks are not needed.
- Any compatibility risks for llama.cpp integration or cross-platform
  bit-exactness.

Documentation-only PRs should still explain which files were reviewed so the
guide stays aligned with the implementation.

## Reporting Issues

When filing a bug, include the platform, compiler, build command, input shape,
expected result, actual result, and any benchmark or self-test output. For
numerical differences, include enough detail to tell whether the issue is a
packing mismatch, centroid/boundary mismatch, normalization issue, or platform
specific floating-point behavior.
