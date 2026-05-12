# Contributing to TurboQuant-Elyan

Thanks for taking the time to improve TurboQuant-Elyan. This repository is a
clean-room implementation of TurboQuant-style 3-bit KV cache compression for
POWER8 VSX and CUDA, so contributions should keep correctness, portability, and
benchmark reproducibility front and center.

## Good First Contributions

- Improve setup notes for POWER8, CUDA, or scalar fallback builds.
- Add benchmark results for additional GPUs, CPUs, or compiler versions.
- Clarify algorithm notes where the README can be more precise.
- Add small, self-contained tests that verify packing, unpacking, or numerical
  tolerances across supported backends.
- Report portability issues with exact compiler, hardware, and command output.

## Development Workflow

1. Fork the repository and create a focused branch.
2. Keep changes narrowly scoped to one improvement or bug fix.
3. Prefer small commits with clear messages.
4. Run the relevant self-test or compile check before opening a pull request.
5. Include the exact commands, hardware, and compiler/toolchain versions used
   for validation.

## Local Validation

For the POWER8 VSX header, validate the self-test on compatible hardware:

```bash
cat > test.c << 'EOF'
#include "ggml-turboquant-vsx.h"
int main(void) { return tq3_self_test(); }
EOF
gcc -mcpu=power8 -mvsx -maltivec -O2 -std=c11 -o tq3_test test.c -lm
./tq3_test
```

For the CUDA implementation, build and run the kernel self-test with an
architecture flag that matches the target GPU:

```bash
nvcc -arch=sm_70 -O2 -o tq3_cuda_test turboquant_cuda_kernel.cu -DTURBOQUANT_SELF_TEST
./tq3_cuda_test
```

For platforms without VSX or CUDA, use the scalar fallback path described in
the README and include the compiler command in the pull request.

## Pull Request Checklist

- The PR explains what changed and why.
- Validation commands and results are included.
- Numerical changes include before/after tolerance or error details.
- Benchmark changes include hardware, compiler, flags, and sample size.
- The implementation remains clean-room and does not copy proprietary or
  incompatible licensed source code.

## Clean-Room and Licensing Notes

Do not paste code from closed-source implementations or from sources with
licenses that conflict with this repository. If an implementation detail comes
from a paper, blog post, benchmark note, or public documentation, cite that
source in the PR description or code comments where appropriate.

## Reporting Issues

When opening an issue, include:

- The file or platform affected.
- Hardware model and operating system.
- Compiler or CUDA toolkit version.
- The command that failed.
- The full error output or benchmark result being questioned.

Clear reproduction details make it much easier to review and merge fixes.
