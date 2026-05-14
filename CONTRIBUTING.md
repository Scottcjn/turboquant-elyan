# Contributing to TurboQuant-Elyan

Thank you for contributing to TurboQuant-Elyan, a production-ready implementation of Google's TurboQuant for AI efficiency with VSX (POWER8) and CUDA TQ3 KV Cache Compression.

## Project Overview

This project provides KV cache compression and quantization techniques for large language model inference, optimized for IBM POWER8/POWER9 with VSX vector extensions and CUDA GPUs.

## Development Setup

### Prerequisites

- IBM POWER8/POWER9 system (ppc64le) or x86_64 with CUDA
- Python 3.10+
- CUDA Toolkit 11.8+ (for GPU support)
- PyTorch 2.0+

### Environment Setup

```bash
git clone https://github.com/Scottcjn/turboquant-elyan.git
cd turboquant-elyan
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Code Style

- Python PEP 8 compliant
- Use `black` for formatting: `black .`
- Use `ruff` for linting: `ruff check .`
- Type hints for all function signatures
- Docstrings in Google style

## Testing

```bash
pytest tests/
pytest --cov=turboquant tests/
python benchmarks/run_benchmark.py --model <model_name>
```

## Submitting Changes

1. Fork the repository
2. Create a branch: `git checkout -b feat/your-feature`
3. Make changes and add tests
4. Ensure all tests pass
5. Submit a pull request

## Ideas for Contributions

- Additional quantization schemes (AWQ, GGUF)
- More VSX-optimized kernels for POWER architecture
- Integration with more LLM frameworks (vLLM, TGI)
- Benchmarking on additional hardware platforms
