# cuTile Experiments

A collection of high-performance GPU kernel implementations using `cuda.tile`, a Python-based domain-specific language (DSL) for GPU programming. This repository demonstrates how to implement common deep learning and numerical kernels with `cuda.tile`, providing both the kernel logic and PyTorch wrappers.

## Project Overview

The project includes the following kernel implementations:

- **Vector Addition (`vecadd.py`)**: A basic example of element-wise vector addition.
- **BabelStream (`babelstream.py`)**: A set of memory-bandwidth benchmarks (Copy, Mul, Add, Triad, Dot).
- **Matrix Multiplication (`matmul.py`)**: Standard and persistent matrix multiplication kernels.
- **Batched Matrix Multiplication (`batch_matmul.py`)**: Efficiently performing multiple matrix multiplications in a single kernel call.
- **Layer Normalization (`layernorm.py`)**: Forward and backward passes for LayerNorm, commonly used in Transformer models.
- **Mixture of Experts (`moe.py`)**: A fused MoE kernel that handles token routing and expert execution efficiently.
- **Matrix Transposition (`transpose.py`)**: Tiled matrix transposition.
- **Fast Fourier Transform (`fft.py`)**: Implementation of the FFT algorithm.
- **1D Stencil (`stencil1d.py`)**: A 1D stencil operation, typical in signal processing and simulations.
- **Tensor Contraction (`tensor_contraction_gen.py`)**: A generator for various tensor contraction operations.
- **Dot Product (`dot_product.py`)**: Computes the dot product of two vectors.
- **L1 Norm (`norm1.py`)**: Calculates the L1 norm of a vector.
- **L2 Norm (`norm2.py`)**: Calculates the L2 norm of a vector.

## Requirements

- Python 3.x
- PyTorch (with CUDA support)
- `cuda.tile` (NVIDIA's cuTile library)

## How to Run

Each script can be run independently. Most scripts include a correctness check against a standard PyTorch implementation.

For example, to run the Matrix Multiplication example:

```bash
python matmul.py
```

To run with/without correctness checks (where supported):

```bash
python layernorm.py --correctness-check
```

## Implementation Details

Most examples follow a consistent structure:
1. **Kernel Definition**: Decorated with `@ct.kernel`, these functions contain the `cuda.tile` logic that executes on the GPU.
2. **PyTorch Wrapper**: A Python function (e.g., `cutile_matmul`) that handles input validation, grid/block calculation, and launches the kernel using `ct.launch`.
3. **Reference Implementation**: A standard PyTorch version used to verify the correctness of the cuTile implementation.
4. **Main Block**: A test suite that runs the kernel on random data and compares the result with the reference.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note:** Files containing the `SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES` header are originally from the NVIDIA repository and are subject to the Apache-2.0 license as specified in their respective headers.
