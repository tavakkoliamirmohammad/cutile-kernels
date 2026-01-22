# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
from math import ceil
import cuda.tile as ct
import torch


ConstInt = ct.Constant[int]


@ct.kernel
def batch_matmul_kernel(A, B, C, tm: ConstInt, tn: ConstInt, tk: ConstInt):
    """CuTile kernel for batch matrix multiplication
    A has shape (Batch, M, K), B has shape (Batch, K, N) and C has shape (Batch, M, N)
    Each block computes one (tm x tn) tile for a specific batch item.
    The grid is 3D: (Batch_idx, M_tile_idx, N_tile_idx).
    """
    pid_batch = ct.bid(0)  # Batch dimension
    pidx = ct.bid(1)  # M dimension
    pidy = ct.bid(2)  # N dimension

    # Calculate number of K tiles
    # A is (Batch, M, K), so K is axis 2
    # Use A.shape[2] for the total K dimension and ct.cdiv for ceiling division
    num_k_tiles = ct.cdiv(A.shape[2], tk)

    # Initialize accumulator
    accumulator = ct.full((tm, tn), 0.0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    # K-dimension loop
    for k in range(num_k_tiles):
        # Load tiles with 3D index and 3D shape
        # A is (Batch, M, K), load (1, tm, tk) tile
        a = ct.load(A, index=(pid_batch, pidx, k), shape=(1, tm, tk), padding_mode=zero_pad)
        a = ct.reshape(a, (tm, tk))  # Reshape to 2D for ct.mma

        # B is (Batch, K, N), load (1, tk, tn) tile
        b = ct.load(B, index=(pid_batch, k, pidy), shape=(1, tk, tn), padding_mode=zero_pad)
        b = ct.reshape(b, (tk, tn))  # Reshape to 2D for ct.mma

        accumulator = ct.mma(a, b, acc=accumulator)

    # Convert to output dtype and store
    result = ct.astype(accumulator, C.dtype)
    # Store with 3D index and 3D shape, C is (Batch, M, N)
    result_3d = ct.reshape(result, (1, tm, tn))
    ct.store(C, index=(pid_batch, pidx, pidy), tile=result_3d)


def bmm(a: torch.Tensor, b: torch.Tensor, out_dtype: torch.dtype, tile_config: tuple = None) -> torch.Tensor:
    """
    Batch Matrix Multiplication using cuTile's standard tiled kernel.

    Args:
        a (torch.Tensor): Input tensor A with shape (Batch, M, K).
        b (torch.Tensor): Input tensor B with shape (Batch, K, N).

    Returns:
        Output tensor C with shape (Batch, M, N).
    """
    # --- Input Validation ---
    if a.ndim != 3 or b.ndim != 3:
        raise ValueError("Input tensors for BMM must be 3D (Batch, M, K) and (Batch, K, N).")
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"""Batch dimensions must match:
                         A.shape[0]={a.shape[0]}, B.shape[0]={b.shape[0]}.""")
    if a.device != b.device or not a.is_cuda or not b.is_cuda or a.dtype != b.dtype:
        raise ValueError("""Input tensors must be on the same CUDA device
                         and have the same data type.""")

    # Get M, K, N dimensions
    Batch, M, K = a.shape
    _, K_b, N = b.shape
    assert K == K_b, f"Incompatible K dimensions: A's K is {K}, B's K is {K_b}"

    # Create output tensor
    output = torch.empty((Batch, M, N), device=a.device, dtype=out_dtype)

    # --- Determine Tile Shapes for Optimization (Fixed for float16 as per previous request) ---
    if tile_config is not None:
        tm_val, tn_val, tk_val = tile_config
    else:
        tm_val, tn_val, tk_val = 128, 256, 64  # Larger tiles for Tensor Core benefits

    # --- Grid calculation for standard 3D tiled kernel ---
    grid = (Batch, ceil(M / tm_val), ceil(N / tn_val))

    # --- Launch kernel ---
    ct.launch(torch.cuda.current_stream(), grid, batch_matmul_kernel,
              (a, b, output, tm_val, tn_val, tk_val))

    return output


def torch_batch_matmul_fp8(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    inv_sa = torch.tensor(1.0, device=A.device, dtype=torch.float32)
    inv_sb = torch.tensor(1.0, device=B.device, dtype=torch.float32)
    bs = A.shape[0]
    C = torch.empty((bs, A.shape[1], B.shape[2]), device=A.device, dtype=torch.float32)
    for i in range(bs):
        # Only multiplication of row-major and column-major matrices is supported by cuBLASLt
        # So we need to transpose B to column-major view
        A_row = A[i].contiguous()
        B_col = B[i].transpose(-2, -1).contiguous().transpose(-2, -1)
        C[i] = torch._scaled_mm(
            A_row, B_col, scale_a=inv_sa, scale_b=inv_sb, out_dtype=torch.float32
        )
    return C

def benchmark(kernel_func, kernel_name, output_filename):
    import time
    import matplotlib.pyplot as plt

    # Tile configs to test: (tm, tn, tk)
    tile_configs = [
        (64, 64, 32),
        (128, 128, 32),
        (128, 256, 32),
        (256, 128, 32),
        (128, 256, 64),
    ]
    
    # Vary Batch Size or N. Let's vary N with a fixed Batch Size.
    BATCH = 16
    problem_sizes = [256, 512, 1024, 2048, 4096]
    
    plt.figure(figsize=(12, 8))

    print(f"Benchmarking {kernel_name} with tile configs: {tile_configs}")
    
    for config in tile_configs:
        tm, tn, tk = config
        config_name = f"Tile {tm}x{tn}x{tk}"
        print(f"\nTesting {config_name}")
        speedups = []
        
        for N in problem_sizes:
            # Let's keep sizes relatively small (Batch=16) since BMM can get big
            M, K = N, N
            
            # Use float16
            A = torch.randn(BATCH, M, K, dtype=torch.float16, device='cuda')
            B = torch.randn(BATCH, K, N, dtype=torch.float16, device='cuda')
            
            # Warmup PyTorch (torch.bmm)
            for _ in range(10):
                torch.bmm(A, B)
            
            # Timing PyTorch
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            for _ in range(50):
                torch.bmm(A, B)
            end_event.record()
            torch.cuda.synchronize()
            torch_time = start_event.elapsed_time(end_event) / 50
            
            # Correctness & Warmup cuTile
            try:
                for _ in range(5):
                    bmm(A, B, out_dtype=torch.float16, tile_config=config)
            except Exception as e:
                print(f"Error for N={N}, config={config}: {e}")
                speedups.append(0)
                continue

            # Timing cuTile
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(50):
                bmm(A, B, out_dtype=torch.float16, tile_config=config)
            end_event.record()
            torch.cuda.synchronize()
            cutile_time = start_event.elapsed_time(end_event) / 50
            
            speedup = torch_time / cutile_time if cutile_time > 0 else 0.0
            speedups.append(speedup)
            
            tflops = (2 * BATCH * M * N * K) / (cutile_time * 1e-3) / 1e12
            print(f"N={N:<6} | Speedup={speedup:.2f}x | TFLOPS={tflops:.2f}")
        
        plt.plot(problem_sizes, speedups, label=config_name, marker='o')

    plt.xscale('log', base=2)
    plt.xlabel('Matrix Size (N)')
    plt.ylabel('Normalized Speedup (PyTorch / cuTile)')
    plt.title(f'{kernel_name} Speedup vs Matrix Size (Batch={BATCH})')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig(output_filename)
    print(f"\nBenchmark complete for {kernel_name}. Plot saved to {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
        default=True,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark",
        default=False,
    )
    args = parser.parse_args()

    if args.benchmark:
        print("\n\n=== Benchmarking Batch Matmul ===")
        benchmark(bmm, "Batch Matmul", "batch_matmul_benchmark.png")
        exit(0)
    print("--- Running cuTile Batched Matrix Multiplication (Standard Tiled) Sample ---")

    # --- User Configuration for BMM Example ---
    BATCH_DIM = 4
    M_DIM = 512
    K_DIM = 256
    N_DIM = 1024

    # --- Test Case 1: Standard BMM (float16) ---
    print("\n--- Test 1: Standard BMM (float16) ---")
    A_fp16 = torch.randn(BATCH_DIM, M_DIM, K_DIM, dtype=torch.float16, device='cuda')
    B_fp16 = torch.randn(BATCH_DIM, K_DIM, N_DIM, dtype=torch.float16, device='cuda')
    print(f"Input A shape: {A_fp16.shape}, dtype: {A_fp16.dtype}")
    print(f"Input B shape: {B_fp16.shape}, dtype: {B_fp16.dtype}")

    C_bmm_cutile_fp16 = bmm(A_fp16, B_fp16, A_fp16.dtype)
    print(f"""cuTile Standard BMM Output C
            shape:{C_bmm_cutile_fp16.shape},
            dtype: {C_bmm_cutile_fp16.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(C_bmm_cutile_fp16, A_fp16 @ B_fp16)
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    # --- Test Case 2: Standard BMM (float8_e4m3fn) ---
    print("\n--- Test 2: Standard BMM (float8_e4m3fn) ---")
    A_fp8 = torch.randn(
        BATCH_DIM, M_DIM, K_DIM, dtype=torch.float32, device='cuda'
    ).to(torch.float8_e4m3fn)
    B_fp8 = torch.randn(
        BATCH_DIM, K_DIM, N_DIM, dtype=torch.float32, device='cuda'
    ).to(torch.float8_e4m3fn)
    print(f"Input A shape: {A_fp8.shape}, dtype: {A_fp8.dtype}")
    print(f"Input B shape: {B_fp8.shape}, dtype: {B_fp8.dtype}")

    C_bmm_cutile_fp32 = bmm(A_fp8, B_fp8, torch.float32)
    print(f"""cuTile Standard BMM Output C
            shape:{C_bmm_cutile_fp32.shape},
            dtype: {C_bmm_cutile_fp32.dtype}""")
    if args.correctness_check:
        torch.testing.assert_close(C_bmm_cutile_fp32, torch_batch_matmul_fp8(A_fp8, B_fp8))
        print("Correctness check passed")
    else:
        print("Correctness check disabled")

    print("\n--- cuTile Batched Matrix Multiplication (Standard Tiled) examples complete ---")
