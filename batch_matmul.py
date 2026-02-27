import argparse
from math import ceil
import cuda.tile as ct
import torch
import triton
import os
import math

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

# --- Triton Benchmarking Boilerplate ---

TILE_CONFIGS = [(64, 64, 32), (128, 128, 32), (128, 256, 32), (128, 256, 64)]
CONFIG_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "float16", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['N'], x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg='config_str',
        line_vals=['torch'] + [f'{c[0]}x{c[1]}x{c[2]}' for c in TILE_CONFIGS],
        line_names=['Torch'] + [f'cuTile-{c[0]}x{c[1]}x{c[2]}' for c in TILE_CONFIGS],
        styles=[('tab:green', '-')] + [(CONFIG_COLORS[i], '-') for i in range(len(TILE_CONFIGS))],
        ylabel='TFLOPS', plot_name=f'batch_matmul-{PRECISION}', args={'Batch': 16},
    )
]

@triton.testing.perf_report(configs)
def benchmark_batch_matmul(N, config_str, Batch):
    dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'float16': torch.float16, 'half': torch.float16}
    torch_dtype = dtype_map[PRECISION]
    M, K = N, N
    A = torch.randn(Batch, M, K, dtype=torch_dtype, device='cuda')
    B = torch.randn(Batch, K, N, dtype=torch_dtype, device='cuda')
    C = torch.empty((Batch, M, N), dtype=torch_dtype, device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    if config_str == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.bmm(A, B), quantiles=quantiles)
    else:
        tm, tn, tk = [int(x) for x in config_str.split('x')]
        grid = (Batch, math.ceil(M / tm), math.ceil(N / tn))
        stream = torch.cuda.current_stream()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, grid, batch_matmul_kernel, (A, B, C, tm, tn, tk)), quantiles=quantiles)
    if N == 512 and config_str != 'torch':
        torch.testing.assert_close(C, A @ B, atol=1e-2, rtol=1e-2)
    tflops = (2.0 * Batch * M * N * K) / (ms * 1e-3) / 1e12
    tflops_max = (2.0 * Batch * M * N * K) / (min_ms * 1e-3) / 1e12
    tflops_min = (2.0 * Batch * M * N * K) / (max_ms * 1e-3) / 1e12
    return tflops, tflops_max, tflops_min

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_batch_matmul.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/batch_matmul_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"batch_matmul-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")
