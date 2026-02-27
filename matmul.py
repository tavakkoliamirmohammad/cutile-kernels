import argparse
import cuda.tile as ct
import torch
from math import ceil
import triton
import os
import math

ConstInt = ct.Constant[int]

def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
    # Get the global IDs of a given block in a 1D grid.
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    num_bid_in_group = GROUP_SIZE_M * num_bid_n
    group_id = bid // num_bid_in_group
    first_bid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_bid_m - first_bid_m, GROUP_SIZE_M)
    bid_m = first_bid_m + (bid % group_size_m)
    bid_n = (bid % num_bid_in_group) // group_size_m
    return bid_m, bid_n

def swizzle_2d(M, N, tm, tn, GROUP_SIZE_M):
    # Get the global IDs of the current block in a 1D grid.
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)

@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def matmul_kernel(A, B, C,
                  tm: ConstInt,         # Tile size along M dimension (rows of C)
                  tn: ConstInt,         # Tile size along N dimension (columns of C)
                  tk: ConstInt):        # Tile size along K dimension (inner product dimension)
    GROUP_SIZE_M = 8
    M = A.shape[0]
    N = B.shape[1]
    bidx, bidy = swizzle_2d(M, N, tm, tn, GROUP_SIZE_M)
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    for k in range(num_tiles_k):
        a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
        b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)
        accumulator = ct.mma(a, b, accumulator)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.store(C, index=(bidx, bidy), tile=accumulator)

@ct.kernel
def persistent_matmul_kernel(A, B, C,
                             tm: ConstInt,   # Tile size along M dimension (rows of C)
                             tn: ConstInt,   # Tile size along N dimension (columns of C)
                             tk: ConstInt):  # Tile size along K dimension
    GROUP_SIZE_M = 8
    bid = ct.bid(0)
    M = A.shape[0]
    N = B.shape[1]
    num_tiles_k = ct.num_tiles(A, axis=1, shape=(tm, tk))
    zero_pad = ct.PaddingMode.ZERO
    dtype = ct.tfloat32 if A.dtype == ct.float32 else A.dtype
    num_bid_m = ct.cdiv(M, tm)
    num_bid_n = ct.cdiv(N, tn)
    upper_bound = num_bid_m * num_bid_n
    num_tile_blocks = ct.num_blocks(0)
    for current_bid in range(bid, upper_bound, num_tile_blocks):
        accumulator = ct.full((tm, tn), 0, dtype=ct.float32)
        bidx, bidy = swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, current_bid)
        for k in range(num_tiles_k):
            a = ct.load(A, index=(bidx, k), shape=(tm, tk), padding_mode=zero_pad).astype(dtype)
            b = ct.load(B, index=(k, bidy), shape=(tk, tn), padding_mode=zero_pad).astype(dtype)
            accumulator = ct.mma(a, b, accumulator)
        accumulator = ct.astype(accumulator, C.dtype)
        ct.store(C, index=(bidx, bidy), tile=accumulator)

# --- Triton Benchmarking Boilerplate ---

MATMUL_CONFIGS = [(64, 64, 32), (128, 128, 32), (128, 256, 32), (128, 256, 64)]
CONFIG_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "float16", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[512, 1024, 2048, 4096, 8192, 16384],
        line_arg='config_str',
        line_vals=['torch'] + [f'standard-{c[0]}x{c[1]}x{c[2]}' for c in MATMUL_CONFIGS] + [f'persistent-{c[0]}x{c[1]}x{c[2]}' for c in MATMUL_CONFIGS],
        line_names=['Torch'] + [f'cuTile-Std-{c[0]}x{c[1]}x{c[2]}' for c in MATMUL_CONFIGS] + [f'cuTile-Persist-{c[0]}x{c[1]}x{c[2]}' for c in MATMUL_CONFIGS],
        styles=[('tab:green', '-')] + [(CONFIG_COLORS[i], '-') for i in range(len(MATMUL_CONFIGS))] + [(CONFIG_COLORS[i], '--') for i in range(len(MATMUL_CONFIGS))],
        ylabel='TFLOPS',
        plot_name=f'matmul-{PRECISION}',
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark_matmul(N, config_str):
    dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'float16': torch.float16, 'half': torch.float16}
    torch_dtype = dtype_map[PRECISION]
    M, K = N, N
    A = torch.randn(M, K, dtype=torch_dtype, device='cuda')
    B = torch.randn(K, N, dtype=torch_dtype, device='cuda')
    C = torch.empty((M, N), dtype=torch_dtype, device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    if config_str == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(A, B), quantiles=quantiles)
    else:
        parts = config_str.split('-')
        mode = parts[0]
        tm, tn, tk = [int(x) for x in parts[1].split('x')]
        stream = torch.cuda.current_stream()
        if mode == 'standard':
            grid = (math.ceil(M / tm) * math.ceil(N / tn), 1, 1)
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, grid, matmul_kernel, (A, B, C, tm, tn, tk)), quantiles=quantiles)
        else:
            NUM_SMS = torch.cuda.get_device_properties("cuda").multi_processor_count
            grid_size = min(NUM_SMS, math.ceil(M / tm) * math.ceil(N / tn))
            grid = (grid_size, 1, 1)
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, grid, persistent_matmul_kernel, (A, B, C, tm, tn, tk)), quantiles=quantiles)

    if N == 1024 and config_str != 'torch':
         torch.testing.assert_close(C, A @ B, atol=1e-2, rtol=1e-2)
    tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
    tflops_max = (2 * M * N * K) / (min_ms * 1e-3) / 1e12
    tflops_min = (2 * M * N * K) / (max_ms * 1e-3) / 1e12
    return tflops, tflops_max, tflops_min

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_matmul.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/matmul_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"matmul-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")
