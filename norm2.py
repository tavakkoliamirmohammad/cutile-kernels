import argparse
import torch
import cuda.tile as ct
import math
import triton
import os

ConstInt = ct.Constant[int]

@ct.kernel
def sum_of_squares_kernel(a, result, tile_size: ConstInt):
    """
    cuTile kernel for computing the sum of squares of a vector.
    """
    pid = ct.bid(0)
    
    # Load input tile
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    
    # Square elements
    sq_tile = a_tile * a_tile
    
    # Reduce tile to scalar sum
    partial_sum = ct.sum(sq_tile)
    
    # Atomically add partial sum to global result
    ct.atomic_add(result, (0,), partial_sum)

@ct.kernel
def sum_of_squares_no_atomic_kernel(a, result, tile_size: ConstInt):
    """
    cuTile kernel for computing the sum of squares of a vector without atomics.
    Each block writes its partial sum to a specific index in the result array.
    """
    pid = ct.bid(0)
    
    # Load input tile
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    
    # Square elements
    sq_tile = a_tile * a_tile
    
    # Reduce tile to scalar sum
    partial_sum = ct.sum(sq_tile)
    
    # Write partial sum to result array at index pid
    ct.store(result, index=(pid,), tile=partial_sum)

# --- Triton Benchmarking Boilerplate ---

TILE_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
TILE_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "float16", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 28)],
        line_arg='provider_ts',
        line_vals=['torch'] + [f'cutile-{ts}' for ts in TILE_SIZES] + [f'cutile-atomic-{ts}' for ts in TILE_SIZES],
        line_names=['Torch'] + [f'cuTile-{ts}' for ts in TILE_SIZES] + [f'cuTile-Atomic-{ts}' for ts in TILE_SIZES],
        styles=[('tab:green', '-')] + [(TILE_COLORS[i], '-') for i in range(len(TILE_SIZES))] + [(TILE_COLORS[i], '--') for i in range(len(TILE_SIZES))],
        ylabel='GB/s',
        plot_name=f'norm2-{PRECISION}',
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark_norm2(N, provider_ts):
    dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'float16': torch.float16, 'half': torch.float16}
    torch_dtype = dtype_map[PRECISION]
    element_size = torch.tensor([], dtype=torch_dtype).element_size()
    a = torch.randn(N, dtype=torch_dtype, device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    if provider_ts == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.norm(a, p=2), quantiles=quantiles)
    elif 'atomic' in provider_ts:
        tile_size = int(provider_ts.split('-')[-1])
        num_blocks = (N + tile_size - 1) // tile_size
        result = torch.zeros(1, dtype=torch_dtype, device='cuda')
        stream = torch.cuda.current_stream()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (num_blocks, 1, 1), sum_of_squares_kernel, (a, result, tile_size)), quantiles=quantiles)
    else:
        tile_size = int(provider_ts.split('-')[1])
        num_blocks = (N + tile_size - 1) // tile_size
        result_partials = torch.zeros(num_blocks, dtype=torch_dtype, device='cuda')
        stream = torch.cuda.current_stream()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (num_blocks, 1, 1), sum_of_squares_no_atomic_kernel, (a, result_partials, tile_size)), quantiles=quantiles)
    if N == 2**20 and provider_ts != 'torch':
         if 'atomic' in provider_ts:
             tile_size = int(provider_ts.split('-')[-1]); num_blocks = (N + tile_size - 1) // tile_size
             result = torch.zeros(1, dtype=torch_dtype, device='cuda')
             ct.launch(torch.cuda.current_stream(), (num_blocks, 1, 1), sum_of_squares_kernel, (a, result, tile_size))
             cutile_res = torch.sqrt(result[0])
         else:
             tile_size = int(provider_ts.split('-')[1]); num_blocks = (N + tile_size - 1) // tile_size
             result_partials = torch.zeros(num_blocks, dtype=torch_dtype, device='cuda')
             ct.launch(torch.cuda.current_stream(), (num_blocks, 1, 1), sum_of_squares_no_atomic_kernel, (a, result_partials, tile_size))
             cutile_res = torch.sqrt(result_partials.sum())
         torch.testing.assert_close(cutile_res, torch.norm(a, p=2), rtol=1e-3, atol=1e-3)
    bytes_transferred = N * element_size
    gbps = lambda ms: (bytes_transferred / 1e9) / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_norm2.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/norm2_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"norm2-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")
