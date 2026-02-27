import argparse
import torch
import cuda.tile as ct
import math
import triton
import os

ConstInt = ct.Constant[int]

@ct.kernel
def stencil_1d_kernel(left_arr, center_arr, right_arr, output_arr, tile_size: ConstInt):
    pid = ct.bid(0)
    left   = ct.load(left_arr,   index=(pid,), shape=(tile_size,))
    center = ct.load(center_arr, index=(pid,), shape=(tile_size,))
    right  = ct.load(right_arr,  index=(pid,), shape=(tile_size,))
    result = (left + (center * 2) + right) * 0.25
    ct.store(output_arr, index=(pid,), tile=result)

# --- Triton Benchmarking Boilerplate ---

TILE_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
TILE_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "half", "float16"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 28)],
        line_arg='tile_size',
        line_vals=TILE_SIZES,
        line_names=[f'cuTile-{ts}' for ts in TILE_SIZES],
        styles=[(TILE_COLORS[i], '-') for i in range(len(TILE_SIZES))],
        ylabel='GB/s',
        plot_name=f'stencil1d-{PRECISION}',
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark_stencil1d(N, tile_size):
    dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'half': torch.float16, 'float16': torch.float16}
    torch_dtype = dtype_map[PRECISION]
    element_size = torch.tensor([], dtype=torch_dtype).element_size()
    
    # Alloc data with halo
    input_arr = torch.randn(N + 2, dtype=torch_dtype, device='cuda')
    left_view = input_arr[0 : -2]
    center_view = input_arr[1 : -1]
    right_view = input_arr[2 : ]
    output_arr = torch.empty(N, dtype=torch_dtype, device='cuda')
    
    grid = (math.ceil(N / tile_size), 1, 1)
    stream = torch.cuda.current_stream()
    quantiles = [0.5, 0.2, 0.8]
    
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, grid, stencil_1d_kernel, (left_view, center_view, right_view, output_arr, tile_size)), quantiles=quantiles)
    
    if N == 2**20:
         ref = (left_view + 2*center_view + right_view) * 0.25
         torch.testing.assert_close(output_arr, ref, atol=1e-2, rtol=1e-2)
         
    # 3 reads (left, center, right) + 1 write
    bytes_transferred = 4 * N * element_size
    gbps = lambda ms: (bytes_transferred / 1e9) / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_stencil1d.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/stencil1d_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"stencil1d-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")