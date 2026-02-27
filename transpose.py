import argparse
import cuda.tile as ct
import torch
from math import ceil
import triton
import os

ConstInt = ct.Constant[int]

@ct.kernel
def transpose_kernel(x, y,
                     tm: ConstInt,  # Tile size along M dimension (rows of original x)
                     tn: ConstInt):  # Tile size along N dimension (columns of original x)
    bidx = ct.bid(0)
    bidy = ct.bid(1)
    input_tile = ct.load(x, index=(bidx, bidy), shape=(tm, tn))
    transposed_tile = ct.transpose(input_tile)
    ct.store(y, index=(bidy, bidx), tile=transposed_tile)

# --- Triton Benchmarking Boilerplate ---

TILE_SIZES = [8, 16, 32, 64, 128, 256]
TILE_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "float16", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[128, 256, 512, 1024, 2048, 4096, 8192],
        line_arg='tile_size',
        line_vals=TILE_SIZES,
        line_names=[f'cuTile-{ts}x{ts}' for ts in TILE_SIZES],
        styles=[(TILE_COLORS[i], '-') for i in range(len(TILE_SIZES))],
        ylabel='GB/s',
        plot_name=f'transpose-{PRECISION}',
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark_transpose(N, tile_size):
    dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'float16': torch.float16, 'half': torch.float16}
    torch_dtype = dtype_map[PRECISION]
    element_size = torch.tensor([], dtype=torch_dtype).element_size()
    x = torch.randn(N, N, dtype=torch_dtype, device='cuda')
    y = torch.empty((N, N), dtype=torch_dtype, device='cuda')
    quantiles = [0.5, 0.2, 0.8]
    tm, tn = tile_size, tile_size
    grid = (ceil(N / tm), ceil(N / tn), 1)
    stream = torch.cuda.current_stream()
    ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, grid, transpose_kernel, (x, y, tm, tn)), quantiles=quantiles)
    if N == 1024:
         torch.testing.assert_close(y, x.T)
    bytes_transferred = 2 * N * N * element_size
    gbps = lambda ms: (bytes_transferred / 1e9) / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_transpose.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/transpose_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"transpose-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")