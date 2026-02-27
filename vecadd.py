import argparse
import cuda.tile as ct
import torch
import math
import triton
import os

ConstInt = ct.Constant[int]

@ct.kernel
def vec_add_kernel_1d(a, b, c, TILE: ConstInt):
    bid = ct.bid(0)
    a_tile = ct.load(a, index=(bid,), shape=(TILE,))
    b_tile = ct.load(b, index=(bid,), shape=(TILE,))
    sum_tile = a_tile + b_tile
    ct.store(c, index=(bid,), tile=sum_tile)

@ct.kernel
def vec_add_kernel_2d(a, b, c, TILE_X: ConstInt, TILE_Y: ConstInt):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    a_tile = ct.load(a, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))
    b_tile = ct.load(b, index=(bid_x, bid_y), shape=(TILE_X, TILE_Y))
    sum_tile = a_tile + b_tile
    ct.store(c, index=(bid_x, bid_y), tile=sum_tile)

@ct.kernel
def vec_add_kernel_1d_gather(a, b, c, TILE: ConstInt):
    bid = ct.bid(0)
    indices = bid * TILE + ct.arange(TILE, dtype=torch.int32)
    a_tile = ct.gather(a, indices)
    b_tile = ct.gather(b, indices)
    sum_tile = a_tile + b_tile
    ct.scatter(c, indices, sum_tile)

@ct.kernel
def vec_add_kernel_2d_gather(a, b, c, TILE_X: ConstInt, TILE_Y: ConstInt):
    bid_x = ct.bid(0)
    bid_y = ct.bid(1)
    x = (bid_x * TILE_X + ct.arange(TILE_X, dtype=torch.int32))[:, None]
    y = (bid_y * TILE_Y + ct.arange(TILE_Y, dtype=torch.int32))[None, :]
    a_tile = ct.gather(a, (x, y))
    b_tile = ct.gather(b, (x, y))
    sum_tile = a_tile + b_tile
    ct.scatter(c, (x, y), sum_tile)

# --- Triton Benchmarking Boilerplate ---

TILE_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
TILE_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['N'],
        x_vals=[2**i for i in range(10, 28)],
        line_arg='provider_ts',
        line_vals=['torch'] + [f'cutile-{ts}' for ts in TILE_SIZES] + [f'cutile-gather-{ts}' for ts in TILE_SIZES],
        line_names=['Torch'] + [f'cuTile-{ts}' for ts in TILE_SIZES] + [f'cuTile-Gather-{ts}' for ts in TILE_SIZES],
        styles=[('tab:green', '-')] + [(TILE_COLORS[i], '-') for i in range(len(TILE_SIZES))] + [(TILE_COLORS[i], '--') for i in range(len(TILE_SIZES))],
        ylabel='GB/s',
        plot_name=f'vecadd-{PRECISION}',
        args={},
    )
]

@triton.testing.perf_report(configs)
def benchmark_vecadd(N, provider_ts):
    dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'half': torch.float16}
    torch_dtype = dtype_map[PRECISION]
    element_size = torch.tensor([], dtype=torch_dtype).element_size()
    a = torch.randn(N, dtype=torch_dtype, device='cuda')
    b = torch.randn(N, dtype=torch_dtype, device='cuda')
    c = torch.empty_like(a)
    quantiles = [0.5, 0.2, 0.8]
    if provider_ts == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: a + b, quantiles=quantiles)
    elif 'gather' in provider_ts:
        tile_size = int(provider_ts.split('-')[-1])
        grid = (math.ceil(N / tile_size), 1, 1)
        stream = torch.cuda.current_stream()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, grid, vec_add_kernel_1d_gather, (a, b, c, tile_size)), quantiles=quantiles)
    else:
        tile_size = int(provider_ts.split('-')[1])
        grid = (math.ceil(N / tile_size), 1, 1)
        stream = torch.cuda.current_stream()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, grid, vec_add_kernel_1d, (a, b, c, tile_size)), quantiles=quantiles)
    if N == 2**20 and provider_ts != 'torch':
         torch.testing.assert_close(c, a + b, atol=1e-2, rtol=1e-2)
    bytes_transferred = 3 * N * element_size
    gbps = lambda ms: (bytes_transferred / 1e9) / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_vecadd.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/vecadd_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"vecadd-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")