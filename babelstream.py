import cuda.tile as ct
import torch
import triton
import argparse

SCALAR     = 0.4
INIT_A     = 2.0
INIT_B     = 0.5
INIT_C     = 0.0
TILE_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]
TILE_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "float16", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

@ct.kernel
def k_copy(a, c, size: ct.Constant[int]):
    pid = ct.bid(0)
    # Load: ptr, index, shape
    t_a = ct.load(a, index=(pid,), shape=(size,))
    # Store: ptr, index, value
    ct.store(c, (pid,), t_a)

@ct.kernel
def k_mul(b, c, scalar, size: ct.Constant[int]):
    pid = ct.bid(0)
    t_c = ct.load(c, index=(pid,), shape=(size,))
    t_b = ct.mul(scalar, t_c)
    ct.store(b, (pid,), t_b)

@ct.kernel
def k_add(a, b, c, size: ct.Constant[int]):
    pid = ct.bid(0)
    t_a = ct.load(a, index=(pid,), shape=(size,))
    t_b = ct.load(b, index=(pid,), shape=(size,))
    t_c = ct.add(t_a, t_b)
    ct.store(c, (pid,), t_c)

@ct.kernel
def k_triad(a, b, c, scalar, size: ct.Constant[int]):
    pid = ct.bid(0)
    t_b = ct.load(b, index=(pid,), shape=(size,))
    t_c = ct.load(c, index=(pid,), shape=(size,))
    t_res = ct.add(t_b, ct.mul(scalar, t_c))
    ct.store(a, (pid,), t_res)

@ct.kernel
def k_dot(a, b, partial_sums, size: ct.Constant[int]):
    pid = ct.bid(0)
    t_a = ct.load(a, index=(pid,), shape=(size,))
    t_b = ct.load(b, index=(pid,), shape=(size,))
    t_prod = ct.mul(t_a, t_b)
    
    t_sum = ct.sum(t_prod)
    
    ct.store(partial_sums, (pid,), t_sum)

# -----------------------------------------------------------------------------
# BENCHMARKING LOGIC
# -----------------------------------------------------------------------------
configs = [
    triton.testing.Benchmark(
        x_names=['N'],  # Argument names to use as an x-axis for the plot
        x_vals=[2**i for i in range(7, 26)],  # Different possible values for `x_name`
        line_arg='provider_ts',  # Argument name whose value corresponds to a different line in the plot
        line_vals=['torch'] + [f'cutile-{ts}' for ts in TILE_SIZES],  # Possible values for `line_arg`
        line_names=['Torch'] + [f'cuTile-{ts}' for ts in TILE_SIZES],  # Label name for the lines
        styles=[('tab:green', '-')] + [(TILE_COLORS[i], '-') for i in range(len(TILE_SIZES))],  # Line styles
        ylabel='GB/s',  # Label name for the y-axis
        plot_name=f'babelstream-{op}-{PRECISION}',  # Name for the plot.
        args={'op': op},  # Values for function arguments not in `x_names` and `y_name`
    ) for op in ['Copy', 'Mul', 'Add', 'Triad', 'Dot']
]

@triton.testing.perf_report(configs)
def benchmark_babelstream(N, provider_ts, op, dtype_str=PRECISION):
    # Parse provider and tile_size
    if provider_ts == 'torch':
        provider = 'torch'
        tile_size = 1024  # dummy
    else:
        provider = 'cutile'
        tile_size = int(provider_ts.split('-')[1])
    # Map string to torch/cupy types
    dtype_map = {
        'float64': torch.float64,
        'half': torch.float16,
        'float32': torch.float32,
        'float16': torch.float16,
    }
    torch_dtype = dtype_map[dtype_str]
    element_size = torch.tensor([], dtype=torch_dtype).element_size()

    a = torch.empty((N,), dtype=torch_dtype, device='cuda')
    b = torch.empty((N,), dtype=torch_dtype, device='cuda')
    c = torch.empty((N,), dtype=torch_dtype, device='cuda')
    
    if N == 128 and provider == 'cutile' and op == 'Copy':
        print(f"Benchmark Data Type: {torch_dtype} ({element_size} bytes per element)")
    
    quantiles = [0.5, 0.2, 0.8]
    blocks = (N + tile_size - 1) // tile_size

    if provider == 'torch':
        if op == 'Copy':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: c.copy_(a), quantiles=quantiles)
        elif op == 'Mul':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: b.copy_(c * SCALAR), quantiles=quantiles)
        elif op == 'Add':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: c.copy_(a + b), quantiles=quantiles)
        elif op == 'Triad':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: a.copy_(b + SCALAR * c), quantiles=quantiles)
        elif op == 'Dot':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.dot(a, b), quantiles=quantiles)
    elif provider == 'cutile':
        stream = torch.cuda.current_stream()
        if op == 'Copy':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (blocks, 1, 1), k_copy, (a, c, tile_size)), quantiles=quantiles)
        elif op == 'Mul':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (blocks, 1, 1), k_mul, (b, c, SCALAR, tile_size)), quantiles=quantiles)
        elif op == 'Add':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (blocks, 1, 1), k_add, (a, b, c, tile_size)), quantiles=quantiles)
        elif op == 'Triad':
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (blocks, 1, 1), k_triad, (a, b, c, SCALAR, tile_size)), quantiles=quantiles)
        elif op == 'Dot':
            partials = torch.empty(blocks, dtype=a.dtype, device=a.device)
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (blocks, 1, 1), k_dot, (a, b, partials, tile_size)), quantiles=quantiles)

    # Verification
    if N == 2**25 and provider == 'cutile':
        if op == 'Copy':
            torch.testing.assert_close(c, a, rtol=1e-3, atol=1e-3)
        elif op == 'Mul':
            torch.testing.assert_close(b, c * SCALAR, rtol=1e-3, atol=1e-3)
        elif op == 'Add':
            torch.testing.assert_close(c, a + b, rtol=1e-3, atol=1e-3)
        elif op == 'Triad':
            torch.testing.assert_close(a, b + SCALAR * c, rtol=1e-3, atol=1e-3)

    # GB/s calculation
    sizes = {
        'Copy': 2 * N * element_size,
        'Mul':  2 * N * element_size,
        'Add':  3 * N * element_size,
        'Triad': 3 * N * element_size,
        'Dot':  2 * N * element_size
    }
    gbps = lambda ms: (sizes[op] / 1e9) / (ms * 1e-3)
    
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    # Re-parse with help enabled for final run
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "float16", "half"])
    args = parser.parse_args()
    
    # Run the benchmark and get back the DataFrames
    results = benchmark_babelstream.run(print_data=False, return_df=True, show_plots=False, dtype_str=args.precision)
    
    # Save the formatted tables to a file
    output_file = f"results/babelstream_{args.precision}.txt"
    ops = ['Copy', 'Mul', 'Add', 'Triad', 'Dot']
    with open(output_file, "w") as f:
        for op, df in zip(ops, results):
            f.write(f"babelstream-{op}-{args.precision}:\n")
            f.write(df.to_string())
            f.write("\n\n")
    print(f"\nResults dumped to {output_file}")