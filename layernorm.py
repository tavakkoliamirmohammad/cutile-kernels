import argparse
import math
import torch
import torch.nn.functional as F
import cuda.tile as ct
import triton
import os

ConstInt = ct.Constant[int]
PAD_ZERO = ct.PaddingMode.ZERO

@ct.kernel
def layer_norm_fwd(X, W, B, Y, Mean, Rstd, eps, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]
    mean = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mean += tx
    mean = ct.sum(mean, axis=1) / N
    ct.store(Mean, index=(bid_m,), tile=mean)
    var = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        mask = (j * TILE_N + ct.arange(TILE_N, dtype=ct.int32)) < N
        centered_tx = ct.where(mask, tx - mean, 0)
        var += centered_tx ** 2
    var = ct.sum(var, axis=1) / N
    rstd = 1 / ct.sqrt(var + eps)
    ct.store(Rstd, index=(bid_m,), tile=rstd)
    for j in range(num_tiles):
        tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        tb = ct.load(B, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
        ty = (tx - mean) * rstd
        ty = ty * tw + tb
        ct.store(Y, index=(bid_m, j), tile=ty.astype(Y.dtype))

def bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N):
    tx = ct.load(X, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    tw = ct.load(W, index=(j,), shape=(TILE_N,), padding_mode=PAD_ZERO)
    tdy = ct.load(DY, index=(bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
    xhat = (tx - mean) * rstd
    wdy = tw * tdy
    mask = j * TILE_N + ct.arange(TILE_N, dtype=ct.int32) < N
    xhat = ct.where(mask, xhat, 0)
    wdy = ct.where(mask, wdy, 0)
    return tdy, xhat, wdy

@ct.kernel
def layer_norm_bwd_dx_partial_dwdb(DX, DY, DW, DB, X, W, Mean, Rstd, Locks, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    num_tiles = ct.num_tiles(X, axis=1, shape=(1, TILE_N))
    N = X.shape[1]
    GROUP_SIZE_M = DW.shape[0]
    group_bid_m = bid_m % GROUP_SIZE_M
    mean = ct.load(Mean, index=(bid_m,), shape=(1,))
    rstd = ct.load(Rstd, index=(bid_m,), shape=(1,))
    c1 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    c2 = ct.full((1, TILE_N), 0, dtype=ct.float32)
    for j in range(num_tiles):
        _, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        c1 += xhat * wdy
        c2 += wdy
    c1 = ct.sum(c1, axis=1) / N
    c2 = ct.sum(c2, axis=1) / N
    for j in range(num_tiles):
        tdy, xhat, wdy = bwd_helper(X, W, DY, bid_m, j, mean, rstd, TILE_N, N)
        tdx = (wdy - (xhat * c1 + c2)) * rstd
        ct.store(DX, index=(bid_m, j), tile=tdx.astype(DX.dtype))
        partial_dw = (tdy * xhat).astype(DW.dtype)
        partial_db = tdy.astype(DB.dtype)
        while ct.atomic_cas(Locks, group_bid_m, 0, 1, memory_order=ct.MemoryOrder.ACQUIRE) == 1: pass
        partial_dw += ct.load(DW, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        partial_db += ct.load(DB, index=(group_bid_m, j), shape=(1, TILE_N), padding_mode=PAD_ZERO)
        ct.store(DW, index=(group_bid_m, j), tile=partial_dw)
        ct.store(DB, index=(group_bid_m, j), tile=partial_db)
        ct.atomic_xchg(Locks, group_bid_m, 0, memory_order=ct.MemoryOrder.RELEASE)

@ct.kernel
def layer_norm_bwd_dwdb(DW, DB, FINAL_DW, FINAL_DB, TILE_M: ConstInt, TILE_N: ConstInt):
    bid_n = ct.bid(0)
    num_tiles = ct.num_tiles(DW, axis=0, shape=(TILE_M, TILE_N))
    dw = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    db = ct.zeros((TILE_M, TILE_N), dtype=ct.float32)
    for i in range(num_tiles):
        dw += ct.load(DW, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
        db += ct.load(DB, index=(i, bid_n), shape=(TILE_M, TILE_N), padding_mode=PAD_ZERO)
    sum_dw = ct.sum(dw, axis=0)
    sum_db = ct.sum(db, axis=0)
    ct.store(FINAL_DW, index=(bid_n,), tile=sum_dw.astype(FINAL_DW.dtype))
    ct.store(FINAL_DB, index=(bid_n,), tile=sum_db.astype(FINAL_DB.dtype))

# --- Triton Benchmarking Boilerplate ---

TILE_SIZES = [512, 1024, 2048, 4096, 8192]
TILE_COLORS = ['tab:blue', 'tab:red', 'tab:orange', 'tab:purple', 'tab:brown']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float32", choices=["float64", "float32", "float16", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['N'], x_vals=[1024, 2048, 4096, 8192, 16384],
        line_arg='provider_ts',
        line_vals=['torch'] + [f'cutile-{ts}' for ts in TILE_SIZES],
        line_names=['Torch'] + [f'cuTile-{ts}' for ts in TILE_SIZES],
        styles=[('tab:green', '-')] + [(TILE_COLORS[i], '-') for i in range(len(TILE_SIZES))],
        ylabel='GB/s', plot_name=f'layernorm-{PRECISION}', args={'M': 4096},
    )
]

@triton.testing.perf_report(configs)
def benchmark_layernorm(N, provider_ts, M):
    dtype_map = {'float64': torch.float64, 'float32': torch.float32, 'float16': torch.float16, 'half': torch.float16}
    torch_dtype = dtype_map[PRECISION]
    element_size = torch.tensor([], dtype=torch_dtype).element_size()
    X = torch.randn((M, N), dtype=torch_dtype, device='cuda')
    W = torch.randn(N, dtype=torch_dtype, device='cuda')
    B = torch.randn(N, dtype=torch_dtype, device='cuda')
    Y = torch.empty_like(X)
    Mean = torch.empty(M, dtype=torch.float32, device='cuda')
    Rstd = torch.empty(M, dtype=torch.float32, device='cuda')
    eps, quantiles = 1e-5, [0.5, 0.2, 0.8]
    if provider_ts == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.nn.functional.layer_norm(X, (N,), W, B, eps), quantiles=quantiles)
    else:
        tile_size = int(provider_ts.split('-')[1])
        stream = torch.cuda.current_stream()
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: ct.launch(stream, (M,), layer_norm_fwd, (X, W, B, Y, Mean, Rstd, eps, tile_size)), quantiles=quantiles)
    if N == 8192 and provider_ts != 'torch':
        ref = torch.nn.functional.layer_norm(X, (N,), W, B, eps)
        torch.testing.assert_close(Y, ref, atol=1e-2, rtol=1e-2)
    bytes_transferred = (2 * M * N) * element_size
    gbps = lambda ms: (bytes_transferred / 1e9) / (ms * 1e-3)
    return gbps(ms), gbps(max_ms), gbps(min_ms)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_layernorm.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/layernorm_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"layernorm-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")