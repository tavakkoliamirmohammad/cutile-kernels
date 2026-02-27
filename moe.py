import argparse
import math
import torch
import torch.nn.functional as F
import cuda.tile as ct
import triton
import os

ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

@ct.kernel
def fused_moe_kernel(
    A, B, C, topk_weights, sorted_token_ids, sorted_expert_ids,
    num_token_replicas: int, mul_routed_weight: ConstBool,
    TILE_M: ConstInt, TILE_N: ConstInt, TILE_K: ConstInt,
):
    M = sorted_token_ids.shape[0]
    N = B.shape[1]
    K = B.shape[2]
    GROUP_SIZE_M = 8
    bid_m, bid_n = swizzle_2d(M, N, TILE_M, TILE_N, GROUP_SIZE_M)
    zero_pad = ct.PaddingMode.ZERO
    token_id_indices = bid_m * TILE_M + ct.arange(TILE_M, dtype=ct.int32)
    token_ids = ct.gather(sorted_token_ids, token_id_indices)
    a_row_indices = token_ids // num_token_replicas
    expert_id = ct.load(sorted_expert_ids, index=bid_m, shape=())
    accumulator = ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32)
    for k in range(0, ct.cdiv(K, TILE_K)):
        a_col_indices = k * TILE_K + ct.arange(TILE_K, dtype=ct.int32)
        a = ct.gather(A, (a_row_indices[:, None], a_col_indices[None, :]))
        b = ct.load(B, (expert_id, k, bid_n), shape=(1, TILE_K, TILE_N),
                    order=(0, 2, 1), padding_mode=zero_pad).reshape((TILE_K, TILE_N))
        accumulator = ct.mma(a, b, accumulator)
    if mul_routed_weight:
        moe_weight = ct.gather(topk_weights, token_ids)
        accumulator = accumulator * moe_weight[:, None]
    c_col_indices = bid_n * TILE_N + ct.arange(TILE_N, dtype=ct.int32)
    accumulator = ct.astype(accumulator, C.dtype)
    ct.scatter(C, (token_ids[:, None], c_col_indices[None, :]), accumulator)

@ct.kernel
def silu_and_mul_kernel(A, B, C, TILE_N: ConstInt):
    bid_m = ct.bid(0)
    ta = ct.load(A, (bid_m, 0), (1, TILE_N)).astype(ct.float32)
    tb = ct.load(B, (bid_m, 0), (1, TILE_N)).astype(ct.float32)
    denom = ct.add(1, ct.exp(-ta), flush_to_zero=True)
    sigmoid_ta = ct.truediv(1.0, denom, flush_to_zero=True, rounding_mode=ct.RoundingMode.APPROX)
    silu_ta = ct.mul(ta, sigmoid_ta, flush_to_zero=True)
    tc = ct.mul(silu_ta, tb, flush_to_zero=True)
    ct.store(C, (bid_m, 0), tc.astype(C.dtype))

def moe_align_tile_size_torch(topk_ids, tile_m, num_experts):
    device = topk_ids.device
    num_tokens, topk = topk_ids.shape
    total_tokens = num_tokens * topk
    flat_expert_ids = topk_ids.reshape(-1)
    sorted_token_indices = torch.argsort(flat_expert_ids, stable=True)
    expert_token_counts = torch.bincount(flat_expert_ids, minlength=num_experts)
    expert_block_counts = (expert_token_counts - 1 + tile_m) // tile_m
    total_blocks = expert_block_counts.sum()
    sorted_token_ids = torch.full((total_blocks * tile_m,), total_tokens, device=device, dtype=torch.int32)
    sorted_expert_ids = torch.zeros((total_blocks,), device=device, dtype=torch.int32)
    current_block = 0
    current_token = 0
    for expert_id in range(num_experts):
        token_count = expert_token_counts[expert_id]
        block_count = expert_block_counts[expert_id]
        sorted_expert_ids[current_block:current_block+block_count] = expert_id
        sorted_token_start = current_block * tile_m
        sorted_token_ids[sorted_token_start:sorted_token_start+token_count] = (
            sorted_token_indices[current_token:current_token+token_count]
        )
        current_token += token_count
        current_block += block_count
    return sorted_token_ids, sorted_expert_ids

def swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid):
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
    bid = ct.bid(0)
    return swizzle_2d_from_bid(M, N, tm, tn, GROUP_SIZE_M, bid)

def next_power_of_2(n: int):
    n -= 1
    n |= n >> 1; n |= n >> 2; n |= n >> 4; n |= n >> 8; n |= n >> 16; n |= n >> 32
    n += 1
    return n

# --- Triton Benchmarking ---

TILE_SIZES = [128, 256]
TILE_COLORS = ['tab:blue', 'tab:red']

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--precision", type=str, default="float16", choices=["float16", "half"])
args, _ = parser.parse_known_args()
PRECISION = args.precision

configs = [
    triton.testing.Benchmark(
        x_names=['num_tokens'],
        x_vals=[32, 64, 128, 256, 512, 1024],
        line_arg='tile_size',
        line_vals=[128, 256],
        line_names=[f'cuTile-{ts}' for ts in [128, 256]],
        styles=[(TILE_COLORS[i], '-') for i in range(len(TILE_SIZES))],
        ylabel='ms',
        plot_name=f'moe-{PRECISION}',
        args={'hidden_size': 512, 'num_experts': 64, 'intermediate_size': 1024, 'topk': 8},
    )
]

@triton.testing.perf_report(configs)
def benchmark_moe(num_tokens, tile_size, hidden_size, num_experts, intermediate_size, topk):
    dtype = torch.float16
    device = "cuda"
    
    hidden_states = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    w1 = torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device, dtype=dtype)
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, device=device, dtype=dtype)
    topk_ids = torch.stack([torch.randperm(num_experts, device=device)[:topk] for _ in range(num_tokens)])
    topk_weights = torch.softmax(torch.randn(num_tokens, topk, device=device), dim=-1).to(dtype)
    
    tile_m, tile_n, tile_k = tile_size, tile_size, 64
    
    def run_moe():
        # Setup tokens
        s_token_ids, s_expert_ids = moe_align_tile_size_torch(topk_ids, tile_m, num_experts)
        # Fwd 1
        m_fwd1 = s_token_ids.shape[0]
        grid1 = (math.ceil(m_fwd1 / tile_m) * math.ceil((intermediate_size * 2) / tile_n),)
        ic1 = torch.zeros((num_tokens, topk, intermediate_size * 2), device=device, dtype=dtype)
        ct.launch(torch.cuda.current_stream(), grid1, fused_moe_kernel, (hidden_states, w1, ic1.view(-1, intermediate_size*2), topk_weights.view(-1), s_token_ids, s_expert_ids, topk, False, tile_m, tile_n, tile_k))
        # SiLU
        ic2 = torch.zeros((num_tokens * topk, intermediate_size), device=device, dtype=dtype)
        ct.launch(torch.cuda.current_stream(), (num_tokens * topk,), silu_and_mul_kernel, (ic1.view(-1, intermediate_size*2).chunk(2, dim=-1)[0], ic1.view(-1, intermediate_size*2).chunk(2, dim=-1)[1], ic2, next_power_of_2(intermediate_size)))
        # Fwd 2
        m_fwd2 = s_token_ids.shape[0]
        grid2 = (math.ceil(m_fwd2 / tile_m) * math.ceil(hidden_size / tile_n),)
        ic3 = torch.zeros((num_tokens, topk, hidden_size), device=device, dtype=dtype)
        ct.launch(torch.cuda.current_stream(), grid2, fused_moe_kernel, (ic2, w2, ic3.view(-1, hidden_size), topk_weights.view(-1), s_token_ids, s_expert_ids, 1, True, tile_m, tile_n, tile_k))
        return torch.sum(ic3, dim=1)

    quantiles = [0.5, 0.2, 0.8]
    ms, min_ms, max_ms = triton.testing.do_bench(run_moe, quantiles=quantiles)
    return ms, max_ms, min_ms

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    import pandas as pd
    pd.DataFrame.to_csv = lambda *args, **kwargs: None
    results = benchmark_moe.run(print_data=False, return_df=True, show_plots=False)
    if isinstance(results, list): results = results[0]
    output_file = f"results/moe_{PRECISION}.txt"
    with open(output_file, "w") as f:
        f.write(f"moe-{PRECISION}:\n{results.to_string()}\n\n")
    print(f"\nResults dumped to {output_file}")
