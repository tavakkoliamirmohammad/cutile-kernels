import argparse
import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]

@ct.kernel
def norm1_kernel(a, result, tile_size: ConstInt):
    """
    cuTile kernel for computing the L1 norm of a vector.
    """
    pid = ct.bid(0)
    
    # Load input tile
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    
    # Compute Absolute Value
    abs_tile = ct.maximum(a_tile, -a_tile)
    
    # Reduce tile to scalar sum
    partial_sum = ct.sum(abs_tile)
    
    # Atomically add partial sum to global result
    ct.atomic_add(result, (0,), partial_sum)

def norm1(a: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for L1 norm kernel.
    """
    if not a.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    
    N = a.shape[0]
    TILE_SIZE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (math.ceil(N / TILE_SIZE), 1, 1)
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        norm1_kernel,
        (a, result, TILE_SIZE)
    )
    
    return result

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
        default=True,
    )
    args = parser.parse_args()

    N = 1024 * 1024
    
    # Generate Data
    a = torch.randn(N, dtype=torch.float32, device='cuda')
    
    print(f"Launching Norm 1 kernel with N={N}...")
    cutile_res = norm1(a)
    
    # Verification
    if args.correctness_check:
        torch_res = torch.norm(a, p=1)
        print(f"cuTile:  {cutile_res.item():.4f}")
        print(f"PyTorch: {torch_res.item():.4f}")
        torch.testing.assert_close(cutile_res[0], torch_res, rtol=1e-4, atol=1e-4)
        print("✓ Norm 1 Passed")

if __name__ == "__main__":
    run_test()
