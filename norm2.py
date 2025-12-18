import argparse
import torch
import cuda.tile as ct
import math

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

def norm2(a: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for L2 norm kernel.
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
        sum_of_squares_kernel,
        (a, result, TILE_SIZE)
    )
    
    # Finalize L2 Norm: sqrt(sum_of_squares)
    return torch.sqrt(result)

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
    
    print(f"Launching Norm 2 (L2 Norm) kernel with N={N}...")
    cutile_res = norm2(a)
    
    # Verification
    if args.correctness_check:
        torch_res = torch.norm(a, p=2)
        print(f"cuTile:  {cutile_res.item():.4f}")
        print(f"PyTorch: {torch_res.item():.4f}")
        torch.testing.assert_close(cutile_res[0], torch_res, rtol=1e-4, atol=1e-4)
        print("✓ Norm 2 Passed")

if __name__ == "__main__":
    run_test()
