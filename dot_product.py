import argparse
import torch
import cuda.tile as ct
import math

ConstInt = ct.Constant[int]

@ct.kernel
def dot_product_kernel(a, b, result, tile_size: ConstInt):
    """
    cuTile kernel for computing the dot product of two vectors.
    """
    pid = ct.bid(0)
    
    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    
    # Element-wise multiplication
    prod = a_tile * b_tile
    
    # Reduce tile to a single scalar sum
    partial_sum = ct.sum(prod)
    
    # Atomically add partial sum to global result
    ct.atomic_add(result, (0,), partial_sum)

def dot_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Wrapper for dot product kernel.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")
    
    N = a.shape[0]
    # Use a tile size of 1024 or smaller
    TILE_SIZE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1
    
    result = torch.zeros(1, dtype=a.dtype, device=a.device)
    
    grid = (math.ceil(N / TILE_SIZE), 1, 1)
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        dot_product_kernel,
        (a, b, result, TILE_SIZE)
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
    b = torch.randn(N, dtype=torch.float32, device='cuda')
    
    print(f"Launching Dot Product kernel with N={N}...")
    cutile_res = dot_product(a, b)
    
    # Verification
    if args.correctness_check:
        torch_res = torch.dot(a, b)
        print(f"cuTile: {cutile_res.item():.4f}")
        print(f"PyTorch: {torch_res.item():.4f}")
        torch.testing.assert_close(cutile_res[0], torch_res, rtol=1e-4, atol=1e-4)
        print("✓ Dot Product Passed")

if __name__ == "__main__":
    run_test()
