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

@ct.kernel
def dot_product_no_atomic_kernel(a, b, result, tile_size: ConstInt):
    """
    cuTile kernel for computing the dot product of two vectors without atomics.
    Each block writes its partial sum to a specific index in the result array.
    """
    pid = ct.bid(0)
    
    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    
    # Element-wise multiplication
    prod = a_tile * b_tile
    
    # Reduce tile to a single scalar sum
    partial_sum = ct.sum(prod)
    
    # Write partial sum to result array at index pid
    # result is expected to be of size (grid_size,)
    ct.store(result, index=(pid,), tile=partial_sum)

def dot_product(a: torch.Tensor, b: torch.Tensor, tile_size: int = None) -> torch.Tensor:
    """
    Wrapper for dot product kernel.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")
    
    N = a.shape[0]
    # Use a tile size of 1024 or smaller, or user provided
    if tile_size is not None:
        TILE_SIZE = tile_size
    else:
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


    return result


def dot_product_no_atomic(a: torch.Tensor, b: torch.Tensor, tile_size: int = None) -> torch.Tensor:
    """
    Wrapper for dot product kernel without atomics.
    Reduces on the host.
    """
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape.")
    if not a.is_cuda or not b.is_cuda:
        raise ValueError("Input tensors must be on a CUDA device.")
    
    N = a.shape[0]
    # Use a tile size of 1024 or smaller, or user provided
    if tile_size is not None:
        TILE_SIZE = tile_size
    else:
        TILE_SIZE = min(1024, 2 ** math.ceil(math.log2(N))) if N > 0 else 1
    
    num_blocks = math.ceil(N / TILE_SIZE)
    grid = (num_blocks, 1, 1)
    
    # Result array size of num_blocks
    result_partials = torch.zeros(num_blocks, dtype=a.dtype, device=a.device)
    
    ct.launch(
        torch.cuda.current_stream(),
        grid,
        dot_product_no_atomic_kernel,
        (a, b, result_partials, TILE_SIZE)
    )
    
    # Reduce on host (sum the partials)
    # result_partials.sum() returns a 0-d tensor, we might want to return 1-d scalar tensor to match dot_product signature behavior if needed
    # dot_product returns shape (1,)
    return result_partials.sum().unsqueeze(0)


def benchmark(kernel_func, kernel_name, output_filename):
    import time
    import matplotlib.pyplot as plt
    
    tile_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    powers = range(7, 30) # 2^7 to 2^20
    problem_sizes = [2**p for p in powers]
    
    plt.figure(figsize=(12, 8))
    
    print(f"Benchmarking {kernel_name} with tile sizes: {tile_sizes}")
    
    for tile_size in tile_sizes:
        print(f"\nTesting tile_size = {tile_size}")
        speedups = []
        
        for N in problem_sizes:
            a = torch.randn(N, dtype=torch.float32, device='cuda')
            b = torch.randn(N, dtype=torch.float32, device='cuda')
            
            # Ground Truth
            torch_gt = torch.dot(a, b)
            
            # Warmup and Correctness Check (cuTile)
            try:
                for _ in range(10):
                    cutile_res = kernel_func(a, b, tile_size=tile_size)
                    torch.testing.assert_close(cutile_res[0], torch_gt, rtol=1e-4, atol=1e-4)
            except Exception as e:
                print(f"Error for N={N}, tile_size={tile_size}: {e}")
            
            # Timing cuTile
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            start_event.record()
            for _ in range(100):
                kernel_func(a, b, tile_size=tile_size)
            end_event.record()
            torch.cuda.synchronize()
            cutile_time = start_event.elapsed_time(end_event) / 100
            
            # Warmup PyTorch
            for _ in range(10):
                torch.dot(a, b)
            
            # Timing PyTorch
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(100):
                torch.dot(a, b)
            end_event.record()
            torch.cuda.synchronize()
            torch_time = start_event.elapsed_time(end_event) / 100
            
            speedup = torch_time / cutile_time if cutile_time > 0 else 0.0
            speedups.append(speedup)
            print(f"N={N:<8} | Speedup={speedup:.2f}x")
        
        plt.plot(problem_sizes, speedups, label=f'Tile Size {tile_size}', marker='o')

    plt.xscale('log', base=2)
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Normalized Speedup (PyTorch / cuTile)')
    plt.title(f'{kernel_name} Speedup vs Problem Size for Different Tile Sizes')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig(output_filename)
    print(f"\nBenchmark complete for {kernel_name}. Plot saved to {output_filename}")

def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tile-size",
        default=2048,
        type=int,          # Ensures command line input is converted to int
        help="Size of the tile"
    )
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
        default=True,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark varying N from 128 to 2^20",
        default=False,
    )
    args = parser.parse_args()

    if args.benchmark:
        print("\n\n=== Benchmarking Atomic Dot Product ===")
        benchmark(dot_product, "Dot Product (Atomic)", "dot_product_atomic.png")
        
        print("\n\n=== Benchmarking Non-Atomic Dot Product ===")
        benchmark(dot_product_no_atomic, "Dot Product (No Atomic)", "dot_product_no_atomic.png")
        return

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

