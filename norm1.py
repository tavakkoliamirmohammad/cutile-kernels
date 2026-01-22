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

@ct.kernel
def norm1_no_atomic_kernel(a, result, tile_size: ConstInt):
    """
    cuTile kernel for computing the L1 norm of a vector without atomics.
    Each block writes its partial sum to a specific index in the result array.
    """
    pid = ct.bid(0)
    
    # Load input tile
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    
    # Compute Absolute Value
    abs_tile = ct.maximum(a_tile, -a_tile)
    
    # Reduce tile to scalar sum
    partial_sum = ct.sum(abs_tile)
    
    # Write partial sum to result array at index pid
    ct.store(result, index=(pid,), tile=partial_sum)


def norm1(a: torch.Tensor, tile_size: int = None) -> torch.Tensor:
    """
    Wrapper for L1 norm kernel (Atomic).
    """
    if not a.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    
    N = a.shape[0]
    
    if tile_size is not None:
        TILE_SIZE = tile_size
    else:
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


def norm1_no_atomic(a: torch.Tensor, tile_size: int = None) -> torch.Tensor:
    """
    Wrapper for L1 norm kernel without atomics.
    Reduces on the host.
    """
    if not a.is_cuda:
        raise ValueError("Input tensor must be on a CUDA device.")
    
    N = a.shape[0]
    
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
        norm1_no_atomic_kernel,
        (a, result_partials, TILE_SIZE)
    )
    
    # Reduce on host (sum the partials)
    # Return shape (1,) to be consistent with norm1/torch.norm return type behavior if it was returning 0-d
    # But usually torch reduction returns 0-d tensor. Let's return 0-d wrapped in 1-d or just 0-d?
    # norm1 wrapper returns `result` which is initialized as size (1,). So let's return size (1,)
    return result_partials.sum().unsqueeze(0)


def benchmark(kernel_func, kernel_name, output_filename):
    import time
    import matplotlib.pyplot as plt
    
    tile_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    powers = range(20, 30) # 2^20 to 2^29 (approx 1M to 500M elements)
    problem_sizes = [2**p for p in powers]
    
    plt.figure(figsize=(12, 8))
    
    print(f"Benchmarking {kernel_name} with tile sizes: {tile_sizes}")
    
    for tile_size in tile_sizes:
        print(f"\nTesting tile_size = {tile_size}")
        speedups = []
        
        for N in problem_sizes:
            a = torch.randn(N, dtype=torch.float32, device='cuda')
            
            # Ground Truth
            torch_gt = torch.norm(a, p=1)
            
            # Warmup and Correctness Check (cuTile)
            try:
                for _ in range(5):
                    cutile_res = kernel_func(a, tile_size=tile_size)
                    torch.testing.assert_close(cutile_res[0], torch_gt, rtol=1e-4, atol=1e-4)
            except Exception as e:
                print(f"Error for N={N}, tile_size={tile_size}: {e}")
            
            # Timing cuTile
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(50):
                kernel_func(a, tile_size=tile_size)
            end_event.record()
            torch.cuda.synchronize()
            cutile_time = start_event.elapsed_time(end_event) / 50
            
            # Warmup PyTorch
            for _ in range(5):
                 torch.norm(a, p=1)
            
            # Timing PyTorch
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(50):
                 torch.norm(a, p=1)
            end_event.record()
            torch.cuda.synchronize()
            torch_time = start_event.elapsed_time(end_event) / 50
            
            speedup = torch_time / cutile_time if cutile_time > 0 else 0.0
            speedups.append(speedup)
            print(f"N={N:<12} | Speedup={speedup:.2f}x")
        
        plt.plot(problem_sizes, speedups, label=f'Tile Size {tile_size}', marker='o')

    plt.xscale('log', base=2)
    plt.xlabel('Vector Size (N)')
    plt.ylabel('Normalized Speedup (PyTorch / cuTile)')
    plt.title(f'{kernel_name} Speedup vs Vector Size')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    plt.savefig(output_filename)
    print(f"\nBenchmark complete for {kernel_name}. Plot saved to {output_filename}")


def run_test():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--correctness-check",
        action="store_true",
        help="Check the correctness of the results",
        default=True,
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark varying Vector Size N",
        default=False,
    )
    args = parser.parse_args()

    if args.benchmark:
        print("\n\n=== Benchmarking Norm1 (Atomic) ===")
        benchmark(norm1, "Norm1 (Atomic)", "norm1_atomic.png")
        
        print("\n\n=== Benchmarking Norm1 (Non-Atomic) ===")
        benchmark(norm1_no_atomic, "Norm1 (No Atomic)", "norm1_no_atomic.png")
        return

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
