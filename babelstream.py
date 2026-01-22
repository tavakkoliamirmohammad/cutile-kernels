import argparse
import cuda.tile as ct
import cupy as cp
import time
import math
import torch

# -----------------------------------------------------------------------------
# CONSTANTS & CONFIG (Defaults)
# -----------------------------------------------------------------------------
DEFAULT_ARRAY_SIZE = 33554432
DEFAULT_TILE_SIZE  = 1024
DEFAULT_N_TIMES    = 100
SCALAR     = 0.4
INIT_A     = 2.0
INIT_B     = 0.5
INIT_C     = 0.0

# -----------------------------------------------------------------------------
# KERNELS
# -----------------------------------------------------------------------------
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
# VERIFICATION LOGIC
# -----------------------------------------------------------------------------
def run_verification(a_d, b_d, c_d, dot_result, n_times, scalar, array_size):
    print(f"\n{'='*25}\nVerification Phase\n{'='*25}")
    gold_a, gold_b, gold_c = INIT_A, INIT_B, INIT_C
    
    # Simulate on CPU
    for _ in range(n_times):
        gold_c = gold_a
        gold_b = scalar * gold_c
        gold_c = gold_a + gold_b
        gold_a = gold_b + scalar * gold_c
        
    gold_sum = array_size * (gold_a * gold_b)
    
    # Retrieve data from GPU
    try:
        gpu_a = float(a_d[0])
        gpu_b = float(b_d[0])
        gpu_c = float(c_d[0])
        gpu_sum = float(dot_result)
    except:
        gpu_a = float(a_d[0].get())
        gpu_b = float(b_d[0].get())
        gpu_c = float(c_d[0].get())
        gpu_sum = float(dot_result)

    errors = []
    epsilon = 1e-4
    
    if abs(gpu_a - gold_a) > epsilon:
        errors.append(f"Array A: Expected {gold_a:.6f}, Got {gpu_a:.6f}")
    if abs(gpu_b - gold_b) > epsilon:
        errors.append(f"Array B: Expected {gold_b:.6f}, Got {gpu_b:.6f}")
    if abs(gpu_c - gold_c) > epsilon:
        errors.append(f"Array C: Expected {gold_c:.6f}, Got {gpu_c:.6f}")
    
    if abs((gpu_sum - gold_sum)/gold_sum) > epsilon:
        errors.append(f"Dot Sum: Expected {gold_sum:.6e}, Got {gpu_sum:.6e}")

    if not errors:
        print("[SUCCESS] All results match expected values.")
    else:
        print("[FAILURE] Verification errors found:")
        for e in errors: print(e)

# -----------------------------------------------------------------------------
# MAIN BENCHMARK DRIVER
# -----------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# BENCHMARKING LOGIC
# -----------------------------------------------------------------------------
def run_benchmark_op(op_name, kernel_wrapper, torch_wrapper, problem_sizes, tile_sizes, output_filename):
    """
    Generic benchmark driver for a single operation.
    kernel_wrapper: function(a, b, c, scalar, tile_size) -> result (or None)
    torch_wrapper: function(a, b, c, scalar) -> result (or None)
    """
    import matplotlib.pyplot as plt
    
    print(f"\nBenchmarking {op_name}...")
    plt.figure(figsize=(12, 8))
    
    for tile_size in tile_sizes:
        print(f"  Tile Size: {tile_size}")
        speedups = []
        
        for N in problem_sizes:
            # Data setup (PyTorch)
            try:
                a = torch.full((N,), INIT_A, dtype=torch.float64, device='cuda')
                b = torch.full((N,), INIT_B, dtype=torch.float64, device='cuda')
                c = torch.full((N,), INIT_C, dtype=torch.float64, device='cuda')
            except torch.cuda.OutOfMemoryError:
                print(f"    N={N}: OOM (Skipping)")
                speedups.append(0.0)
                continue

            # Warmup & Correctness
            try:
                # Run cuTile
                ct_res = kernel_wrapper(a, b, c, SCALAR, tile_size)
                
                # Run PyTorch (on clones to verify)
                a_ref = a.clone(); b_ref = b.clone(); c_ref = c.clone()
                torch_res = torch_wrapper(a_ref, b_ref, c_ref, SCALAR)
                
                # Verify
                if op_name == 'Dot':
                     # For dot, result is scalar. 1e-4 might be too tight for large sums?
                     # Dot product of many numbers can accumulate error.
                     # But float64 should be okay.
                     torch.testing.assert_close(ct_res, torch_res, rtol=1e-4, atol=1e-4)
                else:
                    if op_name in ['Copy', 'Add']:
                        torch.testing.assert_close(c, c_ref, rtol=1e-4, atol=1e-4)
                    elif op_name == 'Mul':
                        torch.testing.assert_close(b, b_ref, rtol=1e-4, atol=1e-4)
                    elif op_name == 'Triad':
                        torch.testing.assert_close(a, a_ref, rtol=1e-4, atol=1e-4)

            except Exception as e:
                print(f"    N={N}: Error in functionality check: {e}")
                import traceback
                traceback.print_exc()
                speedups.append(0.0)
                continue
            
            # Reset Data for Timing
            a.fill_(INIT_A); b.fill_(INIT_B); c.fill_(INIT_C)
            
            # Measure cuTile
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Warmup
            for _ in range(5):
                 kernel_wrapper(a, b, c, SCALAR, tile_size)
            
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(20): # 20 iterations
                kernel_wrapper(a, b, c, SCALAR, tile_size)
            end_event.record()
            torch.cuda.synchronize()
            cutile_time = start_event.elapsed_time(end_event) / 20.0
            
            # Reset Data
            a.fill_(INIT_A); b.fill_(INIT_B); c.fill_(INIT_C)

            # Measure PyTorch
            # Warmup
            for _ in range(5):
                torch_wrapper(a, b, c, SCALAR)
                
            torch.cuda.synchronize()
            start_event.record()
            for _ in range(20):
                torch_wrapper(a, b, c, SCALAR)
            end_event.record()
            torch.cuda.synchronize()
            torch_time = start_event.elapsed_time(end_event) / 20.0
            
            speedup = torch_time / cutile_time if cutile_time > 0 else 0.0
            speedups.append(speedup)
            print(f"    N={N:<10} | Speedup={speedup:.2f}x (Torch: {torch_time:.4f}ms, cuTile: {cutile_time:.4f}ms)")
            
        plt.plot(problem_sizes[:len(speedups)], speedups, label=f'Tile {tile_size}', marker='o')

    plt.xscale('log', base=2)
    plt.xlabel('Problem Size (N)')
    plt.ylabel('Speedup (PyTorch / cuTile)')
    plt.title(f'{op_name} Speedup')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")


def benchmark_all():
    tile_sizes = [128, 256, 512, 1024, 2048, 4096, 8192]
    # Powers 7 to 26 (128 to ~67M). 
    powers = range(7, 26) 
    problem_sizes = [2**p for p in powers]
    
    # 1. Copy: c = a
    def run_copy(a, b, c, scalar, tile_size):
        N = a.shape[0]
        blocks = (N + tile_size - 1) // tile_size
        ct.launch(torch.cuda.current_stream(), (blocks, 1, 1), k_copy, (a, c, tile_size))
    
    def run_copy_torch(a, b, c, scalar):
        c.copy_(a)

    run_benchmark_op("Copy", run_copy, run_copy_torch, problem_sizes, tile_sizes, "babelstream_Copy.png")

    # 2. Mul: b = scalar * c
    def run_mul(a, b, c, scalar, tile_size):
        N = a.shape[0]
        blocks = (N + tile_size - 1) // tile_size
        ct.launch(torch.cuda.current_stream(), (blocks, 1, 1), k_mul, (b, c, scalar, tile_size))
        
    def run_mul_torch(a, b, c, scalar):
        b.copy_(c * scalar)

    run_benchmark_op("Mul", run_mul, run_mul_torch, problem_sizes, tile_sizes, "babelstream_Mul.png")

    # 3. Add: c = a + b
    def run_add(a, b, c, scalar, tile_size):
        N = a.shape[0]
        blocks = (N + tile_size - 1) // tile_size
        ct.launch(torch.cuda.current_stream(), (blocks, 1, 1), k_add, (a, b, c, tile_size))

    def run_add_torch(a, b, c, scalar):
        c.copy_(a + b)

    run_benchmark_op("Add", run_add, run_add_torch, problem_sizes, tile_sizes, "babelstream_Add.png")

    # 4. Triad: a = b + scalar * c
    def run_triad(a, b, c, scalar, tile_size):
        N = a.shape[0]
        blocks = (N + tile_size - 1) // tile_size
        ct.launch(torch.cuda.current_stream(), (blocks, 1, 1), k_triad, (a, b, c, scalar, tile_size))

    def run_triad_torch(a, b, c, scalar):
        a.copy_(b + scalar * c)

    run_benchmark_op("Triad", run_triad, run_triad_torch, problem_sizes, tile_sizes, "babelstream_Triad.png")

    # 5. Dot: sum(a * b)
    def run_dot(a, b, c, scalar, tile_size):
        N = a.shape[0]
        blocks = (N + tile_size - 1) // tile_size
        # We need partial sums tensor
        partials = torch.empty(blocks, dtype=a.dtype, device=a.device)
        ct.launch(torch.cuda.current_stream(), (blocks, 1, 1), k_dot, (a, b, partials, tile_size))
        return partials.sum()

    def run_dot_torch(a, b, c, scalar):
        return torch.dot(a, b)

    run_benchmark_op("Dot", run_dot, run_dot_torch, problem_sizes, tile_sizes, "babelstream_Dot.png")


def main():
    parser = argparse.ArgumentParser(description="BabelStream Benchmark (cuTile)")
    parser.add_argument("--array-size", type=int, default=DEFAULT_ARRAY_SIZE, help="Size of the arrays")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Tile size for kernels")
    parser.add_argument("--iterations", type=int, default=DEFAULT_N_TIMES, help="Number of benchmark iterations")
    parser.add_argument("--correctness-check", action="store_true", default=True, help="Run verification checks")
    
    parser.add_argument("--benchmark", action="store_true", help="Run full benchmark suite")
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_all()
        return

    array_size = args.array_size
    tile_size = args.tile_size
    n_times = args.iterations
    
    blocks = (array_size + tile_size - 1) // tile_size
    
    print(f"BabelStream (cuTile Python)")
    print(f"Array Size: {array_size}")
    print(f"Tile Size:  {tile_size}")
    print(f"Iterations: {n_times}")

    # 1. Allocation
    a_d = cp.full(array_size, INIT_A, dtype=cp.float64)
    b_d = cp.full(array_size, INIT_B, dtype=cp.float64)
    c_d = cp.full(array_size, INIT_C, dtype=cp.float64)
    dot_partials_d = cp.zeros(blocks, dtype=cp.float64)
    
    grid_dim = (blocks, 1, 1)

    # 2. Warmup
    stream_ptr = cp.cuda.get_current_stream().ptr
    
    ct.launch(stream_ptr, grid_dim, k_copy, (a_d, c_d, tile_size))
    ct.launch(stream_ptr, grid_dim, k_mul, (b_d, c_d, SCALAR, tile_size))
    ct.launch(stream_ptr, grid_dim, k_add, (a_d, b_d, c_d, tile_size))
    ct.launch(stream_ptr, grid_dim, k_triad, (a_d, b_d, c_d, SCALAR, tile_size))
    ct.launch(stream_ptr, grid_dim, k_dot, (a_d, b_d, dot_partials_d, tile_size))
    
    cp.cuda.Device().synchronize()

    # Re-init
    a_d[:] = INIT_A; b_d[:] = INIT_B; c_d[:] = INIT_C

    # 3. Benchmark Loop
    timings = {'Copy': [], 'Mul': [], 'Add': [], 'Triad': [], 'Dot': []}
    last_dot_sum = 0.0

    print("\nRunning benchmark...")

    for k in range(n_times):
        # --- Copy ---
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        stream_ptr = cp.cuda.get_current_stream().ptr
        ct.launch(stream_ptr, grid_dim, k_copy, (a_d, c_d, tile_size))
        cp.cuda.Device().synchronize()
        timings['Copy'].append(time.perf_counter() - t0)
        
        # --- Mul ---
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        stream_ptr = cp.cuda.get_current_stream().ptr
        ct.launch(stream_ptr, grid_dim, k_mul, (b_d, c_d, SCALAR, tile_size))
        cp.cuda.Device().synchronize()
        timings['Mul'].append(time.perf_counter() - t0)
        
        # --- Add ---
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        stream_ptr = cp.cuda.get_current_stream().ptr
        ct.launch(stream_ptr, grid_dim, k_add, (a_d, b_d, c_d, tile_size))
        cp.cuda.Device().synchronize()
        timings['Add'].append(time.perf_counter() - t0)
        
        # --- Triad ---
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        stream_ptr = cp.cuda.get_current_stream().ptr
        ct.launch(stream_ptr, grid_dim, k_triad, (a_d, b_d, c_d, SCALAR, tile_size))
        cp.cuda.Device().synchronize()
        timings['Triad'].append(time.perf_counter() - t0)
        
        # --- Dot ---
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        stream_ptr = cp.cuda.get_current_stream().ptr
        ct.launch(stream_ptr, grid_dim, k_dot, (a_d, b_d, dot_partials_d, tile_size))
        cp.cuda.Device().synchronize()
        last_dot_sum = cp.sum(dot_partials_d)
        timings['Dot'].append(time.perf_counter() - t0)

    # 4. Results
    sizes = {
        'Copy': 2 * array_size * 8,
        'Mul':  2 * array_size * 8,
        'Add':  3 * array_size * 8,
        'Triad': 3 * array_size * 8,
        'Dot':  2 * array_size * 8
    }

    print(f"\n{'Function':<10} | {'MBytes/sec':<15} | {'Min (sec)':<10} | {'Max (sec)':<10}")
    print("-" * 55)
    for name in ['Copy', 'Mul', 'Add', 'Triad', 'Dot']:
        times = sorted(timings[name])
        min_t = times[0]
        max_t = times[-1]
        bw = (sizes[name] / 1e6) / min_t
        print(f"{name:<10} | {bw:<15.2f} | {min_t:<10.5f} | {max_t:<10.5f}")

    # 5. Execute Verification
    if args.correctness_check:
        run_verification(a_d, b_d, c_d, last_dot_sum, n_times, SCALAR, array_size)
    else:
        print("\nVerification skipped.")

if __name__ == "__main__":
    main()