import argparse
import cuda.tile as ct
import cupy as cp
import time
import math

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
def main():
    parser = argparse.ArgumentParser(description="BabelStream Benchmark (cuTile)")
    parser.add_argument("--array-size", type=int, default=DEFAULT_ARRAY_SIZE, help="Size of the arrays")
    parser.add_argument("--tile-size", type=int, default=DEFAULT_TILE_SIZE, help="Tile size for kernels")
    parser.add_argument("--iterations", type=int, default=DEFAULT_N_TIMES, help="Number of benchmark iterations")
    parser.add_argument("--correctness-check", action="store_true", default=True, help="Run verification checks")
    
    args = parser.parse_args()
    
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