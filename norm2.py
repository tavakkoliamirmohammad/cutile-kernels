import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def sum_of_squares_kernel(a, result, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    
    # Load input tile
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    
    # Square elements
    sq_tile = a_tile * a_tile
    
    # Reduce tile to scalar sum
    partial_sum = ct.sum(sq_tile)
    
    # Atomically add partial sum to global result
    # FIX: Pass arguments positionally
    ct.atomic_add(result, (0,), partial_sum)

def run_test():
    N = 1024 * 1024
    TILE_SIZE = 1024
    
    # Generate Data
    a_gpu = cp.random.uniform(-1.0, 1.0, N).astype(cp.float32)
    result_gpu = cp.zeros(1, dtype=cp.float32)
    
    # Launch Config
    grid = (ct.cdiv(N, TILE_SIZE), 1, 1)
    
    print("Launching Norm 2 (Sum of Squares) kernel...")
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        sum_of_squares_kernel,
        (a_gpu, result_gpu, TILE_SIZE)
    )
    
    # Finalize L2 Norm on Host: sqrt(sum_of_squares)
    sum_sq = float(result_gpu.get())
    cutile_res = np.sqrt(sum_sq)
    
    # Numpy Verification
    numpy_res = float(np.linalg.norm(cp.asnumpy(a_gpu)))
    
    print(f"cuTile: {cutile_res:.4f}")
    print(f"Numpy:  {numpy_res:.4f}")
    np.testing.assert_allclose(cutile_res, numpy_res, rtol=1e-4)
    print("✓ Norm 2 Passed")

if __name__ == "__main__":
    run_test()