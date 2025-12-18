import cupy as cp
import numpy as np
import cuda.tile as ct

@ct.kernel
def dot_product_kernel(a, b, result, tile_size: ct.Constant[int]):
    pid = ct.bid(0)
    
    # Load input tiles
    a_tile = ct.load(a, index=(pid,), shape=(tile_size,))
    b_tile = ct.load(b, index=(pid,), shape=(tile_size,))
    
    # Element-wise multiplication
    prod = a_tile * b_tile
    
    # Reduce tile to a single scalar sum
    partial_sum = ct.sum(prod)
    
    # Atomically add partial sum to global result
    # FIX: Pass arguments positionally: (array, indices, update)
    ct.atomic_add(result, (0,), partial_sum)

def run_test():
    N = 1024 * 1024
    TILE_SIZE = 1024
    
    # Generate Data
    a_gpu = cp.random.uniform(-1.0, 1.0, N).astype(cp.float32)
    b_gpu = cp.random.uniform(-1.0, 1.0, N).astype(cp.float32)
    result_gpu = cp.zeros(1, dtype=cp.float32)
    
    # Launch Config
    grid = (ct.cdiv(N, TILE_SIZE), 1, 1)
    
    print("Launching Dot Product kernel...")
    ct.launch(
        cp.cuda.get_current_stream(),
        grid,
        dot_product_kernel,
        (a_gpu, b_gpu, result_gpu, TILE_SIZE)
    )
    
    # Verification
    cutile_res = float(result_gpu.get())
    numpy_res = float(np.dot(cp.asnumpy(a_gpu), cp.asnumpy(b_gpu)))
    
    print(f"cuTile: {cutile_res:.4f}")
    print(f"Numpy:  {numpy_res:.4f}")
    np.testing.assert_allclose(cutile_res, numpy_res, rtol=1e-4)
    print("✓ Dot Product Passed")

if __name__ == "__main__":
    run_test()