import cupy as cp
import numpy as np
import cuda.tile as ct

# Configuration
TILE_SIZE = 128
# Use a large enough size to measure bandwidth
VECTOR_SIZE = 1024 * 1024 * 64  # 64M elements 
ITERATIONS = 20

@ct.kernel
def stencil_1d(left_arr, center_arr, right_arr, output_arr, tile_size: ct.Constant[int]):
    # 1. Get Tile Index
    # pid corresponds to the index in "Tile Space"
    # If pid=0, we want the 0th tile from all input arrays
    pid = ct.bid(0)
    
    # 2. Load Tiles
    # We load from 3 different array views. 
    # Because the views are shifted in memory (by python slicing), 
    # loading index=(pid,) from 'left_arr' automatically gets us the [i-1] data.
    
    # Shape of all loads is (TILE_SIZE,)
    left   = ct.load(left_arr,   index=(pid,), shape=(tile_size,))
    center = ct.load(center_arr, index=(pid,), shape=(tile_size,))
    right  = ct.load(right_arr,  index=(pid,), shape=(tile_size,))

    # 3. Compute Stencil
    # Formula: 0.25 * (left + 2*center + right)
    # This happens entirely in registers
    result = (left + (center * 2) + right) * 0.25

    # 4. Store Result
    ct.store(output_arr, index=(pid,), tile=result)

def run_benchmark():
    print(f"Running Stencil Benchmark (Size: {VECTOR_SIZE}, Tile: {TILE_SIZE})")

    # 1. Allocate Data
    # Total size includes padding for the halo (1 element on each side)
    # [ P | ... Valid Data ... | P ]
    total_size = VECTOR_SIZE + 2
    
    host_data = np.random.rand(total_size).astype(np.float32)
    host_out  = np.zeros(VECTOR_SIZE, dtype=np.float32)

    dev_data = cp.asarray(host_data)
    dev_out  = cp.asarray(host_out)

    # 2. Create Shifted Views
    # We want `center[i]` to line up with `left[i-1]` and `right[i+1]`.
    # Since ct.load(idx) -> array[idx * tile_size], we shift the base pointers:
    
    # Left View:   Starts at index 0. (Contains elements -1 relative to center)
    # Center View: Starts at index 1. (The valid domain)
    # Right View:  Starts at index 2. (Contains elements +1 relative to center)
    
    # Python slicing creates a view with a new pointer offset
    left_view   = dev_data[0 : -2]
    center_view = dev_data[1 : -1]
    right_view  = dev_data[2 : ]
    
    # 3. Grid Configuration
    # We process the valid domain (VECTOR_SIZE)
    grid_dim = ct.cdiv(VECTOR_SIZE, TILE_SIZE)
    grid = (grid_dim, 1, 1)

    # 4. Warmup
    print("Warming up...")
    ct.launch(cp.cuda.get_current_stream(), grid, stencil_1d, 
              (left_view, center_view, right_view, dev_out, TILE_SIZE))
    cp.cuda.Device().synchronize()

    # 5. Benchmark Loop
    print(f"Executing {ITERATIONS} iterations...")
    start_event = cp.cuda.Event()
    end_event = cp.cuda.Event()

    start_event.record()
    for _ in range(ITERATIONS):
        ct.launch(cp.cuda.get_current_stream(), grid, stencil_1d, 
                  (left_view, center_view, right_view, dev_out, TILE_SIZE))
    end_event.record()
    end_event.synchronize()

    # 6. Metrics
    total_time_ms = cp.cuda.get_elapsed_time(start_event, end_event)
    avg_time_ms = total_time_ms / ITERATIONS
    
    # Bandwidth Calculation:
    # We read 3 arrays and write 1 array of size VECTOR_SIZE.
    # (In reality, cache hits reduce read BW, but this is the 'required' BW)
    total_bytes = VECTOR_SIZE * 4 * 4  # 3 Reads + 1 Write * 4 bytes
    throughput = (total_bytes / 1e9) / (avg_time_ms / 1000)

    print(f"Average Time: {avg_time_ms:.4f} ms")
    print(f"Throughput:   {throughput:.2f} GB/s")

    # 7. Verification
    print("Verifying correctness (subset)...")
    check_size = 1024
    
    # CPU Reference
    # host_data[0:check_size] is "left"
    # host_data[1:check_size+1] is "center"
    # host_data[2:check_size+2] is "right"
    h_left   = host_data[0:check_size]
    h_center = host_data[1:check_size+1]
    h_right  = host_data[2:check_size+2]
    
    expected = (h_left + 2*h_center + h_right) * 0.25
    
    # GPU Result
    result_gpu = cp.asnumpy(dev_out[:check_size])

    if np.allclose(result_gpu, expected, atol=1e-5):
        print("PASS: Results match.")
    else:
        print("FAIL: Results mismatch.")

if __name__ == "__main__":
    run_benchmark()