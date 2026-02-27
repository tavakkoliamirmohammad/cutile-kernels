import os
import subprocess
import glob

# Files to exclude
exclude_files = [
    "tensor_contraction_gen.py",
    "stencil1d.py",
    "moe.py",
    "fft.py"
    "_generated_kernel_tc_bchw_ck_to_bhwk.py",
    "run_benchmarks.py",
    "patch_scripts.py",
    "run_all.py",
    "plot_results.py",
]

# Get all python benchmark files
all_files = glob.glob("*.py")
benchmark_files = [f for f in all_files if f not in exclude_files and f != "fft.py"] # Ignoring fft.py as it behaves completely differently

precisions = ["float64", "float32", "float16"]

for b in benchmark_files:
    for p in precisions:
        cmd = f"python {b} --precision {p}"
        print(f"Running: {cmd}")
        subprocess.run(cmd, shell=True)
