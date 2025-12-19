import subprocess
import sys
import os

def run_script(script_name, args=[]):
    print(f"==========================================")
    print(f"Running {script_name}...")
    print(f"==========================================")
    cmd = [sys.executable, script_name] + args
    try:
        result = subprocess.run(cmd, check=True, text=True, capture_output=False)
        print(f"\n{script_name} completed successfully.\n")
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}:")
        print(e)
        # Optionally continue or exit. Continuing seems safer for a "run all" script.

def main():
    scripts = [
        "babelstream.py",
        "batch_matmul.py",
        "dot_product.py",
        "fft.py",
        "layernorm.py",
        "matmul.py",
        "moe.py",
        "norm1.py",
        "norm2.py",
        "stencil1d.py",
        "tensor_contraction_gen.py",
        "transpose.py",
        "vecadd.py"
    ]

    for script in scripts:
        if os.path.exists(script):
            # For stencil1d.py, we can also run with --benchmark if desired, 
            # but the user asked to "run all python files", implying the default behavior.
            # Most scripts default to --correctness-check=True.
            run_script(script)
        else:
            print(f"Warning: {script} not found.")

if __name__ == "__main__":
    main()
