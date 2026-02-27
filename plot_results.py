import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("plots", exist_ok=True)

files = glob.glob("results/*.txt")
for f in files:
    basename = os.path.basename(f)
    name, ext = os.path.splitext(basename)
    
    with open(f, 'r') as file:
        content = file.read().strip()
    
    blocks = content.split('\n\n')
    
    for block_idx, block in enumerate(blocks):
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
            
        block_name = lines[0].replace(':', '').strip()
        if not block_name:
            block_name = f"{name}_part{block_idx}"
            
        # Parse the CSV from string
        import io
        csv_data = '\n'.join(lines[1:])
        try:
            df = pd.read_csv(io.StringIO(csv_data), sep=r'\s+')
        except Exception as e:
            print(f"Failed to parse block in {f}: {e}")
            continue
            
        has_torch = "Torch" in df.columns
            
        cutile_cols = [c for c in df.columns if 'cuTile' in c]
        if not cutile_cols:
            print(f"No cuTile columns in {block_name} of {f}, skipping plot.")
            continue
            
        # Convert all columns to numeric, coercing errors (like '-' to NaN)
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
            
        # Group columns by cuTile version
        from collections import defaultdict
        version_cols = defaultdict(list)
        for c in cutile_cols:
            parts = c.rsplit('-', 1)
            version = parts[0] if len(parts) > 1 else c
            version_cols[version].append(c)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        x_col = df.columns[0] # Usually 'N'
        x_values = df[x_col]
        
        if has_torch:
            plt.plot(x_values, df['Torch'], marker='o', label='Torch', color='tab:green', linewidth=2)
            
        # Plot best for each cuTile version
        colors = ['tab:blue', 'tab:orange', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray']
        
        for i, (version, cols) in enumerate(version_cols.items()):
            best_version_perf = df[cols].max(axis=1)
            color = colors[i % len(colors)]
            plt.plot(x_values, best_version_perf, marker='s', label=f'Best {version}', color=color, linewidth=2)
        
        plt.xscale('log', base=2)
        plt.xlabel(x_col)
        
        plt.ylabel("Performance")
        plt.title(f"Performance Comparison: {block_name}")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        
        plot_file = f"plots/{block_name}.png"
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {plot_file}")
