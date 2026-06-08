"""
Plot combined time series data for all parameter combinations on the same graph.
Shows h vs t, x vs t, and their log-log versions in a 2x2 layout.

Configuration:
- Set BETA_SUBSET and PE_SUBSET below to choose which parameters to plot
- Leave as None to use all available parameters
"""
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['mathtext.fontset'] = 'dejavusans'
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, '.')
from postprocess_functions import plt_settings, style_axis, find_neck

sweep_dir = "sweep_beta_Pe_results"

# =====================================================================
# CONFIGURATION: Specify which parameters to plot
# =====================================================================
# Set to None to use ALL available parameters
# Set to specific values to use a subset (e.g., [0, 0.1] or [1, 10, 100])
BETA_SUBSET = None   # All available: [0, 0.1, 0.5, 0.9]
PE_SUBSET = None     # All available: [1, 10, 100, 1000, 10000]

# Detect available parameters
print("Scanning for available parameters...")

available_params = []  # List of (folder_name, beta, Pe) tuples

for item in sorted(os.listdir(sweep_dir)):
    if os.path.isdir(os.path.join(sweep_dir, item)) and item.startswith('beta_'):
        parts = item.split('_')
        try:
            # Handle both "beta_0_Pe_1" and "beta_0.1_Pe_1" formats
            if 'Pe' in parts:
                Pe_idx = parts.index('Pe')
                beta_str = '_'.join(parts[1:Pe_idx])
                beta = float(beta_str)
                Pe = int(parts[Pe_idx + 1])
                domain_dir = os.path.join(sweep_dir, item, 'domain')
                if os.path.exists(domain_dir):
                    available_params.append((item, beta, Pe))
        except:
            pass

# Extract unique beta and Pe values
available_beta = sorted(list(set([p[1] for p in available_params])))
available_Pe = sorted(list(set([p[2] for p in available_params])))

beta_to_use = BETA_SUBSET if BETA_SUBSET else available_beta
Pe_to_use = PE_SUBSET if PE_SUBSET else available_Pe

print(f"\nAvailable parameters:")
print(f"  Beta values: {available_beta}")
print(f"  Pe values: {available_Pe}")
print(f"\nPlotting with:")
print(f"  Beta subset: {beta_to_use}")
print(f"  Pe subset: {Pe_to_use}\n")

# Filter parameters to plot
test_params = [(f, b, P) for f, b, P in available_params if b in beta_to_use and P in Pe_to_use]

print(f"Total parameter combinations to plot: {len(test_params)}\n")

fig = plt.figure(figsize=(14, 10))
gs = plt.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)

ax_h_lin = fig.add_subplot(gs[0, 0])
ax_x_lin = fig.add_subplot(gs[0, 1])
ax_h_log = fig.add_subplot(gs[1, 0])
ax_x_log = fig.add_subplot(gs[1, 1])

colors = ['red', 'blue', 'green', 'orange']

for idx, (folder_name, beta, Pe) in enumerate(test_params):
    domain_dir = os.path.join(sweep_dir, folder_name, 'domain')
    
    print(f"\nLoading {folder_name}...")
    
    if not os.path.exists(domain_dir):
        print(f"  Skipped: directory not found")
        continue
    
    try:
        files = sorted([f for f in os.listdir(domain_dir) if f.endswith('.txt')])[:50]  # Limit to first 50 files
        
        time_data = []
        h0_data = []
        x0_data = []
        
        for f in files:
            with open(os.path.join(domain_dir, f)) as file:
                header = file.readline()
                time = float(header.split('@time=')[-1])
                data = np.loadtxt(file)
                x = data[:, 0]
                h = data[:, 1]
                
                time_data.append(time)
                x0, h0 = find_neck(x, h)
                x0_data.append(x0)
                h0_data.append(h0)
        
        time_data = np.array(time_data)
        h0_data = np.array(h0_data)
        x0_data = np.array(x0_data)
        
        # Filter
        valid_time_mask = time_data > 0
        time_data = time_data[valid_time_mask]
        h0_data = h0_data[valid_time_mask]
        x0_data = x0_data[valid_time_mask]
        
        valid_x0_mask = np.abs(x0_data) > 1e-3
        time_data = time_data[valid_x0_mask]
        h0_data = h0_data[valid_x0_mask]
        x0_data = x0_data[valid_x0_mask]
        
        print(f"  Loaded {len(time_data)} timesteps")
        
        label = f'β={beta}, Pe={Pe}'
        color = colors[idx]
        
        # Linear
        ax_h_lin.scatter(time_data, h0_data, c=color, s=20, alpha=0.6, label=label)
        ax_x_lin.scatter(time_data, x0_data, c=color, s=20, alpha=0.6, label=label)
        
        # Log-log
        log_time = np.log(time_data)
        log_h0 = np.log(np.maximum(h0_data, 1e-10))
        log_x0 = np.log(np.maximum(np.abs(x0_data), 1e-10))
        
        ax_h_log.scatter(log_time, log_h0, c=color, s=20, alpha=0.6, label=label)
        ax_x_log.scatter(log_time, log_x0, c=color, s=20, alpha=0.6, label=label)
        
    except Exception as e:
        print(f"  Error: {e}")

# Style
style_axis(ax_h_lin, xlabel='Time t', ylabel='h₀', title='Neck Height (Linear)')
ax_h_lin.legend(fontsize=9)
ax_h_lin.grid(True, alpha=0.3)

style_axis(ax_x_lin, xlabel='Time t', ylabel='x₀', title='Neck Position (Linear)')
ax_x_lin.legend(fontsize=9)
ax_x_lin.grid(True, alpha=0.3)

style_axis(ax_h_log, xlabel='log(t)', ylabel='log(h₀)', title='Neck Height (Log-Log)')
ax_h_log.legend(fontsize=9)
ax_h_log.grid(True, which='both', alpha=0.3)

style_axis(ax_x_log, xlabel='log(t)', ylabel='log(|x₀|)', title='Neck Position (Log-Log)')
ax_x_log.legend(fontsize=9)
ax_x_log.grid(True, which='both', alpha=0.3)

plot_dir = "sweep_comparison_plots"
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

outfile = f'{plot_dir}/combined_timeseries_all_parameters.png'
plt.savefig(outfile, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to: {outfile}")
print("\nPlot layout:")
print("  - Top-left: h₀(t) linear scale")
print("  - Top-right: x₀(t) linear scale")
print("  - Bottom-left: log(h₀) vs log(t)")
print("  - Bottom-right: log(|x₀|) vs log(t)")
print(f"\nLoaded {len([p for p in test_params])} parameter combinations")
print("\nTo change which parameters are plotted, edit:")
print("  BETA_SUBSET = [0, 0.1]  # For example")
print("  PE_SUBSET = [1, 10, 100]  # For example")
