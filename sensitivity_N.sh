#!/bin/bash
# Sensitivity analysis for mesh resolution N
# Varies N from 500, 1000, 2000 (N/2 < N < 2N) with default surfactant parameters

set -e  # Exit on error

echo "=============================================="
echo "Sensitivity Analysis: Mesh Resolution"
echo "=============================================="

# Default surfactant parameters
BETA=0.1
PE=1.0
GAMMA0=0.8
THETA=20
HP=1e-4

# Mesh resolution values to test (N/2, N, 2N where N=1000)
N_VALUES=("500" "1000" "2000")

# Run simulations
for N in "${N_VALUES[@]}"; do
    OUTDIR="sensitivity_N_${N}"

    echo ""
    echo "Running simulation with N = $N"
    echo "Output directory: $OUTDIR"
    echo "----------------------------------------------"

    # Remove old output if exists
    rm -rf "$OUTDIR"

    # Run simulation with output directory
    python coalescence.py \
        --beta $BETA \
        --Pe $PE \
        --Gamma0 $GAMMA0 \
        --theta $THETA \
        --hp $HP \
        --N $N \
        --output-dir "$OUTDIR" 2>&1 | tee "${OUTDIR}_log.txt"

    echo "Completed simulation for N = $N"
done

echo ""
echo "All simulations completed!"
echo "Generating comparison plots..."

# Create comparison plot using Python
python << 'PYTHON_SCRIPT'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

# Publication-quality settings
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['font.serif'] = ['Computer Modern Roman']
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
matplotlib.rcParams['figure.dpi'] = 150
matplotlib.rcParams['lines.linewidth'] = 2.5

plt_settings = {
    'LabelFont': 24,
    'AxesFont': 18,
    'TitleFont': 24,
    'LegendFont': 16,
}

def style_axis(ax, xlabel=None, ylabel=None, title=None):
    ax.tick_params(axis='both', which='major', labelsize=plt_settings['AxesFont'],
                   width=2, length=8, direction='out', pad=8)
    ax.tick_params(which='minor', width=1.5, length=4, direction='out')
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    ax.minorticks_on()
    ax.grid(True, alpha=0.3, linewidth=1)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=plt_settings['LabelFont'], labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=plt_settings['LabelFont'], labelpad=10)
    if title:
        ax.set_title(title, fontsize=plt_settings['TitleFont'], pad=15)

# Mesh resolution values
N_values = ['500', '1000', '2000']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
labels = [r'$N = 500$', r'$N = 1000$', r'$N = 2000$']

# Create output directory for plots
plot_dir = 'sensitivity_N_plots'
os.makedirs(plot_dir, exist_ok=True)

# Extract time series data for each simulation
all_data = {}

for N, color, label in zip(N_values, colors, labels):
    outdir = f'sensitivity_N_{N}'
    domain_dir = os.path.join(outdir, 'domain')
    if not os.path.exists(domain_dir):
        print(f"Warning: {domain_dir} not found, skipping")
        continue

    files = sorted([f for f in os.listdir(domain_dir) if f.endswith('.txt')])

    time_data = []
    h0_data = []      # Bridge height at x=0 (minimum height)
    x0_data = []      # Position of minimum height
    xf_data = []      # Surfactant front position

    for f in files:
        with open(os.path.join(domain_dir, f)) as file:
            header = file.readline()
            time = float(header.split('@time=')[-1])
            data = np.loadtxt(file)

            x = data[:, 0]
            h = data[:, 1]
            gamma = data[:, 3] if data.shape[1] > 3 else np.zeros_like(h)

            # h0: minimum height (bridge height)
            h0 = np.min(h)

            # x0: position of minimum height
            x0 = x[np.argmin(h)]

            # xf: surfactant front position (where Gamma drops to 0.5*Gamma_max)
            gamma_max = np.max(gamma)
            if gamma_max > 0.01:
                # Find where gamma crosses 0.5 * initial value (0.8 * 0.5 = 0.4)
                threshold = 0.4
                above_threshold = gamma > threshold
                if np.any(above_threshold) and np.any(~above_threshold):
                    # Find rightmost crossing point
                    crossings = np.where(np.diff(above_threshold.astype(int)) == -1)[0]
                    if len(crossings) > 0:
                        xf = x[crossings[-1]]
                    else:
                        xf = x[np.argmax(x[above_threshold])]
                else:
                    xf = 0.0
            else:
                xf = 0.0

            time_data.append(time)
            h0_data.append(h0)
            x0_data.append(x0)
            xf_data.append(xf)

    all_data[N] = {
        'time': np.array(time_data),
        'h0': np.array(h0_data),
        'x0': np.array(x0_data),
        'xf': np.array(xf_data),
        'color': color,
        'label': label
    }

# Calculate sensitivity metrics (relative to baseline N=1000)
print("\n" + "="*60)
print("SENSITIVITY ANALYSIS: MESH RESOLUTION")
print("="*60)

if '1000' in all_data:
    baseline = all_data['1000']
    print(f"\nBaseline: N = 1000")
    print(f"  Final h0 = {baseline['h0'][-1]:.6f}")
    print(f"  Final x0 = {baseline['x0'][-1]:.6f}")
    print(f"  Final xf = {baseline['xf'][-1]:.6f}")

    for N in ['500', '2000']:
        if N in all_data:
            data = all_data[N]
            # Interpolate to common time points
            t_common = np.linspace(0, min(baseline['time'][-1], data['time'][-1]), 100)
            h0_base = np.interp(t_common, baseline['time'], baseline['h0'])
            h0_test = np.interp(t_common, data['time'], data['h0'])
            x0_base = np.interp(t_common, baseline['time'], baseline['x0'])
            x0_test = np.interp(t_common, data['time'], data['x0'])
            xf_base = np.interp(t_common, baseline['time'], baseline['xf'])
            xf_test = np.interp(t_common, data['time'], data['xf'])

            # Max relative difference
            h0_diff = np.max(np.abs(h0_test - h0_base) / (np.abs(h0_base) + 1e-10)) * 100
            x0_diff = np.max(np.abs(x0_test - x0_base) / (np.abs(x0_base) + 1e-10)) * 100
            xf_diff = np.max(np.abs(xf_test - xf_base) / (np.abs(xf_base) + 1e-10)) * 100

            print(f"\nN = {N}:")
            print(f"  Final h0 = {data['h0'][-1]:.6f}")
            print(f"  Final x0 = {data['x0'][-1]:.6f}")
            print(f"  Final xf = {data['xf'][-1]:.6f}")
            print(f"  Max relative difference in h0: {h0_diff:.2f}%")
            print(f"  Max relative difference in x0: {x0_diff:.2f}%")
            print(f"  Max relative difference in xf: {xf_diff:.2f}%")

# Plot 1: Bridge height h0(t)
fig1, ax1 = plt.subplots(figsize=(10, 8))
for N, data in all_data.items():
    ax1.plot(data['time'], data['h0'], color=data['color'], label=data['label'], linewidth=2.5)
style_axis(ax1, xlabel=r'Time $t$', ylabel=r'Bridge height $h_0$',
           title=r'Sensitivity to Mesh Resolution: $h_0(t)$')
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False)
ax1.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'{plot_dir}/h0_vs_time.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Minimum position x0(t)
fig2, ax2 = plt.subplots(figsize=(10, 8))
for N, data in all_data.items():
    ax2.plot(data['time'], data['x0'], color=data['color'], label=data['label'], linewidth=2.5)
style_axis(ax2, xlabel=r'Time $t$', ylabel=r'Position of minimum $x_0$',
           title=r'Sensitivity to Mesh Resolution: $x_0(t)$')
ax2.legend(fontsize=plt_settings['LegendFont'], frameon=False)
ax2.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'{plot_dir}/x0_vs_time.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Surfactant front position xf(t)
fig3, ax3 = plt.subplots(figsize=(10, 8))
for N, data in all_data.items():
    ax3.plot(data['time'], data['xf'], color=data['color'], label=data['label'], linewidth=2.5)
style_axis(ax3, xlabel=r'Time $t$', ylabel=r'Surfactant front position $x_f$',
           title=r'Sensitivity to Mesh Resolution: $x_f(t)$')
ax3.legend(fontsize=plt_settings['LegendFont'], frameon=False)
ax3.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'{plot_dir}/xf_vs_time.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Combined 3-panel plot
fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
for N, data in all_data.items():
    ax1.plot(data['time'], data['h0'], color=data['color'], label=data['label'], linewidth=2.5)
    ax2.plot(data['time'], data['x0'], color=data['color'], linewidth=2.5)
    ax3.plot(data['time'], data['xf'], color=data['color'], linewidth=2.5)

style_axis(ax1, ylabel=r'$h_0$', title=r'Sensitivity to Mesh Resolution $N$')
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='upper left')
style_axis(ax2, ylabel=r'$x_0$')
style_axis(ax3, xlabel=r'Time $t$', ylabel=r'$x_f$')
ax3.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'{plot_dir}/sensitivity_N_combined.pdf', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to '{plot_dir}/' directory")
PYTHON_SCRIPT

echo ""
echo "Sensitivity analysis complete!"
echo "Results saved to sensitivity_N_plots/"
