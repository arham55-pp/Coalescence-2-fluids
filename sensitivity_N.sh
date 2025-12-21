#!/bin/bash
# Sensitivity analysis for mesh resolution N
# Runs 4 cases in parallel: 500, 1000, 2000, 4000 (mesh doubling study)
#
# Usage: ./sensitivity_N.sh [--Pe VALUE] [--beta VALUE]
#   --Pe VALUE    Péclet number (default: 1.0)
#   --beta VALUE  Surfactant strength (default: 0.1)

set -e  # Exit on error

# Default physical parameters (can be overridden via command line)
BETA=0.1
PE=1.0

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --Pe)
            PE="$2"
            shift 2
            ;;
        --beta)
            BETA="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--Pe VALUE] [--beta VALUE]"
            exit 1
            ;;
    esac
done

echo "=============================================="
echo "Sensitivity Analysis: Mesh Resolution"
echo "Physical parameters: Pe=$PE, beta=$BETA"
echo "=============================================="

# Fixed parameters for this study
GAMMA0=0.8
THETA=20
HP=1e-4

# Mesh resolution values to test (4 values, doubling each time)
N_VALUES=("500" "1000" "2000" "4000")

# Clean up old output directories
for N in "${N_VALUES[@]}"; do
    rm -rf "sensitivity_N_${N}"
done

# Launch all simulations in parallel
echo ""
echo "Launching ${#N_VALUES[@]} simulations in parallel..."
echo "----------------------------------------------"

PIDS=()
for N in "${N_VALUES[@]}"; do
    OUTDIR="sensitivity_N_${N}"
    python coalescence.py \
        --beta $BETA \
        --Pe $PE \
        --Gamma0 $GAMMA0 \
        --theta $THETA \
        --hp $HP \
        --N $N \
        --output-dir "$OUTDIR" > "${OUTDIR}_log.txt" 2>&1 &
    PIDS+=($!)
    echo "Started N = $N (PID: $!)"
done

# Wait for all simulations to complete
echo ""
echo "Waiting for all simulations to complete..."
for i in "${!PIDS[@]}"; do
    wait ${PIDS[$i]}
    echo "Completed N = ${N_VALUES[$i]}"
done

echo ""
echo "All simulations completed!"

# Run convergence analysis
echo ""
echo "Running convergence analysis..."
python check_convergence.py --N --Pe $PE --beta $BETA

echo ""
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

# Mesh resolution values (4 values)
N_values = ['500', '1000', '2000', '4000']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
labels = [r'$N = 500$', r'$N = 1000$', r'$N = 2000$', r'$N = 4000$']

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
    h0_data = []      # Neck height (min in bridge region |x| < 1.5)
    x0_data = []      # Position of neck minimum
    xe_data = []      # Drop edge position (where h drops below threshold)

    hp = 1e-4  # Precursor film thickness for this study

    for f in files:
        with open(os.path.join(domain_dir, f)) as file:
            header = file.readline()
            time = float(header.split('@time=')[-1])
            data = np.loadtxt(file)

            x = data[:, 0]
            h = data[:, 1]

            # Restrict to bridge region (|x| < 1.5) to exclude precursor film
            bridge_mask = np.abs(x) < 1.5
            x_bridge = x[bridge_mask]
            h_bridge = h[bridge_mask]

            # h0: neck height (minimum in bridge region)
            h0 = np.min(h_bridge)

            # x0: position of neck minimum
            x0 = x_bridge[np.argmin(h_bridge)]

            # xe: drop edge (where h drops below threshold on right side)
            threshold = max(2 * hp, 0.01)
            right_mask = x > 0
            h_right = h[right_mask]
            x_right = x[right_mask]
            below = h_right < threshold
            if np.any(below):
                xe = x_right[np.argmax(below)]
            else:
                xe = x_right[-1]

            time_data.append(time)
            h0_data.append(h0)
            x0_data.append(x0)
            xe_data.append(xe)

    all_data[N] = {
        'time': np.array(time_data),
        'h0': np.array(h0_data),
        'x0': np.array(x0_data),
        'xe': np.array(xe_data),
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
    print(f"  Final xe = {baseline['xe'][-1]:.6f}")

    for N in ['500', '2000', '4000']:
        if N in all_data:
            data = all_data[N]
            # Interpolate to common time points
            t_common = np.linspace(0, min(baseline['time'][-1], data['time'][-1]), 100)
            h0_base = np.interp(t_common, baseline['time'], baseline['h0'])
            h0_test = np.interp(t_common, data['time'], data['h0'])
            x0_base = np.interp(t_common, baseline['time'], baseline['x0'])
            x0_test = np.interp(t_common, data['time'], data['x0'])
            xe_base = np.interp(t_common, baseline['time'], baseline['xe'])
            xe_test = np.interp(t_common, data['time'], data['xe'])

            # Max relative difference
            h0_diff = np.max(np.abs(h0_test - h0_base) / (np.abs(h0_base) + 1e-10)) * 100
            x0_diff = np.max(np.abs(x0_test - x0_base) / (np.abs(x0_base) + 1e-10)) * 100
            xe_diff = np.max(np.abs(xe_test - xe_base) / (np.abs(xe_base) + 1e-10)) * 100

            print(f"\nN = {N}:")
            print(f"  Final h0 = {data['h0'][-1]:.6f}")
            print(f"  Final x0 = {data['x0'][-1]:.6f}")
            print(f"  Final xe = {data['xe'][-1]:.6f}")
            print(f"  Max relative difference in h0: {h0_diff:.2f}%")
            print(f"  Max relative difference in x0: {x0_diff:.2f}%")
            print(f"  Max relative difference in xe: {xe_diff:.2f}%")

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

# Plot 3: Drop edge position xe(t)
fig3, ax3 = plt.subplots(figsize=(10, 8))
for N, data in all_data.items():
    ax3.plot(data['time'], data['xe'], color=data['color'], label=data['label'], linewidth=2.5)
style_axis(ax3, xlabel=r'Time $t$', ylabel=r'Drop edge $x_e$',
           title=r'Sensitivity to Mesh Resolution: $x_e(t)$')
ax3.legend(fontsize=plt_settings['LegendFont'], frameon=False)
ax3.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'{plot_dir}/xe_vs_time.pdf', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Combined 3-panel plot
fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
for N, data in all_data.items():
    ax1.plot(data['time'], data['h0'], color=data['color'], label=data['label'], linewidth=2.5)
    ax2.plot(data['time'], data['x0'], color=data['color'], linewidth=2.5)
    ax3.plot(data['time'], data['xe'], color=data['color'], linewidth=2.5)

style_axis(ax1, ylabel=r'$h_0$', title=r'Sensitivity to Mesh Resolution $N$')
ax1.legend(fontsize=plt_settings['LegendFont'], frameon=False, loc='upper left')
style_axis(ax2, ylabel=r'$x_0$')
style_axis(ax3, xlabel=r'Time $t$', ylabel=r'$x_e$')
ax3.set_xlim(left=0)
plt.tight_layout()
plt.savefig(f'{plot_dir}/sensitivity_N_combined.pdf', dpi=300, bbox_inches='tight')
plt.close()

print(f"\nPlots saved to '{plot_dir}/' directory")
PYTHON_SCRIPT

# Run postprocessing for each individual case
echo ""
echo "Running postprocessing for each case..."
for N in "${N_VALUES[@]}"; do
    OUTDIR="sensitivity_N_${N}"
    echo "  Postprocessing $OUTDIR..."
    python postprocess.py "$OUTDIR"
done

echo ""
echo "Sensitivity analysis complete!"
echo "Results saved to:"
echo "  - sensitivity_N_plots/ (comparison plots)"
for N in "${N_VALUES[@]}"; do
    echo "  - sensitivity_N_${N}_plots/ (individual case plots)"
done
