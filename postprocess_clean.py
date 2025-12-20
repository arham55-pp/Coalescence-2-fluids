import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from matplotlib.gridspec import GridSpec

# Set up matplotlib parameters for nice plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 1.5

# Get base folder from command line, default to "coalescence_clean"
base_folder = sys.argv[1] if len(sys.argv) > 1 else "coalescence_clean"
output_dir = f"{base_folder}/domain"
plot_dir = f"{base_folder}_plots"

# Detect if this is a spreading (axisymmetric) or coalescence case
is_spreading = "spreading" in base_folder.lower()

files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')])

# Create folder for plots
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

print(f"Processing {len(files)} files from {output_dir}...")
print(f"Mode: {'Spreading (axisymmetric)' if is_spreading else 'Coalescence'}")

# Initialize lists for time series data
time_data = []
h_min_data = []
h_max_data = []
h_center_data = []
p_max_data = []

# Plot 1: Height profile evolution at selected times
print("Creating height evolution plot...")
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Select files to plot (every 40th file, or fewer if not enough files)
step = max(1, len(files) // 10)
selected_indices = range(0, len(files), step)
colors = plt.cm.viridis(np.linspace(0, 1, len(list(selected_indices))))

for idx, file_idx in enumerate(range(0, len(files), step)):
    with open(os.path.join(output_dir, files[file_idx])) as f:
        header = f.readline()
        time = float(header.split('@time=')[-1])
        data = np.loadtxt(f)
        x = data[:, 0]
        h = data[:, 1]
        p = data[:, 2]

        # Sort by x for clean plotting
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        h = h[sort_idx]
        p = p[sort_idx]

        ax1.plot(x, h, color=colors[idx], label=f't={time:.2f}')
        ax2.plot(x, p, color=colors[idx])

ax1.set_ylabel('Height $h$')
title = 'Droplet Spreading Evolution' if is_spreading else 'Droplet Coalescence Evolution (Clean)'
ax1.set_title(title)
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Position $x$' if not is_spreading else 'Radial position $r$')
ax2.set_ylabel('Pressure $p$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/height_pressure_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Spacetime diagram of height
print("Creating spacetime diagram...")
times = []
height_matrix = []

# First pass: determine common x grid from first file
with open(os.path.join(output_dir, files[0])) as file:
    file.readline()
    data = np.loadtxt(file)
    x_data = data[:, 0]
    sort_idx = np.argsort(x_data)
    x_sorted = x_data[sort_idx]
    x_common = np.linspace(x_sorted.min(), x_sorted.max(), 500)

for i, f in enumerate(files[::2]):  # Use every other file for speed
    with open(os.path.join(output_dir, f)) as file:
        header = file.readline()
        time = float(header.split('@time=')[-1])
        times.append(time)
        data = np.loadtxt(file)
        x_data = data[:, 0]
        h_data = data[:, 1]
        # Sort and interpolate onto common grid
        sort_idx = np.argsort(x_data)
        h_interp = np.interp(x_common, x_data[sort_idx], h_data[sort_idx])
        height_matrix.append(h_interp)

height_matrix = np.array(height_matrix)

fig2, ax = plt.subplots(figsize=(10, 6))
im = ax.pcolormesh(x_common, times, height_matrix, shading='auto', cmap='viridis')
ax.set_xlabel('Position $x$' if not is_spreading else 'Radial position $r$')
ax.set_ylabel('Time $t$')
ax.set_title('Spacetime Evolution of Film Height')
cbar = plt.colorbar(im, ax=ax, label='Height $h$')
plt.savefig(f'{plot_dir}/spacetime_height.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Time series analysis
print("Analyzing time series data...")
for f in files:
    with open(os.path.join(output_dir, f)) as file:
        header = file.readline()
        time = float(header.split('@time=')[-1])
        data = np.loadtxt(file)
        x = data[:, 0]
        h = data[:, 1]
        p = data[:, 2]

        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        h = h[sort_idx]
        p = p[sort_idx]

        time_data.append(time)
        h_min_data.append(np.min(h))
        h_max_data.append(np.max(h))
        # For coalescence, center is at x=0; for spreading, center is at r=0
        if is_spreading:
            h_center_data.append(h[0])  # r=0 is first point
        else:
            center_idx = np.argmin(np.abs(x))  # closest to x=0
            h_center_data.append(h[center_idx])
        p_max_data.append(np.max(np.abs(p)))

fig3 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig3)

ax1 = fig3.add_subplot(gs[0, :])
if is_spreading:
    ax1.plot(time_data, h_max_data, 'b-', label='Maximum height (center)')
    ax1.set_ylabel('Maximum height')
    ax1.set_title('Evolution of Droplet Height During Spreading')
else:
    ax1.plot(time_data, h_min_data, 'b-', label='Minimum height (bridge)')
    ax1.set_ylabel('Minimum height')
    ax1.set_title('Evolution of Bridge Height During Coalescence')
ax1.set_xlabel('Time $t$')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = fig3.add_subplot(gs[1, 0])
ax2.plot(time_data, h_max_data, 'r-', label='Maximum height')
ax2.plot(time_data, h_center_data, 'g--', label='Center height')
if not is_spreading:
    ax2.plot(time_data, h_min_data, 'b:', label='Minimum height')
ax2.set_xlabel('Time $t$')
ax2.set_ylabel('Height')
ax2.set_title('Height Evolution at Different Positions')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3 = fig3.add_subplot(gs[1, 1])
ax3.plot(time_data, p_max_data, 'm-')
ax3.set_xlabel('Time $t$')
ax3.set_ylabel('Max $|p|$')
ax3.set_title('Maximum Pressure')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Zoom on region of interest
print("Creating zoom plot...")
if is_spreading:
    # For spreading: zoom on contact line region
    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
else:
    # For coalescence: zoom on bridge region
    fig4, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot at different stages
stages = [0, len(files)//4, len(files)//2, 3*len(files)//4, len(files)-1]
colors = ['blue', 'green', 'orange', 'red', 'purple']
labels = ['Initial', 'Early', 'Middle', 'Late', 'Final']

for idx, (stage, color, label) in enumerate(zip(stages, colors, labels)):
    with open(os.path.join(output_dir, files[stage])) as f:
        header = f.readline()
        time = float(header.split('@time=')[-1])
        data = np.loadtxt(f)
        x = data[:, 0]
        h = data[:, 1]
        p = data[:, 2]

        # Sort by x
        sort_idx = np.argsort(x)
        x = x[sort_idx]
        h = h[sort_idx]
        p = p[sort_idx]

        if is_spreading:
            # For spreading: show full profile, focus on contact line
            mask = x < 2.0  # zoom on inner region
        else:
            # For coalescence: zoom on bridge at x=0
            mask = np.abs(x) < 0.5

        ax1.plot(x[mask], h[mask], color=color, label=f'{label} (t={time:.2f})')
        ax2.plot(x[mask], p[mask], color=color)

ax1.set_ylabel('Height $h$')
if is_spreading:
    ax1.set_title('Contact Line Region Evolution')
else:
    ax1.set_title('Bridge Region Evolution During Coalescence')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Position $x$' if not is_spreading else 'Radial position $r$')
ax2.set_ylabel('Pressure $p$')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{plot_dir}/zoom_region.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Phase portrait (height dynamics)
print("Creating phase portrait...")
fig5, ax = plt.subplots(figsize=(8, 8))

# Use color gradient for time
colors = plt.cm.plasma(np.linspace(0, 1, len(time_data)))

if is_spreading:
    # For spreading: max height vs max pressure
    ax.scatter(h_max_data, p_max_data, c=colors, alpha=0.6, s=20)
    ax.plot(h_max_data, p_max_data, 'k-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Maximum height')
    ax.set_ylabel('Maximum $|p|$')
    ax.set_title('Phase Portrait: Height vs Pressure')
else:
    # For coalescence: min height vs max pressure
    ax.scatter(h_min_data, p_max_data, c=colors, alpha=0.6, s=20)
    ax.plot(h_min_data, p_max_data, 'k-', alpha=0.3, linewidth=0.5)
    ax.set_xlabel('Minimum height (bridge)')
    ax.set_ylabel('Maximum $|p|$')
    ax.set_title('Phase Portrait: Bridge Height vs Pressure')

ax.grid(True, alpha=0.3)

# Add colorbar for time
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=time_data[0], vmax=time_data[-1]))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Time')

plt.savefig(f'{plot_dir}/phase_portrait.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Contact line / bridge position over time (for spreading)
if is_spreading:
    print("Tracking contact line position...")
    contact_positions = []
    for f in files:
        with open(os.path.join(output_dir, f)) as file:
            file.readline()
            data = np.loadtxt(file)
            x = data[:, 0]
            h = data[:, 1]
            sort_idx = np.argsort(x)
            x = x[sort_idx]
            h = h[sort_idx]
            # Find where height drops below threshold (contact line)
            threshold = 0.01  # slightly above precursor
            contact_idx = np.where(h > threshold)[0]
            if len(contact_idx) > 0:
                contact_positions.append(x[contact_idx[-1]])
            else:
                contact_positions.append(x[-1])

    fig6, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time_data, contact_positions, 'b-', linewidth=2)
    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Contact line position $r_c$')
    ax.set_title('Contact Line Evolution During Spreading')
    ax.grid(True, alpha=0.3)
    plt.savefig(f'{plot_dir}/contact_line_position.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"\nAll plots have been saved to the '{plot_dir}' folder!")
print(f"Total number of timesteps analyzed: {len(files)}")
print(f"Time range: {time_data[0]:.2f} to {time_data[-1]:.2f}")
if is_spreading:
    print(f"Maximum height: {max(h_max_data):.6f}")
else:
    print(f"Minimum bridge height reached: {min(h_min_data):.6f}")
