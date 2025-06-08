import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec

# Set up matplotlib parameters for nice plots
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['lines.linewidth'] = 1.5

# Get all files in output directory
output_dir = "lubrication/domain"
files = sorted([f for f in os.listdir(output_dir) if f.endswith('.txt')])

# Create folder for plots
if not os.path.exists("lubrication_plots"):
    os.makedirs("lubrication_plots")

# Initialize lists for time series data
time_data = []
h_min_data = []
h_max_data = []
h_center_data = []
gamma_max_data = []

# Read first and last files to understand the evolution
with open(os.path.join(output_dir, files[0])) as f:
    header = f.readline()
    data_first = np.loadtxt(f)
    
with open(os.path.join(output_dir, files[-1])) as f:
    header = f.readline()
    data_last = np.loadtxt(f)

# Plot 1: Height profile evolution at selected times
fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Select files to plot (every 40th file)
selected_indices = range(0, len(files), 40)
colors = plt.cm.viridis(np.linspace(0, 1, len(selected_indices)))

for idx, file_idx in enumerate(selected_indices):
    with open(os.path.join(output_dir, files[file_idx])) as f:
        header = f.readline()
        time = float(header.split('@time=')[-1])
        data = np.loadtxt(f)
        x = data[:, 0]
        h = data[:, 1]
        gamma = data[:, 3]
        
        ax1.plot(x, h, color=colors[idx], label=f't={time:.1f}')
        ax2.plot(x, gamma, color=colors[idx])

ax1.set_ylabel('Height h')
ax1.set_title('Evolution of Droplet Coalescence with Surfactants')
ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Position x')
ax2.set_ylabel('Surfactant concentration Γ')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lubrication_plots/height_surfactant_evolution.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Spacetime diagram of height
print("Creating spacetime diagram...")
times = []
height_matrix = []

for i, f in enumerate(files[::2]):  # Use every other file for speed
    with open(os.path.join(output_dir, f)) as file:
        header = file.readline()
        time = float(header.split('@time=')[-1])
        times.append(time)
        data = np.loadtxt(file)
        if i == 0:
            x = data[:, 0]
        h = data[:, 1]
        height_matrix.append(h)

height_matrix = np.array(height_matrix)

fig2, ax = plt.subplots(figsize=(10, 6))
im = ax.pcolormesh(x, times, height_matrix, shading='auto', cmap='viridis')
ax.set_xlabel('Position x')
ax.set_ylabel('Time t')
ax.set_title('Spacetime Evolution of Film Height')
cbar = plt.colorbar(im, ax=ax, label='Height h')
plt.savefig('lubrication_plots/spacetime_height.png', dpi=300, bbox_inches='tight')
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
        gamma = data[:, 3]
        
        time_data.append(time)
        h_min_data.append(np.min(h))
        h_max_data.append(np.max(h))
        h_center_data.append(h[len(h)//2])
        gamma_max_data.append(np.max(np.abs(gamma)))

fig3 = plt.figure(figsize=(12, 8))
gs = GridSpec(2, 2, figure=fig3)

ax1 = fig3.add_subplot(gs[0, :])
ax1.plot(time_data, h_min_data, 'b-', label='Minimum height')
ax1.set_xlabel('Time t')
ax1.set_ylabel('Minimum height')
ax1.set_title('Evolution of Bridge Height During Coalescence')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2 = fig3.add_subplot(gs[1, 0])
ax2.plot(time_data, h_max_data, 'r-', label='Maximum height')
ax2.plot(time_data, h_center_data, 'g--', label='Center height')
ax2.set_xlabel('Time t')
ax2.set_ylabel('Height')
ax2.set_title('Height Evolution at Different Positions')
ax2.grid(True, alpha=0.3)
ax2.legend()

ax3 = fig3.add_subplot(gs[1, 1])
ax3.semilogy(time_data, gamma_max_data, 'm-')
ax3.set_xlabel('Time t')
ax3.set_ylabel('Max |Γ|')
ax3.set_title('Maximum Surfactant Concentration')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lubrication_plots/time_series_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Zoom on the bridge region
print("Creating bridge region plot...")
fig4, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# Plot at different stages of coalescence
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
        gamma = data[:, 3]
        
        # Find bridge region (around minimum height)
        h_min_idx = np.argmin(h)
        x_bridge = x[h_min_idx]
        
        # Zoom window
        mask = np.abs(x - x_bridge) < 0.5
        
        ax1.plot(x[mask], h[mask], color=color, label=f'{label} (t={time:.1f})')
        ax2.plot(x[mask], p[mask], color=color)
        ax3.plot(x[mask], gamma[mask], color=color)

ax1.set_ylabel('Height h')
ax1.set_title('Bridge Region Evolution During Coalescence')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_ylabel('Pressure p')
ax2.grid(True, alpha=0.3)

ax3.set_xlabel('Position x')
ax3.set_ylabel('Surfactant Γ')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lubrication_plots/bridge_region_zoom.png', dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Phase portrait
print("Creating phase portrait...")
fig5, ax = plt.subplots(figsize=(8, 8))

# Use color gradient for time
colors = plt.cm.plasma(np.linspace(0, 1, len(time_data)))

ax.scatter(h_min_data, gamma_max_data, c=colors, alpha=0.6, s=20)
ax.plot(h_min_data, gamma_max_data, 'k-', alpha=0.3, linewidth=0.5)

# Add arrows to show direction
arrow_indices = np.linspace(0, len(h_min_data)-2, 10, dtype=int)
for idx in arrow_indices:
    ax.annotate('', xy=(h_min_data[idx+1], gamma_max_data[idx+1]), 
                xytext=(h_min_data[idx], gamma_max_data[idx]),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5))

ax.set_xlabel('Minimum height')
ax.set_ylabel('Maximum |Γ|')
ax.set_title('Phase Portrait: Bridge Height vs Surfactant Concentration')
ax.grid(True, alpha=0.3)

# Add colorbar for time
sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma, norm=plt.Normalize(vmin=time_data[0], vmax=time_data[-1]))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label='Time')

plt.savefig('lubrication_plots/phase_portrait.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots have been saved to the 'lubrication_plots' folder!")
print(f"Total number of timesteps analyzed: {len(files)}")
print(f"Time range: {time_data[0]:.2f} to {time_data[-1]:.2f}")
print(f"Minimum bridge height reached: {min(h_min_data):.6f}")