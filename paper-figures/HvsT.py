#!/usr/bin/env python3
"""
HvsT.py - Plot h(t) for droplet coalescence with surfactants.

Generates two publication-quality log-log plots:
1. Uncompensated: h_min(t) vs t
2. Compensated: h_min(t)/h_a vs t (normalized by theoretical scaling)

Converted from HvsT.m
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# Publication-quality plot settings (LaTeX fonts, thick spines)
# =============================================================================
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman'],
    'font.size': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.linewidth': 1.5,
    'xtick.major.width': 1.2,
    'ytick.major.width': 1.2,
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'legend.frameon': False,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# =============================================================================
# Data loading
# =============================================================================
data_dir = Path(__file__).parent / 'h_f_vals'

# Load all 8 datasets
min_h = []
for k in range(1, 9):
    file_path = data_dir / f'min_h_values_{k}.txt'
    df = pd.read_csv(file_path, delimiter='\t')
    min_h.append(df['Min_h'].values)

# =============================================================================
# Parameters
# =============================================================================
# Time arrays
t1 = np.linspace(0, 1000, 10001)  # For datasets 1-4 (theta = 10 deg)
t2 = np.linspace(0, 250, 2501)    # For datasets 5-8 (theta = 20 deg)

# t_0 correction for t1
t1 = t1 + 0.39

# Physical parameters for each dataset
beta = np.array([0.8, 0.8, 0.5, 0.5, 0.8, 0.8, 0.5, 0.5])
Pe = np.array([1, 10, 1, 10, 1, 10, 1, 10])
theta1 = 10 * np.pi / 180
theta2 = 20 * np.pi / 180
theta = np.array([theta1, theta1, theta1, theta1, theta2, theta2, theta2, theta2])

# Theoretical scaling for compensation
Gamma_0 = 0.8
h_a = (1 - beta * Gamma_0 / 2) * theta**4

# =============================================================================
# Colors (Tableau-10)
# =============================================================================
colors = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
]

# =============================================================================
# Plot configuration: which datasets to plot and their ranges
# =============================================================================
# Format: (dataset_index, time_array, end_index, label)
plot_config = [
    (0, t1, 10001, r'$\beta=0.8$, $\mathrm{Pe}=1$, $\theta=10^\circ$'),
    (1, t1, 10001, r'$\beta=0.8$, $\mathrm{Pe}=10$, $\theta=10^\circ$'),
    (2, t1, 10001, r'$\beta=0.5$, $\mathrm{Pe}=1$, $\theta=10^\circ$'),
    (3, t1, 10001, r'$\beta=0.5$, $\mathrm{Pe}=10$, $\theta=10^\circ$'),
    (4, t2, 2501, r'$\beta=0.8$, $\mathrm{Pe}=1$, $\theta=20^\circ$'),
    (5, t2, 2501, r'$\beta=0.8$, $\mathrm{Pe}=10$, $\theta=20^\circ$'),
    (6, t2, 2501, r'$\beta=0.5$, $\mathrm{Pe}=1$, $\theta=20^\circ$'),
    (7, t2, 2501, r'$\beta=0.5$, $\mathrm{Pe}=10$, $\theta=20^\circ$'),
]

# Marker settings
marker_size = 8
marker_edge_width = 0.8

# =============================================================================
# Figure 1: Uncompensated h_min(t)
# =============================================================================
fig1, ax1 = plt.subplots(figsize=(5, 4))

for idx, t_arr, end_idx, label in plot_config:
    ax1.loglog(
        t_arr[1:end_idx], min_h[idx][1:end_idx],
        'o', markersize=marker_size,
        markerfacecolor=colors[idx], markeredgecolor='none',
        alpha=0.7, label=label
    )

# Reference power law: h ~ t
t_ref = t1[9:110]
ax1.loglog(t_ref, 0.000872 * t_ref**1, 'k-', linewidth=2.5, label=r'$\sim t$')

ax1.set_xlabel(r'$t$')
ax1.set_ylabel(r'$h_{\mathrm{min}}$')
ax1.legend(loc='lower right', fontsize=8)

fig1.tight_layout()
fig1.savefig(Path(__file__).parent / 'HvsT_uncompensated.pdf')
print("Saved: HvsT_uncompensated.pdf")

# =============================================================================
# Figure 2: Compensated h_min(t)/h_a
# =============================================================================
fig2, ax2 = plt.subplots(figsize=(5, 4))

for idx, t_arr, end_idx, label in plot_config:
    ax2.loglog(
        t_arr[1:end_idx], min_h[idx][1:end_idx] / h_a[idx],
        'o', markersize=marker_size,
        markerfacecolor=colors[idx], markeredgecolor='none',
        alpha=0.7, label=label
    )

# Reference power law: h/h_a ~ t
t_ref2 = t1[1:700]
ax2.loglog(t_ref2, 0.272 * t_ref2**1, 'r-', linewidth=2.5, label=r'$\sim t$')

ax2.set_xlabel(r'$t$')
ax2.set_ylabel(r'$h_{\mathrm{min}} / h_a$')
ax2.legend(loc='lower right', fontsize=8)

fig2.tight_layout()
fig2.savefig(Path(__file__).parent / 'HvsT_compensated.pdf')
print("Saved: HvsT_compensated.pdf")

plt.close('all')
