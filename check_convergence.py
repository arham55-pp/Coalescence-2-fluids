#!/usr/bin/env python3
"""
Convergence analysis for sensitivity studies.

Usage:
    python check_convergence.py --hp    # Check precursor film sensitivity
    python check_convergence.py --Lx    # Check domain size sensitivity
    python check_convergence.py --N     # Check mesh resolution sensitivity
    python check_convergence.py --all   # Run all checks
"""

import argparse
import glob
import re
import numpy as np
from scipy.optimize import curve_fit
import sys


def linear_model(t, v, t0):
    """h0(t) = v * (t - t0)"""
    return v * (t - t0)


def load_simulation_data(outdir, hp=1e-4):
    """
    Load h0(t), x0(t), xe(t) from simulation output directory.

    Variables:
        h0: Neck height — minimum film thickness in bridge region (|x| < 1.5)
        x0: Neck position — x-coordinate where h = h0
        xe: Drop edge — rightmost x where h > threshold (drop footprint)

    Args:
        outdir: Path to simulation output directory
        hp: Precursor film thickness (used for xe threshold)
    """
    files = sorted(glob.glob(f'{outdir}/domain/domain_*.txt'))
    if not files:
        return None

    time_data = []
    h0_data = []
    x0_data = []
    xe_data = []

    for f in files:
        # Read time from header
        with open(f, 'r') as fp:
            header = fp.readline()
            time_match = re.search(r'@time=([0-9.eE+-]+)', header)
            time = float(time_match.group(1)) if time_match else 0.0

        # Load data
        data = np.loadtxt(f, skiprows=1)
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

    return {
        'time': np.array(time_data),
        'h0': np.array(h0_data),
        'x0': np.array(x0_data),
        'xe': np.array(xe_data)
    }


def fit_coalescence_velocity(time, h0, t_min=1.0, t_max=50.0):
    """
    Fit h0(t) = v * (t - t0) to extract coalescence velocity v.

    Returns: (v, t0, r_squared)
    """
    # Select fitting window
    mask = (time >= t_min) & (time <= t_max) & (h0 > 0.01)
    t_fit = time[mask]
    h_fit = h0[mask]

    if len(t_fit) < 5:
        return None, None, None

    try:
        # Initial guess: slope from endpoints, intercept
        v_guess = (h_fit[-1] - h_fit[0]) / (t_fit[-1] - t_fit[0])
        t0_guess = t_fit[0] - h_fit[0] / v_guess if v_guess != 0 else 0

        popt, pcov = curve_fit(linear_model, t_fit, h_fit, p0=[v_guess, t0_guess])
        v, t0 = popt

        # Compute R-squared
        h_pred = linear_model(t_fit, v, t0)
        ss_res = np.sum((h_fit - h_pred) ** 2)
        ss_tot = np.sum((h_fit - np.mean(h_fit)) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        return v, t0, r_squared
    except Exception as e:
        print(f"    Warning: fitting failed: {e}")
        return None, None, None


def compute_metrics(data):
    """Compute convergence metrics from simulation data."""
    if data is None:
        return None

    # Fit for coalescence velocity
    v, t0, r2 = fit_coalescence_velocity(data['time'], data['h0'])

    # Max x0 (maximum neck displacement)
    x0_max = np.max(np.abs(data['x0']))

    # Final xe (drop edge)
    xe_final = data['xe'][-1]

    return {
        'v': v,
        't0': t0,
        'r2': r2,
        'x0_max': x0_max,
        'xe': xe_final,
        't_max': data['time'][-1]
    }


def relative_diff(val, baseline):
    """Compute relative difference in percent."""
    if baseline is None or val is None or baseline == 0:
        return None
    return abs(val - baseline) / abs(baseline) * 100


def check_hp_sensitivity(beta=0.1, Pe=1.0, hp_default=1e-4):
    """Check convergence for precursor film thickness sensitivity."""
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("SENSITIVITY ANALYSIS: PRECURSOR FILM THICKNESS (hp)")
    log(f"Parameters: beta = {beta}, Pe = {Pe}")
    log("=" * 70)

    hp_values = ['1e-5', '1e-4', '1e-3', '1e-2']
    baseline_hp = '1e-4'

    all_metrics = {}

    for hp_str in hp_values:
        outdir = f'sensitivity_hp_{hp_str}'
        hp_val = float(hp_str)  # Use actual hp for this directory
        log(f"\nLoading {outdir}...")
        data = load_simulation_data(outdir, hp=hp_val)
        if data is None:
            log(f"  No data found")
            continue

        metrics = compute_metrics(data)
        if metrics is None:
            continue

        all_metrics[hp_str] = metrics
        log(f"  t_max = {metrics['t_max']:.1f}")
        log(f"  v = {metrics['v']:.6f} (R² = {metrics['r2']:.4f})")
        log(f"  max(|x0|) = {metrics['x0_max']:.6f}")
        log(f"  xe = {metrics['xe']:.6f}")

    # Compute relative differences
    if baseline_hp in all_metrics:
        baseline = all_metrics[baseline_hp]
        log(f"\n{'-' * 70}")
        log(f"Relative differences (baseline: hp = {baseline_hp})")
        log(f"{'-' * 70}")
        log(f"{'hp':<10} {'v diff (%)':<15} {'max(|x0|) diff (%)':<20} {'xe diff (%)':<15}")
        log(f"{'-' * 70}")

        max_v_diff = 0
        max_x0_diff = 0
        max_xe_diff = 0

        for hp_str in hp_values:
            if hp_str not in all_metrics or hp_str == baseline_hp:
                continue

            m = all_metrics[hp_str]
            v_diff = relative_diff(m['v'], baseline['v'])
            x0_diff = relative_diff(m['x0_max'], baseline['x0_max'])
            xe_diff = relative_diff(m['xe'], baseline['xe'])

            if v_diff is not None:
                max_v_diff = max(max_v_diff, v_diff)
            if x0_diff is not None:
                max_x0_diff = max(max_x0_diff, x0_diff)
            if xe_diff is not None:
                max_xe_diff = max(max_xe_diff, xe_diff)

            v_str = f"{v_diff:.2f}" if v_diff is not None else "N/A"
            x0_str = f"{x0_diff:.2f}" if x0_diff is not None else "N/A"
            xe_str = f"{xe_diff:.2f}" if xe_diff is not None else "N/A"

            log(f"{hp_str:<10} {v_str:<15} {x0_str:<20} {xe_str:<15}")

        log(f"\n{'=' * 70}")
        log(f"SUMMARY (hp: {baseline_hp} to 1e-2):")
        log(f"  Max difference in v:         {max_v_diff:.1f}%")
        log(f"  Max difference in max(|x0|): {max_x0_diff:.1f}%")
        log(f"  Max difference in xe:        {max_xe_diff:.1f}%")
        log(f"{'=' * 70}")

    # Save to file
    filename = f'check-convergence-Pe{Pe}_beta{beta}-hp.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nResults saved to {filename}")

    return all_metrics


def check_Lx_sensitivity(beta=0.1, Pe=1.0, hp=1e-4):
    """Check convergence for domain size sensitivity."""
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("SENSITIVITY ANALYSIS: DOMAIN SIZE (Lx)")
    log(f"Parameters: beta = {beta}, Pe = {Pe}, hp = {hp}")
    log("=" * 70)

    Lx_values = ['6', '8', '10', '12']
    baseline_Lx = '6'

    all_metrics = {}

    for Lx in Lx_values:
        outdir = f'sensitivity_Lx_{Lx}'
        log(f"\nLoading {outdir}...")
        data = load_simulation_data(outdir, hp=hp)
        if data is None:
            log(f"  No data found")
            continue

        metrics = compute_metrics(data)
        if metrics is None:
            continue

        all_metrics[Lx] = metrics
        log(f"  t_max = {metrics['t_max']:.1f}")
        log(f"  v = {metrics['v']:.6f} (R² = {metrics['r2']:.4f})")
        log(f"  max(|x0|) = {metrics['x0_max']:.6f}")
        log(f"  xe = {metrics['xe']:.6f}")

    # Compute relative differences
    if baseline_Lx in all_metrics:
        baseline = all_metrics[baseline_Lx]
        log(f"\n{'-' * 70}")
        log(f"Relative differences (baseline: Lx = {baseline_Lx})")
        log(f"{'-' * 70}")
        log(f"{'Lx':<10} {'v diff (%)':<15} {'max(|x0|) diff (%)':<20} {'xe diff (%)':<15}")
        log(f"{'-' * 70}")

        max_v_diff = 0
        max_x0_diff = 0
        max_xe_diff = 0
        largest_Lx = baseline_Lx

        for Lx in Lx_values:
            if Lx not in all_metrics or Lx == baseline_Lx:
                continue

            m = all_metrics[Lx]
            v_diff = relative_diff(m['v'], baseline['v'])
            x0_diff = relative_diff(m['x0_max'], baseline['x0_max'])
            xe_diff = relative_diff(m['xe'], baseline['xe'])

            if v_diff is not None:
                max_v_diff = max(max_v_diff, v_diff)
            if x0_diff is not None:
                max_x0_diff = max(max_x0_diff, x0_diff)
            if xe_diff is not None:
                max_xe_diff = max(max_xe_diff, xe_diff)

            largest_Lx = Lx

            v_str = f"{v_diff:.2f}" if v_diff is not None else "N/A"
            x0_str = f"{x0_diff:.2f}" if x0_diff is not None else "N/A"
            xe_str = f"{xe_diff:.2f}" if xe_diff is not None else "N/A"

            log(f"{Lx:<10} {v_str:<15} {x0_str:<20} {xe_str:<15}")

        log(f"\n{'=' * 70}")
        log(f"SUMMARY (Lx: {baseline_Lx} to {largest_Lx}):")
        log(f"  Max difference in v:         {max_v_diff:.1f}%")
        log(f"  Max difference in max(|x0|): {max_x0_diff:.1f}%")
        log(f"  Max difference in xe:        {max_xe_diff:.1f}%")
        log(f"{'=' * 70}")

    # Save to file
    filename = f'check-convergence-Pe{Pe}_beta{beta}-Lx.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nResults saved to {filename}")

    return all_metrics


def check_N_sensitivity(beta=0.1, Pe=1.0, hp=1e-4):
    """Check convergence for mesh resolution sensitivity."""
    lines = []
    def log(msg=""):
        print(msg)
        lines.append(msg)

    log("=" * 70)
    log("SENSITIVITY ANALYSIS: MESH RESOLUTION (N)")
    log(f"Parameters: beta = {beta}, Pe = {Pe}, hp = {hp}")
    log("=" * 70)

    N_values = ['500', '1000', '2000', '4000']
    baseline_N = '1000'

    all_metrics = {}

    for N in N_values:
        outdir = f'sensitivity_N_{N}'
        log(f"\nLoading {outdir}...")
        data = load_simulation_data(outdir, hp=hp)
        if data is None:
            log(f"  No data found")
            continue

        metrics = compute_metrics(data)
        if metrics is None:
            continue

        all_metrics[N] = metrics
        log(f"  t_max = {metrics['t_max']:.1f}")
        log(f"  v = {metrics['v']:.6f} (R² = {metrics['r2']:.4f})")
        log(f"  max(|x0|) = {metrics['x0_max']:.6f}")
        log(f"  xe = {metrics['xe']:.6f}")

    # Compute relative differences
    if baseline_N in all_metrics:
        baseline = all_metrics[baseline_N]
        log(f"\n{'-' * 70}")
        log(f"Relative differences (baseline: N = {baseline_N})")
        log(f"{'-' * 70}")
        log(f"{'N':<10} {'v diff (%)':<15} {'max(|x0|) diff (%)':<20} {'xe diff (%)':<15}")
        log(f"{'-' * 70}")

        max_v_diff = 0
        max_x0_diff = 0
        max_xe_diff = 0

        for N in N_values:
            if N not in all_metrics or N == baseline_N:
                continue

            m = all_metrics[N]
            v_diff = relative_diff(m['v'], baseline['v'])
            x0_diff = relative_diff(m['x0_max'], baseline['x0_max'])
            xe_diff = relative_diff(m['xe'], baseline['xe'])

            if v_diff is not None:
                max_v_diff = max(max_v_diff, v_diff)
            if x0_diff is not None:
                max_x0_diff = max(max_x0_diff, x0_diff)
            if xe_diff is not None:
                max_xe_diff = max(max_xe_diff, xe_diff)

            v_str = f"{v_diff:.2f}" if v_diff is not None else "N/A"
            x0_str = f"{x0_diff:.2f}" if x0_diff is not None else "N/A"
            xe_str = f"{xe_diff:.2f}" if xe_diff is not None else "N/A"

            log(f"{N:<10} {v_str:<15} {x0_str:<20} {xe_str:<15}")

        log(f"\n{'=' * 70}")
        log(f"SUMMARY (N: 500 to 4000, baseline = {baseline_N}):")
        log(f"  Max difference in v:         {max_v_diff:.1f}%")
        log(f"  Max difference in max(|x0|): {max_x0_diff:.1f}%")
        log(f"  Max difference in xe:        {max_xe_diff:.1f}%")
        log(f"{'=' * 70}")

    # Save to file
    filename = f'check-convergence-Pe{Pe}_beta{beta}-N.txt'
    with open(filename, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nResults saved to {filename}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Convergence analysis for sensitivity studies',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python check_convergence.py --hp    # Check precursor film sensitivity
    python check_convergence.py --Lx    # Check domain size sensitivity
    python check_convergence.py --N     # Check mesh resolution sensitivity
    python check_convergence.py --all   # Run all checks
    python check_convergence.py --Lx --beta 0.2 --Pe 10 --hp-val 1e-3
        """
    )
    # Sensitivity type selection
    parser.add_argument('--hp', action='store_true',
                        help='Check precursor film thickness sensitivity')
    parser.add_argument('--Lx', action='store_true',
                        help='Check domain size sensitivity')
    parser.add_argument('--N', action='store_true',
                        help='Check mesh resolution sensitivity')
    parser.add_argument('--all', action='store_true',
                        help='Run all sensitivity checks')
    # Simulation parameters
    parser.add_argument('--beta', type=float, default=0.1,
                        help='Surfactant strength (default: 0.1)')
    parser.add_argument('--Pe', type=float, default=1.0,
                        help='Péclet number (default: 1.0)')
    parser.add_argument('--hp-val', type=float, default=1e-4,
                        help='Precursor film thickness for xe threshold (default: 1e-4)')

    args = parser.parse_args()

    # Default to --all if no sensitivity type specified
    if not (args.hp or args.Lx or args.N or args.all):
        parser.print_help()
        sys.exit(1)

    if args.hp or args.all:
        check_hp_sensitivity(beta=args.beta, Pe=args.Pe)
        print("\n")

    if args.Lx or args.all:
        check_Lx_sensitivity(beta=args.beta, Pe=args.Pe, hp=args.hp_val)
        print("\n")

    if args.N or args.all:
        check_N_sensitivity(beta=args.beta, Pe=args.Pe, hp=args.hp_val)


if __name__ == '__main__':
    main()
