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


def load_simulation_data(outdir):
    """Load h0(t), x0(t), xf from simulation output directory."""
    files = sorted(glob.glob(f'{outdir}/domain/domain_*.txt'))
    if not files:
        return None

    time_data = []
    h0_data = []
    x0_data = []
    xf_data = []

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

        # h0: bridge height at x=0
        h0 = np.interp(0.0, x, h)

        # x0: position of maximum height (bridge peak)
        x0 = x[np.argmax(h)]

        # xf: front position on right side (where h drops below threshold)
        threshold = 0.01
        right_mask = x > 0
        h_right = h[right_mask]
        x_right = x[right_mask]
        below = h_right < threshold
        if np.any(below):
            xf = x_right[np.argmax(below)]
        else:
            xf = x_right[-1]

        time_data.append(time)
        h0_data.append(h0)
        x0_data.append(x0)
        xf_data.append(xf)

    return {
        'time': np.array(time_data),
        'h0': np.array(h0_data),
        'x0': np.array(x0_data),
        'xf': np.array(xf_data)
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

    # Max x0 (maximum bridge peak displacement)
    x0_max = np.max(data['x0'])

    # Final xf
    xf_final = data['xf'][-1]

    return {
        'v': v,
        't0': t0,
        'r2': r2,
        'x0_max': x0_max,
        'xf': xf_final,
        't_max': data['time'][-1]
    }


def relative_diff(val, baseline):
    """Compute relative difference in percent."""
    if baseline is None or val is None or baseline == 0:
        return None
    return abs(val - baseline) / abs(baseline) * 100


def check_hp_sensitivity():
    """Check convergence for precursor film thickness sensitivity."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: PRECURSOR FILM THICKNESS (hp)")
    print("=" * 70)

    hp_values = ['1e-5', '1e-4', '1e-3', '1e-2']
    baseline_hp = '1e-4'

    all_metrics = {}

    for hp in hp_values:
        outdir = f'sensitivity_hp_{hp}'
        print(f"\nLoading {outdir}...")
        data = load_simulation_data(outdir)
        if data is None:
            print(f"  No data found")
            continue

        metrics = compute_metrics(data)
        if metrics is None:
            continue

        all_metrics[hp] = metrics
        print(f"  t_max = {metrics['t_max']:.1f}")
        print(f"  v = {metrics['v']:.6f} (R² = {metrics['r2']:.4f})")
        print(f"  max(x0) = {metrics['x0_max']:.6f}")
        print(f"  xf = {metrics['xf']:.6f}")

    # Compute relative differences
    if baseline_hp in all_metrics:
        baseline = all_metrics[baseline_hp]
        print(f"\n{'-' * 70}")
        print(f"Relative differences (baseline: hp = {baseline_hp})")
        print(f"{'-' * 70}")
        print(f"{'hp':<10} {'v diff (%)':<15} {'max(x0) diff (%)':<20} {'xf diff (%)':<15}")
        print(f"{'-' * 70}")

        max_v_diff = 0
        max_x0_diff = 0
        max_xf_diff = 0

        for hp in hp_values:
            if hp not in all_metrics or hp == baseline_hp:
                continue

            m = all_metrics[hp]
            v_diff = relative_diff(m['v'], baseline['v'])
            x0_diff = relative_diff(m['x0_max'], baseline['x0_max'])
            xf_diff = relative_diff(m['xf'], baseline['xf'])

            if v_diff is not None:
                max_v_diff = max(max_v_diff, v_diff)
            if x0_diff is not None:
                max_x0_diff = max(max_x0_diff, x0_diff)
            if xf_diff is not None:
                max_xf_diff = max(max_xf_diff, xf_diff)

            v_str = f"{v_diff:.2f}" if v_diff is not None else "N/A"
            x0_str = f"{x0_diff:.2f}" if x0_diff is not None else "N/A"
            xf_str = f"{xf_diff:.2f}" if xf_diff is not None else "N/A"

            print(f"{hp:<10} {v_str:<15} {x0_str:<20} {xf_str:<15}")

        print(f"\n{'=' * 70}")
        print(f"SUMMARY (hp: {baseline_hp} to 1e-2):")
        print(f"  Max difference in v:       {max_v_diff:.1f}%")
        print(f"  Max difference in max(x0): {max_x0_diff:.1f}%")
        print(f"  Max difference in xf:      {max_xf_diff:.1f}%")
        print(f"{'=' * 70}")

    return all_metrics


def check_Lx_sensitivity():
    """Check convergence for domain size sensitivity."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: DOMAIN SIZE (Lx)")
    print("=" * 70)

    Lx_values = ['6', '8', '10', '12']
    baseline_Lx = '6'

    all_metrics = {}

    for Lx in Lx_values:
        outdir = f'sensitivity_Lx_{Lx}'
        print(f"\nLoading {outdir}...")
        data = load_simulation_data(outdir)
        if data is None:
            print(f"  No data found")
            continue

        metrics = compute_metrics(data)
        if metrics is None:
            continue

        all_metrics[Lx] = metrics
        print(f"  t_max = {metrics['t_max']:.1f}")
        print(f"  v = {metrics['v']:.6f} (R² = {metrics['r2']:.4f})")
        print(f"  max(x0) = {metrics['x0_max']:.6f}")
        print(f"  xf = {metrics['xf']:.6f}")

    # Compute relative differences
    if baseline_Lx in all_metrics:
        baseline = all_metrics[baseline_Lx]
        print(f"\n{'-' * 70}")
        print(f"Relative differences (baseline: Lx = {baseline_Lx})")
        print(f"{'-' * 70}")
        print(f"{'Lx':<10} {'v diff (%)':<15} {'max(x0) diff (%)':<20} {'xf diff (%)':<15}")
        print(f"{'-' * 70}")

        max_v_diff = 0
        max_x0_diff = 0
        max_xf_diff = 0
        largest_Lx = baseline_Lx

        for Lx in Lx_values:
            if Lx not in all_metrics or Lx == baseline_Lx:
                continue

            m = all_metrics[Lx]
            v_diff = relative_diff(m['v'], baseline['v'])
            x0_diff = relative_diff(m['x0_max'], baseline['x0_max'])
            xf_diff = relative_diff(m['xf'], baseline['xf'])

            if v_diff is not None:
                max_v_diff = max(max_v_diff, v_diff)
            if x0_diff is not None:
                max_x0_diff = max(max_x0_diff, x0_diff)
            if xf_diff is not None:
                max_xf_diff = max(max_xf_diff, xf_diff)

            largest_Lx = Lx

            v_str = f"{v_diff:.2f}" if v_diff is not None else "N/A"
            x0_str = f"{x0_diff:.2f}" if x0_diff is not None else "N/A"
            xf_str = f"{xf_diff:.2f}" if xf_diff is not None else "N/A"

            print(f"{Lx:<10} {v_str:<15} {x0_str:<20} {xf_str:<15}")

        print(f"\n{'=' * 70}")
        print(f"SUMMARY (Lx: {baseline_Lx} to {largest_Lx}):")
        print(f"  Max difference in v:       {max_v_diff:.1f}%")
        print(f"  Max difference in max(x0): {max_x0_diff:.1f}%")
        print(f"  Max difference in xf:      {max_xf_diff:.1f}%")
        print(f"{'=' * 70}")

    return all_metrics


def check_N_sensitivity():
    """Check convergence for mesh resolution sensitivity."""
    print("=" * 70)
    print("SENSITIVITY ANALYSIS: MESH RESOLUTION (N)")
    print("=" * 70)

    N_values = ['500', '1000', '2000', '4000']
    baseline_N = '1000'

    all_metrics = {}

    for N in N_values:
        outdir = f'sensitivity_N_{N}'
        print(f"\nLoading {outdir}...")
        data = load_simulation_data(outdir)
        if data is None:
            print(f"  No data found")
            continue

        metrics = compute_metrics(data)
        if metrics is None:
            continue

        all_metrics[N] = metrics
        print(f"  t_max = {metrics['t_max']:.1f}")
        print(f"  v = {metrics['v']:.6f} (R² = {metrics['r2']:.4f})")
        print(f"  max(x0) = {metrics['x0_max']:.6f}")
        print(f"  xf = {metrics['xf']:.6f}")

    # Compute relative differences
    if baseline_N in all_metrics:
        baseline = all_metrics[baseline_N]
        print(f"\n{'-' * 70}")
        print(f"Relative differences (baseline: N = {baseline_N})")
        print(f"{'-' * 70}")
        print(f"{'N':<10} {'v diff (%)':<15} {'max(x0) diff (%)':<20} {'xf diff (%)':<15}")
        print(f"{'-' * 70}")

        max_v_diff = 0
        max_x0_diff = 0
        max_xf_diff = 0

        for N in N_values:
            if N not in all_metrics or N == baseline_N:
                continue

            m = all_metrics[N]
            v_diff = relative_diff(m['v'], baseline['v'])
            x0_diff = relative_diff(m['x0_max'], baseline['x0_max'])
            xf_diff = relative_diff(m['xf'], baseline['xf'])

            if v_diff is not None:
                max_v_diff = max(max_v_diff, v_diff)
            if x0_diff is not None:
                max_x0_diff = max(max_x0_diff, x0_diff)
            if xf_diff is not None:
                max_xf_diff = max(max_xf_diff, xf_diff)

            v_str = f"{v_diff:.2f}" if v_diff is not None else "N/A"
            x0_str = f"{x0_diff:.2f}" if x0_diff is not None else "N/A"
            xf_str = f"{xf_diff:.2f}" if xf_diff is not None else "N/A"

            print(f"{N:<10} {v_str:<15} {x0_str:<20} {xf_str:<15}")

        print(f"\n{'=' * 70}")
        print(f"SUMMARY (N: 500 to 2000, baseline = {baseline_N}):")
        print(f"  Max difference in v:       {max_v_diff:.1f}%")
        print(f"  Max difference in max(x0): {max_x0_diff:.1f}%")
        print(f"  Max difference in xf:      {max_xf_diff:.1f}%")
        print(f"{'=' * 70}")

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
        """
    )
    parser.add_argument('--hp', action='store_true',
                        help='Check precursor film thickness sensitivity')
    parser.add_argument('--Lx', action='store_true',
                        help='Check domain size sensitivity')
    parser.add_argument('--N', action='store_true',
                        help='Check mesh resolution sensitivity')
    parser.add_argument('--all', action='store_true',
                        help='Run all sensitivity checks')

    args = parser.parse_args()

    # Default to --all if no arguments
    if not (args.hp or args.Lx or args.N or args.all):
        parser.print_help()
        sys.exit(1)

    if args.hp or args.all:
        check_hp_sensitivity()
        print("\n")

    if args.Lx or args.all:
        check_Lx_sensitivity()
        print("\n")

    if args.N or args.all:
        check_N_sensitivity()


if __name__ == '__main__':
    main()
