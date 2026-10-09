"""Recover drift and diffusion in a one-dimensional double-well system."""

from pathlib import Path
import sys

# Allow this script to run directly from the repository root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.bisde import bisde_pythonic
from src.polynomial_library import polynomial_library, drift_targets, diffusion_targets
from src.simulation import simulate_1d


DT = 0.04
N_STEPS = 80_000
C1, C2 = 0.64, -0.0256
D = 2.0


def drift(x):
    return C1 * x + C2 * x**3


def diffusion(x):
    return D


def infer_coefficients(x):
    # The original notebook uses ten candidate powers (1, x, ..., x^9).
    phi = polynomial_library(x[np.newaxis, :], order=10)
    drift_coef = bisde_pythonic(phi, drift_targets(x, DT), max_iters=100)
    drift_coef[np.abs(drift_coef) < 0.02] = 0

    diff_y = diffusion_targets(x, DT, phi, drift_coef)
    diff_coef = bisde_pythonic(phi, diff_y, max_iters=100)
    diff_coef[np.abs(diff_coef) < 0.02] = 0
    return drift_coef, diff_coef


def main():
    time, x = simulate_1d(drift, diffusion, x0=0, dt=DT, n_steps=N_STEPS, seed=42)
    drift_coef, diff_coef = infer_coefficients(x)

    print("1D double-well potential")
    print("Drift: theoretical 0.64*x - 0.0256*x**3")
    print(f"Drift: estimated {drift_coef[1]:.6f}*x + {drift_coef[3]:.6f}*x**3")
    print(f"Diffusion: theoretical {D:.6f}, estimated {diff_coef[0]:.6f}")
    print(f"Relative errors: x={abs((drift_coef[1]-C1)/C1)*100:.2f}%, "
          f"x**3={abs((drift_coef[3]-C2)/C2)*100:.2f}%, "
          f"diffusion={abs((diff_coef[0]-D)/D)*100:.2f}%")
    print("Nonzero drift terms:", [(i, round(v, 6)) for i, v in enumerate(drift_coef) if v != 0])
    print("Nonzero diffusion terms:", [(i, round(v, 6)) for i, v in enumerate(diff_coef) if v != 0])

    grid = np.linspace(-6, 6, 300)
    estimated = sum(coef * grid**i for i, coef in enumerate(drift_coef))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(time[:1500], x[:1500])
    ax[0].set(xlabel="Time", ylabel="x", title="Simulated trajectory (first 1500 steps)")
    ax[1].plot(grid, drift(grid), label="Theoretical")
    ax[1].plot(grid, estimated, "--", label="Estimated")
    ax[1].set(xlabel="x", ylabel="Drift", title="Drift comparison")
    ax[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
