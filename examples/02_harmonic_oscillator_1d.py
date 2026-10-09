"""Recover drift and diffusion in a one-dimensional harmonic oscillator."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.bisde import bisde_pythonic
from src.polynomial_library import polynomial_library, drift_targets, diffusion_targets
from src.simulation import simulate_1d


DT = 0.04
N_STEPS = 50_000
K = 3.0
D = 2.0


def drift(x):
    return -K * x


def diffusion(x):
    return D


def main():
    time, x = simulate_1d(drift, diffusion, x0=0, dt=DT, n_steps=N_STEPS, seed=123)
    phi = polynomial_library(x[np.newaxis, :], order=10)

    drift_coef = bisde_pythonic(phi, drift_targets(x, DT), max_iters=100)
    drift_coef[np.abs(drift_coef) < 0.05] = 0
    diff_y = diffusion_targets(x, DT, phi, drift_coef)
    diff_coef = bisde_pythonic(phi, diff_y, max_iters=100)
    diff_coef[np.abs(diff_coef) < 0.05] = 0

    print("1D harmonic oscillator")
    print(f"Drift: theoretical {-K:.6f}*x, estimated {drift_coef[1]:.6f}*x")
    print(f"Diffusion: theoretical {D:.6f}, estimated {diff_coef[0]:.6f}")
    print(f"Relative errors: drift={abs((drift_coef[1]+K)/K)*100:.2f}%, "
          f"diffusion={abs((diff_coef[0]-D)/D)*100:.2f}%")
    print("Nonzero drift terms:", [(i, round(v, 6)) for i, v in enumerate(drift_coef) if v != 0])
    print("Nonzero diffusion terms:", [(i, round(v, 6)) for i, v in enumerate(diff_coef) if v != 0])

    grid = np.linspace(-3, 3, 300)
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
