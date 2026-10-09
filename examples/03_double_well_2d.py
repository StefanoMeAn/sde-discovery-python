"""Identify drift and diffusion in a two-dimensional double-well system."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from src.bisde import bisde_pythonic
from src.polynomial_library import polynomial_library, drift_targets, diffusion_targets
from src.simulation import simulate_2d

DT = 0.005
N_STEPS = 100_000
ORDER = 4  # Powers 0, 1, 2, 3 in each coordinate


def drift(position):
    x, y = position
    return np.array([4*x - 4*x**3, -2*y])


def diffusion(position):
    return np.array([2.0, 2.0])


def show_terms(coefficients, label):
    powers = list(product(range(ORDER), repeat=2))
    terms = [(f"x^{px} y^{py}", round(float(c), 5))
             for (px, py), c in zip(powers, coefficients) if c != 0]
    print(f"{label}: {terms}")


def main():
    time, trajectory = simulate_2d(drift, diffusion, x0=[0.5, 0.0],
                                   dt=DT, n_steps=N_STEPS, seed=42)
    phi = polynomial_library(trajectory, ORDER)
    estimates = []
    for i in range(2):
        drift_coef = bisde_pythonic(phi, drift_targets(trajectory[i], DT), max_iters=100)
        drift_coef[np.abs(drift_coef) < 0.05] = 0
        diffusion_y = diffusion_targets(trajectory[i], DT, phi, drift_coef)
        diffusion_coef = bisde_pythonic(phi, diffusion_y, max_iters=100)
        diffusion_coef[np.abs(diffusion_coef) < 0.05] = 0
        estimates.append((drift_coef, diffusion_coef))

    # With product ordering: (0,0)->0, (0,1)->1, (1,0)->4, (3,0)->12.
    print("2D double-well potential")
    print("True drift: dx = 4*x - 4*x**3; dy = -2*y")
    print("True diagonal diffusion: (2, 2)")
    for name, (a, d) in zip(('x', 'y'), estimates):
        show_terms(a, f"Drift {name}")
        show_terms(d, f"Diffusion {name}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(trajectory[0, :4000], trajectory[1, :4000], lw=0.7)
    axes[0].set(xlabel='x', ylabel='y', title='Trajectory (first 4000 steps)')
    grid = np.linspace(-1.7, 1.7, 300)
    powers = list(product(range(ORDER), repeat=2))
    for j, (name, values) in enumerate((('x', (grid, np.zeros_like(grid))),
                                        ('y', (np.zeros_like(grid), grid)))):
        x, y = values
        predicted = sum(c * x**px * y**py for (px, py), c in zip(powers, estimates[j][0]))
        actual = np.array([drift(np.array([xx, yy]))[j] for xx, yy in zip(x, y)])
        axes[1].plot(grid, actual, label=f'{name} true')
        axes[1].plot(grid, predicted, '--', label=f'{name} estimated')
    axes[1].set(xlabel='Coordinate (other = 0)', ylabel='Drift', title='Drift along coordinate axes')
    axes[1].legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
