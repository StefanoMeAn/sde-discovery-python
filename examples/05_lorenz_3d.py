"""Recover the drift and diagonal diffusion of a noisy Lorenz system."""
from pathlib import Path
import sys
from itertools import product

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.bisde import bisde_pythonic
from src.polynomial_library import polynomial_library, drift_targets, diffusion_targets
from src.simulation import simulate_3d

DT = 0.001
N_STEPS = 30_000
ORDER = 3  # powers 0, 1, 2 per coordinate: 27 candidate terms
SEED = 42


def drift(position):
    x, y, z = position
    return np.array([10 * (y - x), x * (28 - z) - y, x * y - (8 / 3) * z])


def diffusion(position):
    return np.array([2.0, 2.0, 2.0])


def show_terms(coefficients, label):
    powers = product(range(ORDER), repeat=3)
    terms = [(f'x^{px} y^{py} z^{pz}', round(float(c), 5))
             for (px, py, pz), c in zip(powers, coefficients) if c != 0]
    print(f'{label}: {terms}')


def main():
    time, trajectory = simulate_3d(drift, diffusion, x0=[1.0, 1.0, 1.0],
                                   dt=DT, n_steps=N_STEPS, seed=SEED)
    phi = polynomial_library(trajectory, ORDER)

    print('3D stochastic Lorenz system')
    print('True drift: dx = -10*x + 10*y; dy = 28*x - y - x*z; dz = x*y - (8/3)*z')
    print('True diagonal diffusion: (2, 2, 2)')

    for i, name in enumerate(('x', 'y', 'z')):
        drift_coef = bisde_pythonic(phi, drift_targets(trajectory[i], DT), max_iters=100)
        drift_coef[np.abs(drift_coef) < 0.05] = 0
        diffusion_y = diffusion_targets(trajectory[i], DT, phi, drift_coef)
        diffusion_coef = bisde_pythonic(phi, diffusion_y, max_iters=100)
        diffusion_coef[np.abs(diffusion_coef) < 0.05] = 0
        show_terms(drift_coef, f'Drift {name}')
        show_terms(diffusion_coef, f'Diffusion {name}')

    fig = plt.figure(figsize=(11, 4))
    ax = fig.add_subplot(121, projection='3d')
    ax.plot(*trajectory[:, ::5], lw=0.4)
    ax.set(xlabel='x', ylabel='y', zlabel='z', title='Lorenz trajectory')
    ax2 = fig.add_subplot(122)
    for i, name in enumerate(('x', 'y', 'z')):
        ax2.plot(time[:2000], trajectory[i, :2000], label=name, lw=0.8)
    ax2.set(xlabel='Time', ylabel='Position', title='First 2000 steps')
    ax2.legend()
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
