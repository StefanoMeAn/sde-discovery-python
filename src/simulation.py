"""Simple Euler-Maruyama simulations for the 1D examples."""

import numpy as np


def simulate_1d(drift, diffusion, x0, dt, n_steps, seed=42):
    """Simulate dX = drift(X) dt + sqrt(diffusion(X)) dW.

    Here diffusion means the variance rate, not the noise amplitude.
    """
    if dt <= 0 or n_steps < 2:
        raise ValueError("dt must be positive and n_steps must be at least 2")

    rng = np.random.default_rng(seed)
    noise = rng.normal(size=n_steps - 1) * np.sqrt(dt)
    x = np.empty(n_steps)
    x[0] = x0

    for i in range(1, n_steps):
        previous = x[i - 1]
        x[i] = (previous + drift(previous) * dt
                + np.sqrt(diffusion(previous)) * noise[i - 1])

    return np.arange(n_steps) * dt, x


def simulate_2d(drift, diffusion, x0, dt, n_steps, seed=42):
    """Euler-Maruyama simulation with independent noise in x and y.

    drift(position) returns two drift components; diffusion(position)
    returns their variance rates (diagonal diffusion matrix).
    """
    if dt <= 0 or n_steps < 2:
        raise ValueError("dt must be positive and n_steps must be at least 2")
    position = np.asarray(x0, dtype=float)
    if position.shape != (2,):
        raise ValueError("x0 must contain two coordinates")

    rng = np.random.default_rng(seed)
    trajectory = np.empty((2, n_steps))
    trajectory[:, 0] = position
    for i in range(1, n_steps):
        previous = trajectory[:, i - 1]
        rates = np.asarray(diffusion(previous), dtype=float)
        if rates.shape != (2,) or np.any(rates < 0):
            raise ValueError("Diffusion must return two nonnegative variance rates")
        trajectory[:, i] = (previous + np.asarray(drift(previous)) * dt
                            + np.sqrt(rates * dt) * rng.standard_normal(2))
    return np.arange(n_steps) * dt, trajectory


def simulate_3d(drift, diffusion, x0, dt, n_steps, seed=42):
    """Simulate a 3D SDE with independent Euler-Maruyama noise."""
    if dt <= 0 or n_steps < 2:
        raise ValueError('dt must be positive and n_steps must be at least 2')
    initial = np.asarray(x0, dtype=float)
    if initial.shape != (3,):
        raise ValueError('x0 must contain three coordinates')

    rng = np.random.default_rng(seed)
    trajectory = np.empty((3, n_steps))
    trajectory[:, 0] = initial
    for i in range(1, n_steps):
        previous = trajectory[:, i - 1]
        rates = np.asarray(diffusion(previous), dtype=float)
        if rates.shape != (3,) or np.any(rates < 0):
            raise ValueError('Diffusion must return three nonnegative variance rates')
        trajectory[:, i] = (previous + np.asarray(drift(previous)) * dt
                            + np.sqrt(rates * dt) * rng.standard_normal(3))
    return np.arange(n_steps) * dt, trajectory
