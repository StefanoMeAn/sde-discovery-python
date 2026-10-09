"""Polynomial candidate terms for drift and diffusion estimation."""
from itertools import product
import numpy as np


def polynomial_library(trajectories, order):
    """Build terms using exponents 0, ..., order-1 in each dimension.

    Trajectories have shape (dimensions, time_points). The final time
    point is excluded to align with forward increments.
    """
    positions = np.asarray(trajectories, dtype=float)
    if positions.ndim != 2 or order < 1:
        raise ValueError("Expected 2D trajectories and a positive order")
    terms = []
    for powers in product(range(order), repeat=positions.shape[0]):
        term = np.ones(positions.shape[1] - 1)
        for values, power in zip(positions, powers):
            term *= values[:-1] ** power
        terms.append(term)
    return np.column_stack(terms)


def drift_targets(trajectory, dt):
    """Finite-difference estimate of the drift."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    return np.diff(trajectory) / dt


def diffusion_targets(trajectory, dt, phi, drift_coefficients):
    """Squared residual increments divided by dt."""
    if dt <= 0:
        raise ValueError("dt must be positive")
    increments = np.diff(trajectory)
    return (increments - (phi @ drift_coefficients) * dt)**2 / dt
