"""Sparse Bayesian coefficient estimation for SDE discovery.

Adapted from the user's MATLAB-to-Python translation of the BISDE method.
This module handles one output vector at a time, as in the original experiments.
"""
import numpy as np


def bisde_pythonic(phi, y, max_iters=200, convergence_check=False):
    """Estimate sparse coefficients from a polynomial library.

    phi: array of shape (n_samples, n_terms)
    y: array of shape (n_samples,) or (n_samples, 1)
    Returns coefficients in the original library order. When requested,
    also returns their values after each iteration.
    """
    phi = np.asarray(phi, dtype=float).copy()
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    if phi.ndim != 2 or phi.shape[0] != y.shape[0]:
        raise ValueError("phi and y must have the same number of samples")
    if phi.shape[1] == 0 or max_iters < 1:
        raise ValueError("Provide library terms and at least one iteration")
    if not np.isfinite(phi).all() or not np.isfinite(y).all():
        raise ValueError("phi and y must contain finite values")

    n, n_terms = phi.shape
    gamma = np.full(n_terms, 0.5)
    keep = np.arange(n_terms)
    noise_var = 1.0
    history = np.zeros((max_iters, n_terms)) if convergence_check else None

    for iteration in range(max_iters):
        active = gamma > 1e-4
        if not active.any():
            break
        if not active.all():
            gamma = gamma[active]
            phi = phi[:, active]
            keep = keep[active]

        g = np.sqrt(gamma)
        u, singular, vt = np.linalg.svd(phi * g, full_matrices=False)
        weights = singular / (singular**2 + noise_var + 1e-16)
        xi = (g[:, None] * vt.T * weights) @ u.T
        mu = (xi @ y).ravel()

        # Posterior diagonal and hyperparameter update.
        sigma_diag = gamma - np.sum(xi.T * (phi * gamma), axis=0)
        gamma_old = gamma.copy()
        gamma = sigma_diag + mu**2

        denominator = n - len(gamma) + np.sum(sigma_diag / gamma_old)
        if denominator <= 0 or not np.isfinite(denominator):
            raise FloatingPointError("Invalid noise variance denominator")
        residual = y - phi @ mu[:, None]
        noise_var = float(np.sum(residual**2) / denominator)
        if not np.isfinite(noise_var) or noise_var < 0:
            raise FloatingPointError("Invalid noise variance estimate")

        if convergence_check:
            history[iteration, keep] = mu

    result = np.zeros(n_terms)
    if 'mu' in locals():
        result[keep] = mu
    return (result, history) if convergence_check else result
