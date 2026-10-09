# Stochastic Differential Equation Discovery with BISDE

Recovering the drift and diffusion of stochastic dynamical systems from simulated trajectories using sparse Bayesian inference.

## Overview

This project explores **Bayesian Identification of Stochastic Differential Equations (BISDE)**, a method presented by Wang et al. (2022) for identifying stochastic dynamics from data. Starting from an earlier Python implementation and notebook, I reorganized the code into reusable modules and tested it on seven simulated systems, from one-dimensional potentials to three-dimensional chaotic attractors.

The goal is to estimate the governing equations from noisy trajectories and check both the recovered coefficients and the polynomial terms selected by the method. This is an implementation and numerical study of an existing method, not a new inference algorithm.

## Simulation Data

All trajectories are generated in Python using the **Euler–Maruyama** method. There is no external dataset. Each example specifies its own drift, diagonal diffusion, initial condition, time step, number of samples, and random seed.

The seven systems are:

| Dimension | System | Main dynamics |
|---|---|---|
| 1D | Double-well potential | Cubic drift |
| 1D | Harmonic oscillator | Linear restoring drift |
| 2D | Double-well potential | Nonlinear and linear drift components |
| 2D | Harmonic oscillator | Coupled linear drift |
| 3D | Lorenz | Nonlinear chaotic dynamics |
| 3D | Four-Wing | Nonlinear chaotic dynamics |
| 3D | Halvorsen | Nonlinear chaotic dynamics |

The examples use fixed seeds for reproducibility. Their simulation parameters are defined in the corresponding files under `examples/`.

## Methodology

For a stochastic differential equation of the form

$$
 dX_t = a(X_t)\,dt + \sqrt{D(X_t)}\,dW_t,
$$

$a(X_t)$ is the **drift**, $D(X_t)$ is the **diffusion**, and $W_t$ is a Wiener process. In the multidimensional examples, the code estimates each drift component and each **diagonal diffusion** component separately.

The workflow is:

1. **Simulate trajectories.** Generate observations using Euler–Maruyama and known drift and diffusion functions.
2. **Build a polynomial library.** Evaluate candidate terms, such as $1$, $x$, $x^2$, $xy$, and $xz$, at the observed states. The library size depends on the example.
3. **Estimate the drift.** Fit the finite-difference targets $\Delta X/\Delta t$ with sparse Bayesian regression.
4. **Estimate the diffusion.** Use the squared increments after subtracting the estimated drift contribution, divided by $\Delta t$, as diffusion targets.
5. **Compare with the known equations.** Inspect selected terms, coefficient errors, and trajectory plots.

The sparse Bayesian regression code is in `src/bisde.py`. The candidate-library and simulation functions are kept in separate modules. The `original/` directory preserves the earlier notebook and `msbl.py` implementation for context.

## Results

In the seven fixed-seed experiments, the method identified the expected drift terms and estimated the diagonal diffusion coefficients close to their simulated values. The quality of sparse term selection varied between systems.

| System | Drift recovery | Additional selected drift terms |
|---|---|---|
| 1D Double-Well | Both theoretical terms recovered | None |
| 1D Harmonic Oscillator | Linear term recovered | One |
| 2D Double-Well | All theoretical terms recovered | None |
| 2D Harmonic Oscillator | All theoretical terms recovered | Several |
| 3D Lorenz | All theoretical terms recovered | One |
| 3D Four-Wing | All theoretical terms recovered | Several |
| 3D Halvorsen | All theoretical terms recovered | None |

The **Halvorsen** experiment gave particularly clean results, with the expected drift terms selected and coefficient errors below 0.5% in the recorded run. In contrast, **Four-Wing** recovered the expected structure but retained extra terms; its $0.1x$ coefficient in the second drift equation had a relative error of about 27%.

These examples show why coefficient accuracy and **equation-structure recovery** should be considered separately. Detailed coefficient tables, errors, and plots are available in [`notebooks/01_results_and_analysis.ipynb`](notebooks/01_results_and_analysis.ipynb).

## Project Structure

```text
sde-discovery-python/
├── examples/                         # Seven standalone experiments
│   ├── 01_double_well_1d.py
│   ├── 02_harmonic_oscillator_1d.py
│   ├── 03_double_well_2d.py
│   ├── 04_harmonic_oscillator_2d.py
│   ├── 05_lorenz_3d.py
│   ├── 06_four_wing_3d.py
│   └── 07_halvorsen_3d.py
├── notebooks/
│   └── 01_results_and_analysis.ipynb
├── original/                         # Earlier notebook and inference code
│   ├── method3_BISDE.ipynb
│   └── msbl.py
├── src/
│   ├── __init__.py
│   ├── bisde.py
│   ├── polynomial_library.py
│   └── simulation.py
├── .gitignore
├── environment.yml
├── requirements.txt
└── README.md
```

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/StefanoMeAn/sde-discovery-python.git
cd sde-discovery-python
conda env create -f environment.yml
conda activate sde-discovery
```

The environment uses Python 3.11. Dependencies are listed in `requirements.txt`.

## Usage

Run any experiment from the repository root. For example:

```bash
python examples/01_double_well_1d.py
python examples/05_lorenz_3d.py
python examples/07_halvorsen_3d.py
```

Each script simulates a trajectory, estimates drift and diffusion coefficients, prints the selected polynomial terms, and displays plots.

To explore all seven systems together, start JupyterLab:

```bash
jupyter lab
```

Then open `notebooks/01_results_and_analysis.ipynb` and run its cells. Some experiments may take time depending on the machine and simulation settings.

## Limitations

- **Single-run results:** The reported coefficients come from fixed-seed simulations, not repeated-trial statistics.
- **Extra terms:** Sparse regression sometimes selects polynomial terms that are absent from the true equations.
- **Candidate library:** Recovery depends on whether the chosen polynomial library can represent the underlying dynamics.
- **Diffusion scope:** The multidimensional examples estimate diagonal diffusion terms; correlated noise and full diffusion matrices are not evaluated here.
- **Numerical sensitivity:** Finite time steps, trajectory length, and regression settings can affect the estimates.

## References

**Original BISDE method**

Wang, Y., Fang, H., Jin, J., et al. (2022). *Data-Driven Discovery of Stochastic Differential Equations*. **Engineering, 17**, 244–252. https://doi.org/10.1016/j.eng.2022.02.007

The BISDE methodology is credited to this publication. This repository contains a Python implementation and numerical experiments built around that approach; it does not claim authorship of the original method.
