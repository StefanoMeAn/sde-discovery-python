# Stochastic Differential Equation Discovery with BISDE

I implemented and tested Bayesian Identification of Stochastic Differential Equations (BISDE) in Python to recover drift and diffusion terms from simulated stochastic trajectories. The project includes seven systems, from one-dimensional potentials to three-dimensional chaotic attractors.

In the seven fixed-seed experiments, BISDE recovered **all 36 theoretical drift terms**, although some equations also contained additional selected terms. The 3D Halvorsen system gave the cleanest results, with all 12 drift terms recovered, no extra terms, and relative coefficient errors below **0.5%**.

## Project Overview

This project grew out of an earlier scientific computing notebook in which I explored how stochastic differential equations could be identified from data. I reorganized that implementation into reusable Python modules and used it to study how well sparse Bayesian inference recovers known equations from simulated trajectories.

The inference method is **BISDE**, introduced by Wang et al. (2022). My work here is a Python implementation and numerical evaluation of that method, not a new inference algorithm. I kept the earlier notebook and inference code in `original/` to document the starting point.

## Simulation Data

There is no external dataset. I generate trajectories using **Euler–Maruyama**, with known drift and diagonal diffusion functions. Each example defines its own initial condition, time step, trajectory length, and random seed.

| Dimension | System | Drift characteristics |
|---|---|---|
| 1D | Double-well potential | Cubic drift |
| 1D | Harmonic oscillator | Linear restoring drift |
| 2D | Double-well potential | Nonlinear and linear components |
| 2D | Harmonic oscillator | Coupled linear equations |
| 3D | Lorenz | Nonlinear chaotic dynamics |
| 3D | Four-Wing | Nonlinear chaotic dynamics |
| 3D | Halvorsen | Nonlinear chaotic dynamics |

The simulations use fixed seeds so the examples can be reproduced. Their parameters are available in the scripts under `examples/`.

## Methodology

### Simulation and inference pipeline

```text
Known stochastic system
          ↓
Euler–Maruyama simulation
          ↓
Observed trajectories
          ↓
Polynomial candidate library
          ↓
BISDE drift estimation
          ↓
Diffusion targets from drift residuals
          ↓
BISDE diffusion estimation
          ↓
Comparison with theoretical equations
```

### Stochastic differential equations

The systems are written in the form

$$
dX_t = a(X_t)\,dt + \sqrt{D(X_t)}\,dW_t,
$$

where $a$ is the drift, $D$ is the diffusion, and $W_t$ is a Wiener process. In the multidimensional experiments, I estimate each drift component and each **diagonal diffusion** component separately.

### Polynomial library and sparse Bayesian inference

I build a library of candidate polynomial terms, such as $1$, $x$, $x^2$, $xy$, and $xz$, evaluated along the simulated trajectory. BISDE uses sparse Bayesian regression to select terms and estimate their coefficients.

For the drift, the regression targets are the finite differences $\Delta X/\Delta t$. For diffusion, I use the squared increments after subtracting the estimated drift contribution, divided by $\Delta t$.

The inference implementation is in `src/bisde.py`; library construction and simulation are in `src/polynomial_library.py` and `src/simulation.py`. The number of candidate terms and the coefficient thresholds depend on the experiment.

## Results

### Drift identification

All theoretical drift terms were selected in the seven recorded runs, but some experiments also produced extra polynomial terms.

| System | Theoretical drift terms recovered | Extra drift terms |
|---|---:|---:|
| 1D Double-Well | 2/2 | 0 |
| 1D Harmonic Oscillator | 1/1 | 1 |
| 2D Double-Well | 3/3 | 0 |
| 2D Harmonic Oscillator | 4/4 | 5 |
| 3D Lorenz | 7/7 | 1 |
| 3D Four-Wing | 7/7 | 5 |
| 3D Halvorsen | 12/12 | 0 |

The estimated diagonal diffusion terms were close to their theoretical values in these runs. Detailed coefficient tables, relative errors, and trajectory plots are available in [`notebooks/01_results_and_analysis.ipynb`](notebooks/01_results_and_analysis.ipynb).

### Observations

The **Halvorsen** experiment recovered the correct drift structure without extra terms, and its largest drift-coefficient error was about **0.44%**. The **Four-Wing** experiment recovered all expected terms but retained additional polynomial terms; the estimated coefficient of $x$ in its second drift equation had a relative error of about **27.19%**.

These results helped me distinguish two aspects of equation discovery: estimating coefficients accurately and identifying the correct terms. A model can approximate the main dynamics while still selecting terms that are absent from the theoretical equation.

## Repository Structure

```text
sde-discovery-python/
├── README.md
├── environment.yml
├── requirements.txt
├── examples/
│   ├── 01_double_well_1d.py
│   ├── 02_harmonic_oscillator_1d.py
│   ├── 03_double_well_2d.py
│   ├── 04_harmonic_oscillator_2d.py
│   ├── 05_lorenz_3d.py
│   ├── 06_four_wing_3d.py
│   └── 07_halvorsen_3d.py
├── notebooks/
│   └── 01_results_and_analysis.ipynb
├── original/
│   ├── method3_BISDE.ipynb
│   └── msbl.py
└── src/
    ├── __init__.py
    ├── bisde.py
    ├── polynomial_library.py
    └── simulation.py
```

## Installation

Clone the repository and create the Conda environment:

```bash
git clone https://github.com/StefanoMeAn/sde-discovery-python.git
cd sde-discovery-python
conda env create -f environment.yml
conda activate sde-discovery
```

The environment uses **Python 3.11**. Dependencies are listed in `requirements.txt`.

## Usage

Run individual experiments from the repository root:

```bash
python examples/01_double_well_1d.py
python examples/05_lorenz_3d.py
python examples/07_halvorsen_3d.py
```

Each script generates a trajectory, estimates drift and diffusion coefficients, prints the selected polynomial terms, and displays plots.

To inspect the combined analysis, start JupyterLab:

```bash
jupyter lab
```

Open `notebooks/01_results_and_analysis.ipynb` and run the cells. Execution time varies by system and machine.

## Limitations and Future Work

- Results are based on **one fixed-seed run per system**, not repeated-trial statistics.
- Some experiments retain polynomial terms absent from the theoretical equations.
- Recovery depends on the candidate library, trajectory length, time step, and regression settings.
- The multidimensional examples estimate **diagonal diffusion** only; correlated noise and full diffusion matrices are not evaluated.
- A larger evaluation across random seeds would help quantify how stable the recovered equations are.
- Additional candidate libraries could be explored for systems that are not well represented by polynomials.

## References

Wang, Y., Fang, H., Jin, J., et al. (2022). *Data-Driven Discovery of Stochastic Differential Equations*. **Engineering, 17**, 244–252. [https://doi.org/10.1016/j.eng.2022.02.007](https://doi.org/10.1016/j.eng.2022.02.007).

BISDE is attributed to this publication. The repository contains my Python implementation, simulations, and numerical analysis of the method.

## Technologies

Python · NumPy · SciPy · Matplotlib · pandas · JupyterLab · Conda

## Author

**Stefano Meza** — Physicist with a Master’s degree in Physics of Data from the University of Padova, interested in machine learning, deep learning, computer vision, NLP, and scientific computing.

[LinkedIn](https://www.linkedin.com/in/stefanomean/) · [GitHub](https://github.com/StefanoMeAn)
