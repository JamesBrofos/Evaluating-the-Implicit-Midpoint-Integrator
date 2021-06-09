# Evaluating the Implicit Midpoint Integrator

Evaluating the implicit midpoint integrator for Riemannian Manifold Hamiltonian Monte Carlo.

## Computational Environment

We use a Conda virtual environment to manage Python dependencies. A list of required libraries is contained in `requirements.txt`. Moreover, the examples will require the `hmc` module to be importable. This can be achieved by running `pip install .` from the directory containing `setup.py`. The Conda environment is expected to have the name `implicit-midpoint-devel`.

Code has been tested with Python 3.8.

## Reproducing Results

See the `examples` directory for a list of modules that implement the experiments from our paper. The experiments consist of various parameter configurations, each of which corresponds to a line in the file called `joblist.txt`. We leverage [dSQ](https://github.com/ycrc/dSQ) in order to facilitate running these experiments on a computing cluster.

## Statement of Revisions

This version of the code has incorporated several improvements relative to previous implementations used by the paper. Specifically:

1. Both the implicit midpoint integrators and the G.L.F.(b) integrator have been improved by eliminating one redundant matrix inversion at every iteration.
2. The stochastic volatility model has been improved to exploit the special tridiagonal structure of the Fisher information metric, leading to a more efficient implementation.
3. An error in the computation of the Riemannian metric used in the stochastic volatility model has been corrected and that experimental results updated accordingly.
4. The implementation of the Fitzhugh-Nagumo model has been improved to avoid repeat solutions to identical or related differential equations. This benefits all integrators.
5. The G.L.F.(b) integrator has been improved to incorporate additional caching of repeated computations.
