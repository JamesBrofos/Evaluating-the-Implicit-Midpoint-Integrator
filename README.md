# Evaluating the Implicit Midpoint Integrator

Evaluating the implicit midpoint integrator for Riemannian Manifold Hamiltonian Monte Carlo.

## Computational Environment

We use a Conda virtual environment to manage Python dependencies. A list of required libraries is contained in `requirements.txt`. Moreover, the examples will require the `hmc` module to be importable. This can be achieved by running `pip install .` from the directory containing `setup.py`. The Conda environment is expected to have the name `implicit-midpoint-devel`.

Code has been tested with Python 3.8.

## Reproducing Results

See the `examples` directory for a list of modules that implement the experiments from our paper. The experiments consist of various parameter configurations, each of which corresponds to a line in the file called `joblist.txt`. We leverage [dSQ](https://github.com/ycrc/dSQ) in order to facilitate running these experiments on a computing cluster.
