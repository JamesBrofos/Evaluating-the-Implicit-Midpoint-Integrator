from setuptools import setup


# Import __version__ from code base.
exec(open("hmc/version.py").read())

setup(
    name="hmc",
    version=__version__,
    description="An evaluation of the implicit midpoint integrator for Riemannian manifold HMC.",
    author="James A. Brofos",
    author_email="james@brofos.org",
    url="http://brofos.org",
    keywords="implicit midpoint generalized leapfrog machine learning statistics hamiltonian monte carlo bayesian inference",
)
