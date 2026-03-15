from setuptools import find_packages, setup


setup(
    name="open-quantum-steady-state",
    version="0.1.0",
    description="Python reimplementation of the variational neural-network ansatz for steady states in open quantum systems.",
    packages=find_packages(include=["open_quantum_steady_state", "open_quantum_steady_state.*"]),
    install_requires=["numpy>=1.17", "scipy>=1.7"],
)
