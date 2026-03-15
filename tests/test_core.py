from __future__ import annotations

import numpy as np

from open_quantum_steady_state import (
    MetropolisSampler,
    NeuralDensityMatrix,
    RBMSplit,
    SteadyStateTrainer,
    StochasticReconfiguration,
    dissipative_ising_1d,
    exact_density_matrix,
)


def test_rbm_split_derivative_size() -> None:
    model = RBMSplit(n_visible=3, alpha=1.0, seed=1)
    row = np.array([1.0, -1.0, 1.0])
    col = np.array([-1.0, 1.0, -1.0])
    deriv = model.derivatives(row, col)
    assert deriv.shape == (model.n_parameters,)


def test_density_matrix_shape() -> None:
    model = NeuralDensityMatrix(n_visible=2, alpha_hidden=1.0, alpha_ancilla=1.0, seed=2)
    rho = exact_density_matrix(model, n_sites=2)
    assert rho.shape == (4, 4)
    assert np.isfinite(np.linalg.norm(rho))


def test_training_step_returns_observables() -> None:
    _, _, liouvillian, observables = dissipative_ising_1d(n_sites=2, g=0.4, v=1.0, gamma=1.0)
    model = NeuralDensityMatrix(n_visible=2, alpha_hidden=1.0, alpha_ancilla=1.0, seed=3)
    sampler = MetropolisSampler(n_sites=2, chain_length=32, passes=2, burn_in=16, seed=4)
    trainer = SteadyStateTrainer(
        ansatz=model,
        liouvillian=liouvillian,
        sampler=sampler,
        sr=StochasticReconfiguration(diagonal_shift=1e-3),
        learning_rate=0.01,
        observables={name: observables[name] for name in ("Sx", "Sy", "Sz")},
    )
    step = trainer.step()
    assert "Sx" in step["observables"]
    assert np.isfinite(step["cost"])
