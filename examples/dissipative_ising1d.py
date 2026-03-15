from __future__ import annotations

import os
import sys

if __package__ is None or __package__ == "":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from open_quantum_steady_state import (
    MetropolisSampler,
    NeuralDensityMatrix,
    SteadyStateTrainer,
    StochasticReconfiguration,
    dissipative_ising_1d,
)


def main() -> None:
    n_sites = 5
    g = 0.4
    v = 2.0
    gamma = 1.0

    _, _, liouvillian, observables = dissipative_ising_1d(n_sites=n_sites, g=g, v=v, gamma=gamma)
    ansatz = NeuralDensityMatrix(n_visible=n_sites, alpha_hidden=1.0, alpha_ancilla=1.0, activation="softplus", seed=7)
    sampler = MetropolisSampler(n_sites=n_sites, chain_length=256, passes=n_sites, burn_in=128, seed=11)
    sr = StochasticReconfiguration(diagonal_shift=1e-3)
    trainer = SteadyStateTrainer(
        ansatz=ansatz,
        liouvillian=liouvillian,
        sampler=sampler,
        sr=sr,
        learning_rate=0.01,
        observables={name: observables[name] for name in ("Sx", "Sy", "Sz")},
    )

    history = trainer.fit(iterations=500)
    for index, (cost, error, acceptance, obs) in enumerate(
        zip(history.costs, history.cost_errors, history.acceptance_rates, history.observables),
        start=1,
    ):
        print(
            "iter=%03d cost=%.6e +/- %.3e acc=%.3f Sx=%.6f Sy=%.6f Sz=%.6f"
            % (index, cost, error, acceptance, obs["Sx"].real, obs["Sy"].real, obs["Sz"].real)
        )


if __name__ == "__main__":
    main()
