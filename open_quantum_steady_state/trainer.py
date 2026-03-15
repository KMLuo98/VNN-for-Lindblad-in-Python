from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .ansatz import DensityMatrixAnsatz
from .observables import exact_observables, vectorized_density_matrix
from .operators import Liouvillian, LocalOperator
from .sampler import MetropolisSampler
from .sr import StochasticReconfiguration


@dataclass
class TrainingHistory:
    costs: List[float] = field(default_factory=list)
    cost_errors: List[float] = field(default_factory=list)
    acceptance_rates: List[float] = field(default_factory=list)
    observables: List[Dict[str, complex]] = field(default_factory=list)


class SteadyStateTrainer:
    def __init__(
        self,
        ansatz: DensityMatrixAnsatz,
        liouvillian: Liouvillian,
        sampler: MetropolisSampler,
        sr: StochasticReconfiguration,
        learning_rate: float,
        observables: Optional[Dict[str, LocalOperator]] = None,
    ) -> None:
        self.ansatz = ansatz
        self.liouvillian = liouvillian
        self.sampler = sampler
        self.sr = sr
        self.learning_rate = learning_rate
        self.observables = observables or {}
        self.n_sites = liouvillian.n_sites

    def _sampled_costs(self, rho_vec: np.ndarray, sample_indices: np.ndarray) -> np.ndarray:
        ldagl_rho = self.liouvillian.ldagl.dot(rho_vec)
        local_values = ldagl_rho[sample_indices] / rho_vec[sample_indices]
        return local_values

    def step(self) -> Dict[str, object]:
        rho_vec, _ = vectorized_density_matrix(self.ansatz, self.n_sites)
        sample = self.sampler.sample(self.ansatz)
        local_values = self._sampled_costs(rho_vec, sample.sample_indices)
        log_derivatives = np.asarray(
            [self.ansatz.derivatives(row, col) for row, col in zip(sample.rows, sample.cols)],
            dtype=np.complex128,
        )
        delta = self.sr.solve(
            log_derivatives=log_derivatives,
            local_values=local_values,
            real_parameters=np.issubdtype(self.ansatz.parameter_dtype, np.floating),
        )
        self.ansatz.apply_update(delta, self.learning_rate)
        cost_mean = float(np.real(np.mean(local_values)))
        cost_error = float(np.std(np.real(local_values)) / np.sqrt(len(local_values)))
        exact_obs = exact_observables(self.ansatz, self.observables, self.n_sites) if self.observables else {}
        return {
            "cost": cost_mean,
            "cost_error": cost_error,
            "acceptance_rate": sample.acceptance_rate,
            "observables": exact_obs,
        }

    def fit(self, iterations: int) -> TrainingHistory:
        history = TrainingHistory()
        for _ in range(iterations):
            step_data = self.step()
            history.costs.append(step_data["cost"])
            history.cost_errors.append(step_data["cost_error"])
            history.acceptance_rates.append(step_data["acceptance_rate"])
            history.observables.append(step_data["observables"])
        return history
