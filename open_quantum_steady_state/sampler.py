from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .ansatz import DensityMatrixAnsatz
from .operators import basis_states, spins_to_index


@dataclass
class SamplingResult:
    sample_indices: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    log_values: np.ndarray
    acceptance_rate: float


class MetropolisSampler:
    def __init__(self, n_sites: int, chain_length: int, passes: int, burn_in: int = 0, seed: Optional[int] = None) -> None:
        if chain_length <= 0:
            raise ValueError("chain_length must be positive")
        if passes <= 0:
            raise ValueError("passes must be positive")
        self.n_sites = n_sites
        self.chain_length = chain_length
        self.passes = passes if passes % 2 == 1 else passes + 1
        self.burn_in = burn_in
        self.rng = np.random.default_rng(seed)
        self.single_basis = basis_states(n_sites)

    def _random_state(self) -> Tuple[np.ndarray, np.ndarray]:
        row = self.single_basis[self.rng.integers(0, len(self.single_basis))].copy()
        col = self.single_basis[self.rng.integers(0, len(self.single_basis))].copy()
        return row, col

    def sample(self, ansatz: DensityMatrixAnsatz) -> SamplingResult:
        row, col = self._random_state()
        log_current = ansatz.log_psi(row, col)
        indices: List[int] = []
        rows: List[np.ndarray] = []
        cols: List[np.ndarray] = []
        logs: List[complex] = []
        accepted = 0
        attempted = 0
        total_steps = self.burn_in + self.chain_length
        for step in range(total_steps):
            for _ in range(self.passes):
                attempted += 1
                prop_row = row.copy()
                prop_col = col.copy()
                do_row = self.rng.integers(0, 2) == 0
                site = self.rng.integers(0, self.n_sites)
                if do_row:
                    prop_row[site] *= -1.0
                else:
                    prop_col[site] *= -1.0
                log_prop = ansatz.log_psi(prop_row, prop_col)
                delta = 2.0 * np.real(log_prop - log_current)
                if np.log(self.rng.random()) < min(0.0, delta):
                    row, col, log_current = prop_row, prop_col, log_prop
                    accepted += 1
            if step >= self.burn_in:
                row_idx = spins_to_index(row)
                col_idx = spins_to_index(col)
                indices.append(row_idx * len(self.single_basis) + col_idx)
                rows.append(row.copy())
                cols.append(col.copy())
                logs.append(log_current)
        acceptance_rate = accepted / attempted if attempted else 0.0
        return SamplingResult(
            sample_indices=np.asarray(indices, dtype=np.int64),
            rows=np.asarray(rows, dtype=np.float64),
            cols=np.asarray(cols, dtype=np.float64),
            log_values=np.asarray(logs, dtype=np.complex128),
            acceptance_rate=acceptance_rate,
        )
