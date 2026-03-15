from __future__ import annotations

from typing import Dict

import numpy as np

from .ansatz import DensityMatrixAnsatz
from .operators import LocalOperator, basis_states


def vectorized_density_matrix(ansatz: DensityMatrixAnsatz, n_sites: int):
    single_basis = basis_states(n_sites)
    dim = len(single_basis)
    rows = np.repeat(single_basis, dim, axis=0)
    cols = np.tile(single_basis, (dim, 1))
    log_rho = ansatz.log_psi_batch(rows, cols)
    shift = np.max(np.real(log_rho))
    rho_vec = np.exp(log_rho - shift)
    return rho_vec, log_rho


def exact_density_matrix(ansatz: DensityMatrixAnsatz, n_sites: int) -> np.ndarray:
    rho_vec, _ = vectorized_density_matrix(ansatz, n_sites)
    dim = 1 << n_sites
    return rho_vec.reshape(dim, dim)


def exact_observables(ansatz: DensityMatrixAnsatz, observables: Dict[str, LocalOperator], n_sites: int) -> Dict[str, complex]:
    rho = exact_density_matrix(ansatz, n_sites)
    trace_rho = np.trace(rho)
    if abs(trace_rho) < 1e-14:
        raise ZeroDivisionError("Density matrix trace is too close to zero")
    out: Dict[str, complex] = {}
    for name, observable in observables.items():
        op_mat = observable.to_sparse().toarray()
        out[name] = np.trace(rho @ op_mat) / trace_rho
    return out
