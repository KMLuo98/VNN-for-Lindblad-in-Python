from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from scipy import sparse


def sigmax() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def sigmay() -> np.ndarray:
    return np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)


def sigmaz() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)


def sigmam() -> np.ndarray:
    return np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.complex128)


def _spin_to_bit(spin) -> int:
    return 0 if spin > 0 else 1


def _bits_to_spins(bits: np.ndarray) -> np.ndarray:
    return np.where(bits == 0, 1.0, -1.0)


def basis_states(n_sites: int) -> np.ndarray:
    dim = 1 << n_sites
    states = np.zeros((dim, n_sites), dtype=np.float64)
    for idx in range(dim):
        bits = np.array([(idx >> shift) & 1 for shift in range(n_sites - 1, -1, -1)], dtype=np.int8)
        states[idx] = _bits_to_spins(bits)
    return states


@dataclass
class LocalTerm:
    coefficient: complex
    sites: Tuple[int, ...]
    matrix: np.ndarray

    def __post_init__(self) -> None:
        self.matrix = np.asarray(self.matrix, dtype=np.complex128)
        expected = 1 << len(self.sites)
        if self.matrix.shape != (expected, expected):
            raise ValueError("Local matrix shape does not match number of sites")

    def _local_index(self, config: np.ndarray) -> int:
        bits = [_spin_to_bit(config[site]) for site in self.sites]
        index = 0
        for bit in bits:
            index = (index << 1) | bit
        return index

    def _set_local_index(self, config: np.ndarray, local_index: int) -> np.ndarray:
        updated = np.array(config, copy=True)
        for site in reversed(self.sites):
            bit = local_index & 1
            updated[site] = 1.0 if bit == 0 else -1.0
            local_index >>= 1
        return updated

    def row_connections(self, config: np.ndarray) -> List[Tuple[np.ndarray, complex]]:
        row_idx = self._local_index(config)
        amplitudes = self.coefficient * self.matrix[row_idx, :]
        out = []
        for col_idx, amplitude in enumerate(amplitudes):
            if abs(amplitude) > 1e-14:
                out.append((self._set_local_index(config, col_idx), amplitude))
        return out


class LocalOperator:
    def __init__(self, n_sites: int, terms=None) -> None:
        self.n_sites = n_sites
        self.terms = list(terms or [])

    def add_term(self, coefficient: complex, sites: Sequence[int], local_matrix: np.ndarray) -> None:
        self.terms.append(LocalTerm(complex(coefficient), tuple(sites), local_matrix))

    def __iadd__(self, other: "LocalOperator") -> "LocalOperator":
        if self.n_sites != other.n_sites:
            raise ValueError("Incompatible operator sizes")
        self.terms.extend(other.terms)
        return self

    def scaled(self, coefficient: complex) -> "LocalOperator":
        return LocalOperator(
            self.n_sites,
            [LocalTerm(coefficient * term.coefficient, term.sites, term.matrix) for term in self.terms],
        )

    def dagger(self) -> "LocalOperator":
        return LocalOperator(
            self.n_sites,
            [LocalTerm(np.conj(term.coefficient), term.sites, term.matrix.conj().T) for term in self.terms],
        )

    def to_sparse(self) -> sparse.csr_matrix:
        states = basis_states(self.n_sites)
        dim = len(states)
        rows: List[int] = []
        cols: List[int] = []
        data: List[complex] = []
        for out_idx, config in enumerate(states):
            for term in self.terms:
                for in_config, amplitude in term.row_connections(config):
                    in_idx = spins_to_index(in_config)
                    rows.append(out_idx)
                    cols.append(in_idx)
                    data.append(amplitude)
        return sparse.coo_matrix((data, (rows, cols)), shape=(dim, dim), dtype=np.complex128).tocsr()


def kron_operator_list(ops: Sequence[np.ndarray]) -> np.ndarray:
    out = np.array([[1.0 + 0.0j]], dtype=np.complex128)
    for op in ops:
        out = np.kron(out, op)
    return out


def one_site_operator(n_sites: int, site: int, matrix: np.ndarray, coefficient: complex = 1.0) -> LocalOperator:
    op = LocalOperator(n_sites)
    op.add_term(coefficient, (site,), matrix)
    return op


def two_site_operator(
    n_sites: int, site_a: int, site_b: int, matrix_a: np.ndarray, matrix_b: np.ndarray, coefficient: complex = 1.0
) -> LocalOperator:
    op = LocalOperator(n_sites)
    op.add_term(coefficient, (site_a, site_b), kron_operator_list((matrix_a, matrix_b)))
    return op


def spins_to_index(config: np.ndarray) -> int:
    idx = 0
    for spin in config:
        idx = (idx << 1) | _spin_to_bit(spin)
    return idx


class Liouvillian:
    def __init__(self, hamiltonian: LocalOperator, jumps: Sequence[LocalOperator]) -> None:
        self.hamiltonian = hamiltonian
        self.jumps = list(jumps)
        self.n_sites = hamiltonian.n_sites
        h_mat = hamiltonian.to_sparse()
        dim = h_mat.shape[0]
        ident = sparse.identity(dim, dtype=np.complex128, format="csr")
        dissipator = sparse.csr_matrix((dim, dim), dtype=np.complex128)
        for jump in self.jumps:
            l_mat = jump.to_sparse()
            dissipator = dissipator + l_mat.getH().dot(l_mat)
        h_eff = h_mat - 0.5j * dissipator
        left = (-1.0j) * sparse.kron(h_eff, ident, format="csr")
        right = (1.0j) * sparse.kron(ident, h_eff.conjugate(), format="csr")
        recycle = sparse.csr_matrix((dim * dim, dim * dim), dtype=np.complex128)
        for jump in self.jumps:
            l_mat = jump.to_sparse()
            recycle = recycle + sparse.kron(l_mat, l_mat.conjugate(), format="csr")
        self.matrix = left + right + recycle
        self.ldagl = self.matrix.getH().dot(self.matrix).tocsr()


def dissipative_ising_1d(n_sites: int, g: float, v: float, gamma: float = 1.0) -> Tuple[LocalOperator, List[LocalOperator], Liouvillian, dict]:
    hamiltonian = LocalOperator(n_sites)
    sx = LocalOperator(n_sites)
    sy = LocalOperator(n_sites)
    sz = LocalOperator(n_sites)
    jumps = []
    for site in range(n_sites):
        next_site = (site + 1) % n_sites
        hamiltonian += one_site_operator(n_sites, site, sigmax(), coefficient=0.5 * g)
        hamiltonian += two_site_operator(n_sites, site, next_site, sigmaz(), sigmaz(), coefficient=0.25 * v)
        sx += one_site_operator(n_sites, site, sigmax(), coefficient=1.0 / n_sites)
        sy += one_site_operator(n_sites, site, sigmay(), coefficient=1.0 / n_sites)
        sz += one_site_operator(n_sites, site, sigmaz(), coefficient=1.0 / n_sites)
        jumps.append(one_site_operator(n_sites, site, sigmam(), coefficient=np.sqrt(gamma)))
    liouvillian = Liouvillian(hamiltonian, jumps)
    observables = {"Sx": sx, "Sy": sy, "Sz": sz, "H": hamiltonian}
    return hamiltonian, jumps, liouvillian, observables
