"""
Relatieve concentratietolerantie voor segmentmerging (geen epynet vereist).

M1  Default (rel_tol=0) gedraagt zich als voorheen: geen merge bij 0,3% spreiding
M2  rel_tol merged wel, en de totale massa per stof blijft exact behouden
M3  De drempel schaalt met de concentratie (zelfde absolute verschil, ander effect)
M4  Numba-kernel en NumPy-fallback geven hetzelfde resultaat
"""
from __future__ import annotations

import numpy as np
import pytest

from nzingaflow.segments import SegmentStore
import nzingaflow.merging as merging
from nzingaflow.merging import merge_segments


def _store_linear(n: int, spread: float, c0: float = 1.0,
                  n_species: int = 1, vol: float = 1.0) -> SegmentStore:
    """n segmenten in één leiding, concentratie loopt lineair op over `spread`."""
    st = SegmentStore(capacity=max(4 * n, 64), n_species=n_species)
    for i in range(n):
        c = c0 + spread * i / (n - 1)
        st.add(0, float(i), vol, np.full(n_species, c))
    return st


def _mass(st: SegmentStore) -> np.ndarray:
    n = st.n
    return (st.volume[:n, None] * st.C[:n]).sum(axis=0)


def test_m1_default_is_unchanged():
    st = _store_linear(50, 0.003)
    merge_segments(st, tol=1e-6, n_pipes=1)          # rel_tol default = 0
    assert st.n == 50


def test_m2_relative_merges_and_conserves_mass():
    st = _store_linear(50, 0.003)
    m0 = _mass(st)
    removed = merge_segments(st, tol=1e-6, n_pipes=1, rel_tol=0.005)
    assert st.n < 50
    assert removed == 50 - st.n
    np.testing.assert_allclose(_mass(st), m0, rtol=1e-12)


def test_m3_threshold_scales_with_concentration():
    # Zelfde absoluut verschil (0,02 mg/L), andere relatieve grootte
    st_low = _store_linear(2, 0.02, c0=1.0)     # 2% van C -> boven 0,5%: niet mergen
    st_high = _store_linear(2, 0.02, c0=100.0)  # 0,02% van C -> onder 0,5%: mergen
    assert merge_segments(st_low, tol=1e-6, n_pipes=1, rel_tol=0.005) == 0
    assert merge_segments(st_high, tol=1e-6, n_pipes=1, rel_tol=0.005) == 1


def test_m4_numba_and_numpy_paths_agree(monkeypatch):
    if not merging.USE_NUMBA:
        pytest.skip("numba niet beschikbaar")
    st_numba = _store_linear(200, 0.01)
    merge_segments(st_numba, tol=1e-6, n_pipes=1, rel_tol=0.003)

    monkeypatch.setattr(merging, "_detect_impl", merging._detect_candidates_numpy)
    monkeypatch.setattr(merging, "_merge_impl", merging._merge_pairs_numpy)
    monkeypatch.setattr(merging, "_keep_impl", merging._build_keep_mask_numpy)
    st_numpy = _store_linear(200, 0.01)
    merge_segments(st_numpy, tol=1e-6, n_pipes=1, rel_tol=0.003)

    assert st_numba.n == st_numpy.n
    np.testing.assert_allclose(st_numba.C[:st_numba.n], st_numpy.C[:st_numpy.n])
    np.testing.assert_allclose(st_numba.volume[:st_numba.n], st_numpy.volume[:st_numpy.n])
