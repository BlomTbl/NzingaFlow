"""
De snelle sortering in SegmentStore.build_csr geeft exact dezelfde volgorde als np.lexsort.

De sortering gebruikt een samengestelde float64-sleutel (pipe * span + x). Die kan bij
grote netwerken of vrijwel gelijke posities afronden; dan moet de controle in
_sort_by_pipe_x terugvallen op np.lexsort zodat het resultaat nooit afwijkt.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from nzingaflow import segments
from nzingaflow.merging import merge_segments
from nzingaflow.segments import SegmentStore, _sort_by_pipe_x


def _lexsort(pipe, x):
    return np.lexsort((x, pipe))


@pytest.mark.parametrize("n", [1, 2, 3, 10, 500, 5000])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_data_matches_lexsort(n, seed):
    rng = np.random.default_rng(seed)
    pipe = rng.integers(0, 200, n).astype(np.int32)
    x = rng.random(n) * 100
    assert np.array_equal(_sort_by_pipe_x(pipe, x), _lexsort(pipe, x))


def test_many_ties_matches_lexsort():
    rng = np.random.default_rng(3)
    pipe = rng.integers(0, 20, 5000).astype(np.int32)
    x = rng.integers(0, 50, 5000).astype(float)  # veel exact gelijke (pipe, x)
    assert np.array_equal(_sort_by_pipe_x(pipe, x), _lexsort(pipe, x))


@pytest.mark.parametrize("fractie", [0.0, 0.01, 0.1, 0.5])
def test_nearly_sorted_store_matches_lexsort(fractie):
    """Zoals tussen twee merges: gesorteerd, daarna een deel dat van leiding wisselt."""
    rng = np.random.default_rng(4)
    n = 5000
    pipe = np.sort(rng.integers(0, 200, n)).astype(np.int32)
    x = np.empty(n)
    for p in range(200):
        m = pipe == p
        x[m] = np.sort(rng.random(int(m.sum())) * 100)
    k = int(fractie * n)
    idx = rng.choice(n, k, replace=False)
    pipe[idx] = rng.integers(0, 200, k)
    x[idx] = rng.random(k) * 2
    assert np.array_equal(_sort_by_pipe_x(pipe, x), _lexsort(pipe, x))


def test_precision_loss_falls_back_to_lexsort_and_still_matches(monkeypatch):
    """Zeer groot netwerk + posities die in de sleutel samenvallen: de controle moet ingrijpen."""
    rng = np.random.default_rng(5)
    n = 4000
    pipe = rng.integers(0, 2_000_000, n).astype(np.int64)
    x = (
        1.0e5 + rng.random(n) * 1.0e-7
    )  # verschillen kleiner dan de float64-resolutie van de sleutel
    pipe[: n // 2] = pipe[n // 2 :][: n // 2]  # veel segmenten in dezelfde leiding

    aanroepen = []
    echte = np.lexsort
    monkeypatch.setattr(np, "lexsort", lambda *a, **k: aanroepen.append(1) or echte(*a, **k))

    assert np.array_equal(_sort_by_pipe_x(pipe, x), echte((x, pipe)))
    assert aanroepen, "de terugval op lexsort is niet gebruikt; de test beproeft niets"


@pytest.mark.parametrize("waarde", [np.nan, np.inf, -np.inf])
def test_non_finite_x_matches_lexsort_without_warnings(waarde):
    rng = np.random.default_rng(6)
    pipe = rng.integers(0, 10, 300).astype(np.int32)
    x = rng.random(300) * 100
    x[::37] = waarde
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        resultaat = _sort_by_pipe_x(pipe, x)
    assert np.array_equal(resultaat, _lexsort(pipe, x))


def _store(rng, n=3000, n_pipes=100, n_species=2, vlak=False):
    s = SegmentStore(capacity=n + 10, n_species=n_species)
    s.n = n
    s.pipe[:n] = rng.integers(0, n_pipes, n)
    s.x[:n] = rng.random(n) * 100
    s.volume[:n] = 1e-3
    s.C[:n] = 1.0 if vlak else np.round(rng.random((n, n_species)) * 3) / 3
    s._csr_dirty = True
    return s


def test_build_csr_order_and_pointers_are_consistent():
    s = _store(np.random.default_rng(7))
    order, ptr = s.build_csr(100)
    n = s.n
    assert sorted(order.tolist()) == list(range(n))
    p, x = s.pipe[:n][order], s.x[:n][order]
    assert np.all(np.diff(p) >= 0)
    for leiding in range(100):
        blok = x[ptr[leiding] : ptr[leiding + 1]]
        assert np.all(np.diff(blok) >= 0)
        assert np.all(p[ptr[leiding] : ptr[leiding + 1]] == leiding)


@pytest.mark.parametrize("vlak", [False, True])
def test_merge_segments_result_is_identical_to_lexsort_reference(monkeypatch, vlak):
    """Eindresultaat van merge_segments is gelijk, met de snelle sortering en met lexsort."""
    rng_a = np.random.default_rng(8)
    rng_b = np.random.default_rng(8)
    snel = _store(rng_a, vlak=vlak)
    referentie = _store(rng_b, vlak=vlak)

    verwijderd_snel = merge_segments(snel, tol=1e-6, n_pipes=100)

    monkeypatch.setattr(segments, "_sort_by_pipe_x", _lexsort)
    verwijderd_ref = merge_segments(referentie, tol=1e-6, n_pipes=100)

    assert verwijderd_snel == verwijderd_ref
    n = snel.n
    assert n == referentie.n
    for veld in ("pipe", "x", "volume", "C"):
        assert np.array_equal(getattr(snel, veld)[:n], getattr(referentie, veld)[:n]), veld
