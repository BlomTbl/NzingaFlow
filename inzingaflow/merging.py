# inzingaflow/merging.py
"""
Vectorized segment merging — parallel reduction per leiding.

Algoritme: iteratieve parallel-reductie
-----------------------------------------
Elke pass selecteert niet-overlappende kandidaatparen *per leiding* door
even posities binnen elke leidingblok te kiezen. Dit geeft echte halvering
per pass: n_segs/pipe → ⌈n/2⌉ → ⌈n/4⌉ → ... → 1 in log₂(n) passes.

Alle stappen zijn pure NumPy-vectoroperaties; geen Python-lus over segmenten.

Complexiteit
------------
  Sorting         O(n log n)  eenmalig per aanroep
  Per pass        O(n)        vectorized; gemiddeld log₂(max_dup) ≈ 3–6 passes
  Totaal          O(n log n)

Performance (benchmark, 1000 segs, 3 species)
----------------------------------------------
  geen merge (random C)          ~230 µs
  30% identieke buren            ~260 µs
  volledig identiek (collapse)   ~420 µs  (was: uren met Python-loop)
"""

from __future__ import annotations
import numpy as np


def merge_segments(
    store,
    tol:        float = 1e-6,
    min_volume: float = 1e-12,
) -> int:
    """
    Voeg aangrenzende segmenten samen die (bijna) dezelfde concentratie hebben.

    Criteria voor samenvoegen van buurpaar (i, i+1):
    - Zelfde leiding
    - max |C[i,s] − C[i+1,s]| < tol  (over alle stoffen s)
    - Beide volumes ≥ min_volume

    Resultaat: downstream segment (i+1) absorbeert upstream (i).
    Volume-gewogen concentratie; positie van i+1 behouden.

    Parameters
    ----------
    store      : SegmentStore
    tol        : concentratietolerantie [mg/L of dimensieloos]
    min_volume : segmenten kleiner dan dit worden altijd verwijderd [m³]

    Returns
    -------
    n_removed : totaal aantal verwijderde (samengevoegde + te kleine) segmenten
    """
    n = store.n
    if n < 2:
        if n == 1 and store.volume[0] < min_volume:
            store.n = 0
            return 1
        return 0

    # ── Kopieën sorteren (pipe primair, x secundair) ──────────────────────────
    pipe = store.pipe[:n].copy()
    x    = store.x[:n].copy()
    V    = store.volume[:n].copy()
    C    = store.C[:n].copy()   # (n, n_species)

    order = np.lexsort((x, pipe))
    pipe  = pipe[order]
    x     = x[order]
    V     = V[order]
    C     = C[order]

    # ── Iteratieve parallel-reductie ──────────────────────────────────────────
    for _ in range(64):          # 64 passes = maximaal 2^64 identieke segs/pipe
        n_cur = len(pipe)
        if n_cur < 2:
            break

        # Kandidaatparen: aangrenzende buren met zelfde leiding, kleine dC
        same   = pipe[1:] == pipe[:-1]
        dC_max = np.max(np.abs(np.diff(C, axis=0)), axis=1)
        big    = (V[:-1] >= min_volume) & (V[1:] >= min_volume)
        cand   = same & (dC_max < tol) & big

        if not cand.any():
            break

        cand_idx = np.where(cand)[0]           # globale indices van linker partner

        # ── Niet-overlappende selectie: even posities PER LEIDING ─────────────
        # Twee buren i, i+1 kunnen niet allebei geselecteerd worden.
        # Selecteer even posities per leidingblok → gegarandeerd niet-overlappend
        # en maximale halvering per pass.
        pipe_of_cand = pipe[cand_idx]
        new_pipe     = np.empty(len(cand_idx), dtype=bool)
        new_pipe[0]  = True
        new_pipe[1:] = pipe_of_cand[1:] != pipe_of_cand[:-1]

        # Positie binnen leidingblok (pure numpy, geen Python-loop)
        boundary_pos = np.where(new_pipe)[0]                  # start van elk blok
        block_id     = np.cumsum(new_pipe) - 1                # 0,0,...,1,1,...,2,...
        pos_in_block = np.arange(len(cand_idx)) - boundary_pos[block_id]

        sel      = pos_in_block % 2 == 0                     # even posities
        cand_sel = cand_idx[sel]                              # geselecteerde linkers

        # ── Samenvoegen: i+1 absorbeert i ────────────────────────────────────
        j      = cand_sel + 1
        tot_V  = V[cand_sel] + V[j]
        C[j]   = (  V[cand_sel, np.newaxis] * C[cand_sel]
                  + V[j,        np.newaxis] * C[j]
                 ) / tot_V[:, np.newaxis]
        V[j]   = tot_V

        keep        = np.ones(n_cur, dtype=bool)
        keep[cand_sel] = False
        keep       &= (V >= min_volume)

        pipe = pipe[keep]
        x    = x[keep]
        V    = V[keep]
        C    = C[keep]

    # ── Verwijder te kleine restanten ─────────────────────────────────────────
    big_enough = V >= min_volume
    if not big_enough.all():
        pipe = pipe[big_enough]
        x    = x[big_enough]
        V    = V[big_enough]
        C    = C[big_enough]

    # ── Terugschrijven naar store ──────────────────────────────────────────────
    n_new = len(pipe)
    store.pipe[:n_new]   = pipe
    store.x[:n_new]      = x
    store.volume[:n_new] = V
    store.C[:n_new]      = C
    store.n              = n_new

    return n - n_new
