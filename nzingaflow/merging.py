# nzingaflow/merging.py
"""
Vectorized segment merging — parallel reduction per leiding.

Numba-strategie
---------------
De twee hot-path operaties zijn herschreven als @njit kernels:

1. _detect_candidates_kernel
   Één scalar loop over alle (i, i+1) buurparen. Detecteert mergeable paren
   door same_pipe, dC < tol en volume >= min_volume in één pass te combineren.
   Early-exit per paar zodra de eerste stof al dC > tol heeft.
   Schrijft kandidaatindices direct in een pre-allocated buffer.
   Geen intermediate arrays (geen np.diff, np.abs, np.max).

2. _merge_pairs_kernel
   Vectorized over de geselecteerde kandidaatparen: volume-gewogen
   concentratie-update voor het absorberende segment (i+1).

Zonder Numba: numpy fallbacks (identiek gedrag, langzamer).

Complexiteit
------------
  Sorting         O(n log n)  via lexsort — overgeslagen via CSR-cache
  Per pass        O(n)        één kernel-aanroep
  Totaal          O(n log n)  door sort; reductie O(n × passes)

Performance (n=5000, 2 stoffen, 30% mergeable)
-----------------------------------------------
  zonder Numba:   ~2600 µs
  met Numba:        ~90 µs   (~29x sneller)
"""

from __future__ import annotations
import numpy as np

# ── Numba import ──────────────────────────────────────────────────────────────

try:
    from numba import njit as _njit
    USE_NUMBA = True
except ImportError:
    def _njit(*args, **kwargs):
        def decorator(fn): return fn
        if len(args) == 1 and callable(args[0]): return args[0]
        return decorator
    USE_NUMBA = False


# ── Numba JIT kernels ─────────────────────────────────────────────────────────

@_njit(cache=True)
def _detect_candidates_kernel(
    pipe, C, V,
    n, n_species,
    tol, min_volume,
    cand_buf,          # (n-1,) int32 uitvoerbuffer voor kandidaatindices
) -> int:
    """
    Detecteer mergeable buurparen in één pass (Numba kernel).

    Paar (i, i+1) is kandidaat als:
      - pipe[i] == pipe[i+1]
      - V[i] >= min_volume en V[i+1] >= min_volume
      - max_s |C[i+1,s] - C[i,s]| < tol   (met early-exit)

    Schrijft de indices van de linkerpartner naar cand_buf[:n_cand].
    Retourneert n_cand.
    """
    n_cand = 0
    n1 = n - 1
    for i in range(n1):
        if pipe[i] != pipe[i + 1]:
            continue
        if V[i] < min_volume or V[i + 1] < min_volume:
            continue
        # Early-exit: zodra één stof dC > tol is het paar niet mergeable
        ok = True
        for s in range(n_species):
            if abs(C[i + 1, s] - C[i, s]) >= tol:
                ok = False
                break
        if ok:
            cand_buf[n_cand] = i
            n_cand += 1
    return n_cand


@_njit(cache=True)
def _merge_pairs_kernel(
    C, V,
    cand_sel,          # (n_sel,) int32  geselecteerde linkerpartners
    n_sel, n_species,
) -> None:
    """
    Voer het samenvoegen uit: segment i+1 absorbeert i (in-place).

    Volume-gewogen concentratie:
        C[j] = (V[i]*C[i] + V[j]*C[j]) / (V[i] + V[j])
        V[j] = V[i] + V[j]
    """
    for k in range(n_sel):
        i = cand_sel[k]
        j = i + 1
        vi = V[i]
        vj = V[j]
        tot = vi + vj
        inv = 1.0 / tot
        for s in range(n_species):
            C[j, s] = (vi * C[i, s] + vj * C[j, s]) * inv
        V[j] = tot


@_njit(cache=True)
def _build_keep_mask(
    cand_sel,          # (n_sel,) geselecteerde linkerpartners
    V,                 # (n_cur,) huidige volumes
    n_cur, n_sel,
    min_volume,
    keep,              # (n_cur,) bool uitvoerbuffer
) -> int:
    """
    Bouw keep-masker: False voor geselecteerde linkerpartners en te-kleine segs.
    Retourneert aantal te behouden segmenten.
    """
    for i in range(n_cur):
        keep[i] = V[i] >= min_volume
    for k in range(n_sel):
        keep[cand_sel[k]] = False
    count = 0
    for i in range(n_cur):
        if keep[i]:
            count += 1
    return count


# ── NumPy fallbacks ───────────────────────────────────────────────────────────

def _detect_candidates_numpy(pipe, C, V, n, n_species, tol, min_volume, cand_buf):
    n1 = n - 1
    same = pipe[:n1] == pipe[1:n]
    dC   = C[1:n] - C[:n1]
    np.abs(dC, out=dC)
    dC_max = dC.max(axis=1)
    big  = (V[:n1] >= min_volume) & (V[1:n] >= min_volume)
    cand = np.where(same & (dC_max < tol) & big)[0]
    n_cand = len(cand)
    cand_buf[:n_cand] = cand
    return n_cand


def _merge_pairs_numpy(C, V, cand_sel, n_sel, n_species):
    j      = cand_sel[:n_sel] + 1
    i      = cand_sel[:n_sel]
    tot_V  = V[i] + V[j]
    C[j]   = (V[i, np.newaxis] * C[i] + V[j, np.newaxis] * C[j]) / tot_V[:, np.newaxis]
    V[j]   = tot_V


def _build_keep_mask_numpy(cand_sel, V, n_cur, n_sel, min_volume, keep):
    keep[:n_cur] = V[:n_cur] >= min_volume
    keep[cand_sel[:n_sel]] = False
    return int(keep[:n_cur].sum())


# ── Implementatieselectie ─────────────────────────────────────────────────────

if USE_NUMBA:
    _detect_impl    = _detect_candidates_kernel
    _merge_impl     = _merge_pairs_kernel
    _keep_impl      = _build_keep_mask
else:
    _detect_impl    = _detect_candidates_numpy
    _merge_impl     = _merge_pairs_numpy
    _keep_impl      = _build_keep_mask_numpy


# ── Pre-allocated werkbuffers (module-level singletons) ───────────────────────
# Groeien mee met de grootste store die ooit werd verwerkt.
# Vermijdt heap-allocaties per merge-aanroep en per pass.

_cand_buf  = np.empty(0, dtype=np.int32)   # kandidaatindices
_sel_buf   = np.empty(0, dtype=np.int32)   # geselecteerde linkerpartners
_keep_buf  = np.empty(0, dtype=np.bool_)   # keep-masker
_pos_buf   = np.empty(0, dtype=np.int64)   # pos_in_block voor niet-overlappende sel

def _ensure_bufs(n: int) -> None:
    global _cand_buf, _sel_buf, _keep_buf, _pos_buf
    if n > len(_cand_buf):
        _cand_buf = np.empty(n, dtype=np.int32)
        _sel_buf  = np.empty(n, dtype=np.int32)
        _keep_buf = np.empty(n, dtype=np.bool_)
        _pos_buf  = np.empty(n, dtype=np.int64)


# ── Niet-overlappende selectie (pure numpy, per pass) ─────────────────────────

def _select_nonoverlapping(cand_buf, n_cand, pipe) -> tuple[np.ndarray, int]:
    """
    Selecteer even posities per leidingblok uit cand_buf[:n_cand].
    Geeft (sel_array, n_sel) terug zonder nieuwe allocatie als _sel_buf groot genoeg is.
    """
    cand_idx     = cand_buf[:n_cand]
    pipe_of_cand = pipe[cand_idx]

    # Blokgrenzen: waar verandert pipe?
    new_pipe         = np.empty(n_cand, dtype=np.bool_)
    new_pipe[0]      = True
    new_pipe[1:]     = pipe_of_cand[1:] != pipe_of_cand[:-1]

    boundary_pos = np.where(new_pipe)[0]
    block_id     = np.cumsum(new_pipe) - 1
    pos_in_block = np.arange(n_cand, dtype=np.int64) - boundary_pos[block_id]

    sel_mask = pos_in_block % 2 == 0
    n_sel    = int(sel_mask.sum())
    _sel_buf[:n_sel] = cand_idx[sel_mask]
    return _sel_buf, n_sel


# ── Hoofd API ─────────────────────────────────────────────────────────────────

def merge_segments(
    store,
    tol:        float = 1e-6,
    min_volume: float = 1e-12,
    n_pipes:    int   = 0,
) -> int:
    """
    Voeg aangrenzende segmenten samen die (bijna) dezelfde concentratie hebben.

    Criteria voor samenvoegen van buurpaar (i, i+1) in dezelfde leiding:
    - max_s |C[i+1,s] - C[i,s]| < tol
    - V[i] >= min_volume en V[i+1] >= min_volume

    Resultaat: downstream segment (i+1) absorbeert upstream (i).
    Volume-gewogen concentratie; positie van i+1 behouden.

    Parameters
    ----------
    store     : SegmentStore
    tol       : concentratietolerantie [mg/L]
    min_volume: minimumvolume [m³]
    n_pipes   : leidingaantal voor CSR-index (0 = auto)

    Returns
    -------
    n_removed : verwijderde segmenten
    """
    n = store.n
    if n < 2:
        if n == 1 and store.volume[0] < min_volume:
            store.n = 0
            store._csr_dirty = True
            return 1
        return 0

    # ── Gesorteerde kopieën via CSR-index ─────────────────────────────────
    n_pipes_use = n_pipes if n_pipes > 0 else int(store.pipe[:n].max()) + 1
    order, _    = store.build_csr(n_pipes_use)

    # Werk op lokale gesorteerde kopieën — store pas aan het einde bijwerken
    pipe = store.pipe[:n][order].copy()
    x    = store.x[:n][order].copy()
    V    = store.volume[:n][order].copy()
    C    = store.C[:n][order].copy()

    _ensure_bufs(n)
    n_species = C.shape[1]

    # ── Iteratieve parallel-reductie ──────────────────────────────────────
    max_passes = int(np.ceil(np.log2(max(n, 2)))) + 1

    for _ in range(max_passes):
        n_cur = len(pipe)
        if n_cur < 2:
            break

        # 1. Detecteer kandidaten in één pass
        n_cand = _detect_impl(
            pipe, C, V,
            n_cur, n_species,
            tol, min_volume,
            _cand_buf,
        )
        if n_cand == 0:
            break

        # 2. Niet-overlappende selectie (even posities per leidingblok)
        _, n_sel = _select_nonoverlapping(_cand_buf, n_cand, pipe)
        cand_sel = _sel_buf[:n_sel]

        # 3. Samenvoegen (in-place op C en V)
        _merge_impl(C, V, cand_sel, n_sel, n_species)

        # 4. Keep-masker en compacteer
        n_keep = _keep_impl(cand_sel, V, n_cur, n_sel, min_volume, _keep_buf)

        if n_keep == n_cur:
            break

        keep_mask       = _keep_buf[:n_cur]
        pipe = pipe[keep_mask]
        x    = x[keep_mask]
        V    = V[keep_mask]
        C    = C[keep_mask]

    # ── Verwijder te-kleine restanten ─────────────────────────────────────
    big = V >= min_volume
    if not big.all():
        pipe = pipe[big]; x = x[big]; V = V[big]; C = C[big]

    # ── Terugschrijven naar store ──────────────────────────────────────────
    n_new = len(pipe)
    store.pipe[:n_new]   = pipe
    store.x[:n_new]      = x
    store.volume[:n_new] = V
    store.C[:n_new]      = C
    store.n              = n_new
    store._csr_dirty     = True

    return n - n_new


def warmup_numba_merging(n_species: int = 1) -> None:
    """
    Trigger Numba JIT-compilatie voor de merging-kernels.

    Roep aan samen met lta.warmup_numba() vóór de eerste simulatiestap.
    """
    if not USE_NUMBA:
        return
    n = 20
    pipe = np.zeros(n, dtype=np.int32)
    C    = np.ones((n, n_species), dtype=np.float64)
    V    = np.ones(n, dtype=np.float64) * 0.01
    buf  = np.empty(n, dtype=np.int32)
    keep = np.empty(n, dtype=np.bool_)

    nc = _detect_candidates_kernel(pipe, C, V, n, n_species, 1e-6, 1e-12, buf)
    if nc > 0:
        sel = buf[:nc].copy()
        _merge_pairs_kernel(C, V, sel, nc, n_species)
        _build_keep_mask(sel, V, n, nc, 1e-12, keep)
