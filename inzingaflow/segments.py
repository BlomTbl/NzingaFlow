# nzingaflow/segments.py

from __future__ import annotations
import numpy as np


class SegmentStore:
    """
    Hoge-performance segment-opslag (Structure-of-Arrays).

    Ontwerp
    -------
    - Vaste pre-allocatie van `capacity` slots; geen heap-allocs tijdens simulatie.
    - `n` = aantal actieve segmenten; arrays[0:n] zijn geldig.
    - C is 2-D: shape (capacity, n_species) voor multi-species ondersteuning.
    - Segmenten worden verwijderd via swap-with-last (O(1), volgorde-onafhankelijk).
    - `resize()` verdubbelt de capaciteit als die vol raakt.

    CSR pipe-index
    --------------
    Voor merge_segments is een gesorteerde index per leiding nodig (lexsort op
    (pipe, x)).  Die sort kost ~500 µs voor 5000 segmenten.  De CSR-index
    (pipe_ptr, pipe_order) slaat dit resultaat op en markeert zichzelf dirty
    na elke add()/remove().  merge_segments roept build_csr() aan om de index
    op te bouwen; als hij nog geldig is (not dirty) wordt de sort overgeslagen.

    on_resize callback
    ------------------
    Solvers die exit-buffers pre-alloceren op basis van de capaciteit kunnen
    via on_resize(new_capacity) mee schalen wanneer de store verdubbelt.
    """

    __slots__ = (
        "pipe", "x", "C", "volume", "n", "n_species", "_capacity", "on_resize",
        "_csr_order", "_csr_ptr", "_csr_n_pipes", "_csr_dirty",
    )

    def __init__(self, capacity: int = 100_000, n_species: int = 1):
        self.n_species  = n_species
        self._capacity  = capacity
        self.pipe   = np.empty(capacity, dtype=np.int32)
        self.x      = np.empty(capacity, dtype=np.float64)
        self.C      = np.zeros((capacity, n_species), dtype=np.float64)
        self.volume = np.empty(capacity, dtype=np.float64)
        self.n      = 0
        self.on_resize  = None   # callable(new_capacity) of None
        # CSR-index: pipe_order[pipe_ptr[p]:pipe_ptr[p+1]] = indices in store
        # voor pipe p, gesorteerd op x.  Alleen geldig als _csr_dirty == False.
        self._csr_order:   np.ndarray | None = None
        self._csr_ptr:     np.ndarray | None = None
        self._csr_n_pipes: int               = 0
        self._csr_dirty:   bool              = True

    # ── Toevoegen ─────────────────────────────────────────────────────────────

    def add(self, pipe: int, x: float, volume: float, C_vector) -> None:
        """
        Voeg één segment toe.

        Parameters
        ----------
        pipe     : pipe-index (0-based)
        x        : positie langs de leiding (m vanaf beginpunt)
        volume   : segmentvolume (m³)
        C_vector : concentraties, lengte n_species
        """
        if self.n >= self._capacity:
            self._resize()
        i = self.n
        self.pipe[i]   = pipe
        self.x[i]      = x
        self.volume[i] = volume
        self.C[i]      = C_vector
        self.n        += 1
        self._csr_dirty = True

    # ── Verwijderen ───────────────────────────────────────────────────────────

    def remove(self, indices) -> None:
        """
        Verwijder segmenten op de opgegeven indices via swap-with-last.

        Parameters
        ----------
        indices : gesorteerde lijst/array van te verwijderen indices (aflopend)

        Waarschuwing
        ------------
        Swap-with-last verandert de volgorde van actieve segmenten. Elke
        array-view op ``store.pipe[:n]``, ``store.C[:n]`` etc. die vóór deze
        aanroep is vastgelegd, verwijst daarna naar mogelijk verplaatste data.
        Haal views pas op na de laatste ``remove()``-aanroep in een tijdstap.
        """
        for ri in sorted(indices, reverse=True):
            last = self.n - 1
            if ri != last:
                self.pipe[ri]   = self.pipe[last]
                self.x[ri]      = self.x[last]
                self.C[ri]      = self.C[last]
                self.volume[ri] = self.volume[last]
            self.n -= 1
        self._csr_dirty = True

    # ── CSR pipe-index ────────────────────────────────────────────────────────

    def build_csr(self, n_pipes: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Bouw (of hergebruik) de CSR pipe-index.

        Resultaat: pipe_order, pipe_ptr zodat
            pipe_order[pipe_ptr[p] : pipe_ptr[p+1]]
        de indices geeft van segmenten in pipe p, gesorteerd op x.

        De index wordt gecached; herbouw alleen als _csr_dirty == True of
        n_pipes veranderd is.

        Parameters
        ----------
        n_pipes : aantal leidingen in het netwerk

        Returns
        -------
        pipe_order : (n,) int64  indices in store, gesorteerd op (pipe, x)
        pipe_ptr   : (n_pipes+1,) int64  startposities per pipe
        """
        n = self.n
        if (not self._csr_dirty
                and self._csr_order is not None
                and self._csr_n_pipes == n_pipes
                and len(self._csr_order) == n):
            return self._csr_order, self._csr_ptr

        if n == 0:
            self._csr_order   = np.empty(0, dtype=np.int64)
            self._csr_ptr     = np.zeros(n_pipes + 1, dtype=np.int64)
            self._csr_n_pipes = n_pipes
            self._csr_dirty   = False
            return self._csr_order, self._csr_ptr

        order = np.lexsort((self.x[:n], self.pipe[:n])).astype(np.int64)
        counts = np.bincount(self.pipe[:n], minlength=n_pipes)
        ptr    = np.zeros(n_pipes + 1, dtype=np.int64)
        np.cumsum(counts, out=ptr[1:])

        self._csr_order   = order
        self._csr_ptr     = ptr
        self._csr_n_pipes = n_pipes
        self._csr_dirty   = False
        return order, ptr

    # ── Views op actieve data ─────────────────────────────────────────────────

    def active(self) -> slice:
        """Geeft slice(0, n) terug; gebruik voor array-indexering."""
        return slice(0, self.n)

    @property
    def pipe_a(self)   -> np.ndarray: return self.pipe[:self.n]

    @property
    def x_a(self)      -> np.ndarray: return self.x[:self.n]

    @property
    def C_a(self)      -> np.ndarray: return self.C[:self.n]

    @property
    def volume_a(self) -> np.ndarray: return self.volume[:self.n]

    # ── Capaciteitsbeheer ─────────────────────────────────────────────────────

    def _resize(self) -> None:
        """Verdubbel de opslagcapaciteit (zonder data-wrapping).

        np.resize() herhaalt data bij vergroting — daarom alloceren we
        nieuwe arrays en kopiëren we alleen de actieve data [:n].

        Na de resize wordt ``on_resize`` aangeroepen als die is ingesteld.
        Solvers die exit-buffers pre-alloceren op basis van de capaciteit
        kunnen daarin hun buffers mee schalen.
        """
        new_cap = self._capacity * 2
        new_pipe   = np.empty(new_cap, dtype=np.int32)
        new_x      = np.empty(new_cap, dtype=np.float64)
        new_volume = np.empty(new_cap, dtype=np.float64)
        new_C      = np.zeros((new_cap, self.n_species), dtype=np.float64)

        new_pipe[:self.n]   = self.pipe[:self.n]
        new_x[:self.n]      = self.x[:self.n]
        new_volume[:self.n] = self.volume[:self.n]
        new_C[:self.n]      = self.C[:self.n]

        self.pipe      = new_pipe
        self.x         = new_x
        self.volume    = new_volume
        self.C         = new_C
        self._capacity = new_cap
        self._csr_dirty = True

        if self.on_resize is not None:
            self.on_resize(new_cap)

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f"<SegmentStore n={self.n}/{self._capacity} "
                f"n_species={self.n_species} csr={'valid' if not self._csr_dirty else 'dirty'}>")
