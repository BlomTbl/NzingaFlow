# inzingaflow/segments.py

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
    """

    __slots__ = ("pipe", "x", "C", "volume", "n", "n_species", "_capacity")

    def __init__(self, capacity: int = 100_000, n_species: int = 1):
        self.n_species  = n_species
        self._capacity  = capacity
        self.pipe   = np.empty(capacity, dtype=np.int32)
        self.x      = np.empty(capacity, dtype=np.float64)
        self.C      = np.zeros((capacity, n_species), dtype=np.float64)
        self.volume = np.empty(capacity, dtype=np.float64)
        self.n      = 0

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

    # ── Verwijderen ───────────────────────────────────────────────────────────

    def remove(self, indices) -> None:
        """
        Verwijder segmenten op de opgegeven indices via swap-with-last.

        Parameters
        ----------
        indices : gesorteerde lijst/array van te verwijderen indices (aflopend)
        """
        for ri in sorted(indices, reverse=True):
            last = self.n - 1
            if ri != last:
                self.pipe[ri]   = self.pipe[last]
                self.x[ri]      = self.x[last]
                self.C[ri]      = self.C[last]
                self.volume[ri] = self.volume[last]
            self.n -= 1

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

    def __len__(self) -> int:
        return self.n

    def __repr__(self) -> str:
        return (f"<SegmentStore n={self.n}/{self._capacity} "
                f"n_species={self.n_species}>")
