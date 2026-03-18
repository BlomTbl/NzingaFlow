# nzingaflow/stability.py
"""
Automatische tijdstap-stabiliteitscontrole en massabalansvalidatie.

CFL-voorwaarde (Courant-Friedrichs-Lewy)
-----------------------------------------
Zorgt ervoor dat geen enkel segment verder dan zijn leiding reist in één stap:
    dt_CFL ≤ min(pipe_length / velocity)

Reactie-stabiliteit
--------------------
Expliciete integratie is stabiel als:
    dt_rxn ≤ 0.1 / k_max    (10× veiligheidsmarge)

Aanbevolen dt = min(dt_CFL, dt_rxn) × safety_factor  (default 0.9)

Massabalans
-----------
Per tijdstap of cumulatief:
    - Totale massa = Σ(C[i,s] · V[i]) over alle actieve segmenten
    - Verwacht na verval: M(t+dt) = M(t) · exp(-k · dt)  (bulk only)
    - Relatieve fout < tol → OK
"""

from __future__ import annotations
import warnings
import numpy as np


# ── Tijdstapcontrole ──────────────────────────────────────────────────────────

def recommended_dt(
    pipe_length:  np.ndarray,   # (n_pipes,) [m]
    velocity:     np.ndarray,   # (n_pipes,) [m/s]
    k_bulk:       np.ndarray,   # (n_species,) bulk [1/s]
    k_wall_vol:   np.ndarray | None = None,  # (n_pipes, n_species) [1/s]
    safety:       float = 0.9,
    min_dt:       float = 0.1,
    max_dt:       float = 3600.0,
) -> float:
    """
    Bereken de maximale stabiele tijdstap.

    Parameters
    ----------
    pipe_length : leidinglengtes [m]
    velocity    : stroomsnelheden [m/s]
    k_bulk      : bulkvervalconstanten per stof [1/s]
    k_wall_vol  : volumetrische wandvervalconstanten [1/s] (optioneel)
    safety      : veiligheidsfactor (0 < safety ≤ 1)
    min_dt      : minimum toegestane tijdstap [s]
    max_dt      : maximum toegestane tijdstap [s]

    Returns
    -------
    dt : aanbevolen tijdstap [s]
    """
    limits = {}

    # CFL: geen segment mag zijn leiding overspringen
    v_max = np.maximum(velocity, 1e-10)
    dt_cfl = np.min(pipe_length / v_max)
    limits['CFL']  = dt_cfl

    # Reactiestabiliteit: 10% verandering per stap als grens
    k_total = np.asarray(k_bulk, dtype=float)
    if k_wall_vol is not None:
        k_total = k_total + np.max(k_wall_vol, axis=0)
    k_max = k_total.max()
    if k_max > 0:
        dt_rxn = 0.1 / k_max
        limits['reaction'] = dt_rxn

    dt_rec = min(limits.values()) * safety

    # Klamp NOOIT boven dt_CFL — ook niet als min_dt > dt_CFL.
    # We klampen wél naar beneden op min_dt zodat de tijdstap niet
    # onrealistisch klein wordt voor praktisch stilstaande leidingen,
    # maar als de CFL-grens zelf al kleiner is dan min_dt, dan is
    # min_dt niet van toepassing en hanteren we de CFL-grens.
    dt_cfl_safe = limits.get('CFL', np.inf) * safety
    effective_min = min(min_dt, dt_cfl_safe)
    dt_rec = float(np.clip(dt_rec, effective_min, max_dt))

    return dt_rec


def check_dt(
    dt:           float,
    pipe_length:  np.ndarray,
    velocity:     np.ndarray,
    k_bulk:       np.ndarray,
    k_wall_vol:   np.ndarray | None = None,
    warn:         bool = True,
) -> dict:
    """
    Controleer of een opgegeven tijdstap stabiel is.

    Returns
    -------
    report : dict met sleutels
        'cfl_ok'      : bool
        'reaction_ok' : bool
        'stable'      : bool  (beide OK)
        'dt_rec'      : float aanbevolen dt
        'cfl_ratio'   : dt / dt_CFL  (< 1 is veilig)
        'violations'  : list[str] beschrijvingen van schendingen
    """
    dt_rec = recommended_dt(pipe_length, velocity, k_bulk, k_wall_vol,
                            safety=1.0, min_dt=0.0, max_dt=1e9)

    v_max   = np.maximum(velocity, 1e-10)
    dt_cfl  = np.min(pipe_length / v_max)
    cfl_ok  = dt <= dt_cfl

    k_total = np.asarray(k_bulk, dtype=float)
    if k_wall_vol is not None:
        k_total = k_total + np.max(k_wall_vol, axis=0)
    k_max      = k_total.max()
    dt_rxn     = (0.1 / k_max) if k_max > 0 else np.inf
    rxn_ok     = dt <= dt_rxn

    violations = []
    if not cfl_ok:
        worst_pipe = int(np.argmin(pipe_length / v_max))
        violations.append(
            f"CFL: dt={dt:.1f}s > dt_CFL={dt_cfl:.1f}s "
            f"(worst pipe {worst_pipe}: L={pipe_length[worst_pipe]:.1f}m, "
            f"v={velocity[worst_pipe]:.3f}m/s)"
        )
    if not rxn_ok:
        violations.append(
            f"Reactie: dt={dt:.1f}s > dt_rxn={dt_rxn:.1f}s "
            f"(k_max={k_max:.4f}/s)"
        )

    if warn and violations:
        for v in violations:
            warnings.warn(f"Tijdstap instabiliteit: {v}", RuntimeWarning, stacklevel=2)

    return {
        'cfl_ok':      cfl_ok,
        'reaction_ok': rxn_ok,
        'stable':      cfl_ok and rxn_ok,
        'dt_rec':      dt_rec,
        'cfl_ratio':   dt / dt_cfl,
        'violations':  violations,
    }


# ── Massabalans ───────────────────────────────────────────────────────────────

class MassBalanceTracker:
    """
    Houdt de massabalans bij over de gehele simulatie.

    Volgt:
    - Ingespoten massa per stof (inject_mass)
    - Massa in het systeem (store)
    - Gecumuleerde verdwenen massa via bulkverval
    - Gecumuleerde verdwenen massa via wandreacties
    - Massa bij eindknopen (uit het systeem gevloeid)

    Gebruik
    -------
        tracker = MassBalanceTracker(n_species=2)
        tracker.record_injection(C_vector, volume)   # bij elke injectie
        tracker.record_step(store, node_C, flow,
                            decay_k, k_wall_vol, dt)  # na elke tijdstap
        report = tracker.report()
    """

    def __init__(self, n_species: int):
        self.n_species    = n_species
        self.injected     = np.zeros(n_species)
        self.bulk_decay   = np.zeros(n_species)
        self.wall_decay   = np.zeros(n_species)
        self.outflow      = np.zeros(n_species)
        self._prev_mass   = np.zeros(n_species)
        self._step        = 0

    def record_injection(self, C_vector, volume: float) -> None:
        """Registreer een injectie-segment."""
        self.injected += np.asarray(C_vector) * volume

    def record_step(
        self,
        store,
        decay_k:     np.ndarray,           # (n_species,) bulk [1/s]
        dt:          float,
        k_wall_vol:  np.ndarray | None = None,  # (n_pipes, n_species) [1/s]
        exited_C:    np.ndarray | None = None,  # (n_exit, n_species)
        exited_V:    np.ndarray | None = None,  # (n_exit,)
    ) -> None:
        """
        Registreer één tijdstap voor de massabalans.

        Parameters
        ----------
        store        : SegmentStore na de tijdstap
        decay_k      : bulkvervalconstanten [1/s]
        dt           : tijdstap [s]
        k_wall_vol   : wandvervalconstanten [1/s] (optioneel)
        exited_C     : concentraties van segmenten die het systeem verlieten
        exited_V     : volumes van die segmenten
        """
        n = store.n
        if n > 0:
            curr_mass = (store.C[:n] * store.volume[:n, np.newaxis]).sum(axis=0)
        else:
            curr_mass = np.zeros(self.n_species)

        if self._step == 0:
            self._prev_mass = curr_mass.copy()

        # Schat vervalverliezen via analytische verval
        mass_bulk = self._prev_mass * (1 - np.exp(-decay_k * dt))
        self.bulk_decay += mass_bulk

        if k_wall_vol is not None and n > 0:
            # Gebruik _prev_mass als basis voor de schatting (massa vóór de stap),
            # consistent met de bulk-schatting hierboven. De wandreactiesnelheid
            # varieert per leiding, dus we berekenen een gewogen gemiddelde k_wall
            # over alle actieve segmenten en passen dat toe op _prev_mass.
            k_w_seg = k_wall_vol[store.pipe[:n]]          # (n, n_species)
            vol_n   = store.volume[:n, np.newaxis]        # (n, 1)
            tot_vol = vol_n.sum()
            if tot_vol > 0:
                k_w_mean = (k_w_seg * vol_n).sum(axis=0) / tot_vol   # (n_species,)
                mass_wall = self._prev_mass * (1 - np.exp(-k_w_mean * dt))
                self.wall_decay += mass_wall

        if exited_C is not None and exited_V is not None and len(exited_V) > 0:
            self.outflow += (exited_C * exited_V[:, np.newaxis]).sum(axis=0)

        self._prev_mass = curr_mass
        self._step     += 1

    def current_mass(self, store) -> np.ndarray:
        """Huidige massa in het systeem [mass-eenheid]."""
        n = store.n
        if n == 0:
            return np.zeros(self.n_species)
        return (store.C[:n] * store.volume[:n, np.newaxis]).sum(axis=0)

    def report(self, store=None) -> dict:
        """
        Geeft een massabalansrapport terug.

        Returns
        -------
        dict met:
            injected      : massa ingespoten per stof
            in_system     : massa nog in segmenten (als store opgegeven)
            bulk_decay    : gecumuleerd bulkverlies
            wall_decay    : gecumuleerd wandverlies
            outflow       : massa via eindknopen verlaten
            balance_error : relatieve massabalansfout
            ok            : True als error < 1%
        """
        in_sys = self.current_mass(store) if store is not None else np.zeros(self.n_species)

        accounted = in_sys + self.bulk_decay + self.wall_decay + self.outflow
        total_in  = self.injected

        # Relatieve fout per stof
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_err = np.where(
                total_in > 0,
                np.abs(total_in - accounted) / total_in,
                0.0,
            )

        return {
            'injected':     self.injected.copy(),
            'in_system':    in_sys,
            'bulk_decay':   self.bulk_decay.copy(),
            'wall_decay':   self.wall_decay.copy(),
            'outflow':      self.outflow.copy(),
            'accounted':    accounted,
            'balance_error': rel_err,
            'ok':           bool((rel_err < 0.01).all()),
        }

    def reset(self) -> None:
        """Reset alle tellers."""
        self.injected[:]   = 0.0
        self.bulk_decay[:] = 0.0
        self.wall_decay[:] = 0.0
        self.outflow[:]    = 0.0
        self._prev_mass[:] = 0.0
        self._step         = 0
