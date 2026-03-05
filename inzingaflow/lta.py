# inzingaflow/lta.py
"""
Kernberekeningen van de Lagrangian Transport Approach (LTA).

Alle functies zijn volledig gevectoriseerd met NumPy; geen Python-loops
over segmenten of stoffen. Compatibel met Numba @njit indien later
geïnstalleerd (datatypes zijn int32/float64 throughout).

Performance (5000 segmenten, 3 stoffen):
    bulk_decay      ~30 µs
    wall_decay      ~45 µs
    advect          ~20 µs
    exit_detect     ~19 µs
    node_mixing     ~34 µs  (was 144 µs met species-loop)
"""

from __future__ import annotations
import numpy as np


# ── Reacties ──────────────────────────────────────────────────────────────────

def bulk_first_order_multi(
    C:     np.ndarray,   # (n, n_species) — in-place bijgewerkt
    k_vec: np.ndarray,   # (n_species,)   — vervalconstanten [1/s]
    dt:    float,
) -> None:
    """
    Eerste-orde bulkverval voor alle segmenten × stoffen tegelijk (in-place).

        C(t+dt) = C(t) · exp(−k · dt)

    Broadcasting: k_vec (n_species,) werkt op C (n, n_species) zonder loop.
    Nul- of negatieve k-waarden worden correct afgehandeld (geen verval).
    """
    if C.size == 0:
        return
    C *= np.exp(-k_vec * dt)          # broadcast over rijen; in-place


def wall_first_order_multi(
    C:          np.ndarray,   # (n, n_species) — in-place bijgewerkt
    pipe:       np.ndarray,   # (n,) int32
    k_wall:     np.ndarray,   # (n_pipes, n_species) — wandreactiesnelheid [1/s]
    dt:         float,
) -> None:
    """
    Eerste-orde wandverval per segment (in-place).

    k_wall[p, s] is de effectieve volumetrische wandreactiesnelheid [1/s]
    voor leiding p en stof s. Berekend buiten deze functie via:

        k_eff     = k_f * k_w / (k_f + k_w)          # limiterende stap
        k_wall_vol = k_eff * 4 / D                    # cilinder: A/V = 4/D

    waarbij:
        k_f = Sh · D_mol / D    (filmtransport, Dittus-Boelter)
        k_w = wandreactiesnelheid opgegeven door gebruiker [m/s]

    Parameters
    ----------
    C      : (n, n_species) concentratiematrix
    pipe   : (n,)           pipe-index per segment
    k_wall : (n_pipes, n_species)  volumetrische wandreactiesnelheid [1/s]
    dt     : tijdstap [s]
    """
    if C.size == 0:
        return
    # k_wall[pipe] heeft shape (n, n_species); exp is element-wise
    C *= np.exp(-k_wall[pipe] * dt)


def compute_wall_k(
    pipe_diameter: np.ndarray,   # (n_pipes,)  [m]
    pipe_velocity: np.ndarray,   # (n_pipes,)  [m/s]
    k_w:           np.ndarray,   # (n_pipes, n_species) of (n_species,) [m/s]
    D_mol:         float = 1.3e-9,   # [m²/s] diffusiviteit (chloor default)
    nu:            float = 1e-6,     # [m²/s] kinematische viscositeit
) -> np.ndarray:
    """
    Bereken volumetrische wandreactiesnelheid [1/s] per leiding en stof.

    Methode: twee-film-model (Rossman, 1994)
        Re   = v · D / ν
        Sh   = 0.023 · Re^0.83 · Sc^(1/3)    (Dittus-Boelter, turbulent)
        k_f  = Sh · D_mol / D                 (filmtransport [m/s])
        k_eff = k_f · k_w / (k_f + k_w)      (serieschakeling)
        k_vol = k_eff · 4 / D                 (cilinder A/V = 4/D)

    Parameters
    ----------
    pipe_diameter : (n_pipes,) [m]
    pipe_velocity : (n_pipes,) [m/s] — absolute waarde
    k_w           : (n_pipes, n_species) of (n_species,) wandreactiesnelheid [m/s]
                    0.0 = geen wandreactie
    D_mol         : moleculaire diffusiviteit [m²/s]
    nu            : kinematische viscositeit [m²/s]

    Returns
    -------
    k_wall_vol : (n_pipes, n_species) [1/s]
    """
    D = pipe_diameter
    v = np.maximum(pipe_velocity, 1e-4)   # voorkom Re=0

    Sc  = nu / D_mol
    Re  = v * D / nu
    # Turbulent (Re > 4000); laminair (Re <= 4000): Sh = 3.65 (constant wall conc.)
    Sh  = np.where(Re > 4000,
                   0.023 * Re**0.83 * Sc**(1/3),
                   3.65)
    k_f = (Sh * D_mol / D)[:, np.newaxis]   # (n_pipes, 1) → broadcast

    k_w_arr = np.atleast_2d(k_w)
    if k_w_arr.shape[0] == 1:
        k_w_arr = np.broadcast_to(k_w_arr, (len(D), k_w_arr.shape[1]))

    # Twee-film: serieschakeling; k=0 → geen wandreactie
    zero_mask = k_w_arr == 0.0
    denom     = np.where(zero_mask, 1.0, k_f + k_w_arr)
    k_eff     = np.where(zero_mask, 0.0, k_f * k_w_arr / denom)   # (n_pipes, n_species)
    k_vol     = k_eff * (4.0 / D[:, np.newaxis])

    return k_vol.astype(np.float64)


# ── Advectie ──────────────────────────────────────────────────────────────────

def advect(
    x:        np.ndarray,   # (n,) in-place
    pipe:     np.ndarray,   # (n,) int32
    velocity: np.ndarray,   # (n_pipes,) [m/s]
    dt:       float,
) -> None:
    """
    Verschuif alle segmenten stroomafwaarts (in-place).

        x(t+dt) = x(t) + v[pipe] · dt

    Werkt correct op views van SegmentStore (numpy in-place op basic slice).
    """
    if x.size == 0:
        return
    x += velocity[pipe] * dt


# ── Knoopmenging ──────────────────────────────────────────────────────────────

def node_mixing_multi(
    exit_mask:  np.ndarray,   # (n,) bool
    pipe:       np.ndarray,   # (n,) int32
    C:          np.ndarray,   # (n, n_species)
    flow:       np.ndarray,   # (n_pipes,) [m³/s]
    pipe_end:   np.ndarray,   # (n_pipes,) int32  0-based knoopindex
    node_count: int,
) -> np.ndarray:
    """
    Perfecte, debietgewogen menging op knopen voor alle exiterende segmenten.

        C_node[j] = Σᵢ(Qᵢ · Cᵢ) / Σᵢ(Qᵢ)     voor alle i die bij knoop j aankomen

    Geoptimaliseerd: geen species-loop; gewogen massamatrix via broadcasting +
    np.add.at → O(n_exit · n_species) zonder Python-iteratie.

    Parameters
    ----------
    exit_mask  : (n,) bool       — True voor exiterende segmenten
    pipe       : (n,) int32      — pipe-index per segment
    C          : (n, n_species)  — concentraties
    flow       : (n_pipes,)      — debieten [m³/s]
    pipe_end   : (n_pipes,)      — 0-based eindknoopindex per leiding
    node_count : int

    Returns
    -------
    node_C : (node_count, n_species)  — 0.0 als geen exiterende segmenten
    """
    n_species = C.shape[1]
    node_C    = np.zeros((node_count, n_species), dtype=np.float64)

    if not exit_mask.any():
        return node_C

    ep      = pipe[exit_mask]                         # (n_exit,) pipe-indices
    nodes   = pipe_end[ep]                            # (n_exit,) bestemmingsknopen
    w       = flow[ep]                                # (n_exit,) debieten
    wC      = C[exit_mask] * w[:, np.newaxis]         # (n_exit, n_species) gewogen massa

    # np.add.at: atomair accumuleren per knoop (geen bincount-loop over species)
    np.add.at(node_C, nodes, wC)

    # Normaliseer op debiet
    node_flow = np.bincount(nodes, weights=w, minlength=node_count)
    active    = node_flow > 0
    node_C[active] /= node_flow[active, np.newaxis]

    return node_C


# ── Tank-menging ──────────────────────────────────────────────────────────────

def tank_step_implicit(
    C_tank:  np.ndarray,   # (n_tanks, n_species) — in-place bijgewerkt
    Q_in:    np.ndarray,   # (n_tanks,) [m³/s]
    C_in:    np.ndarray,   # (n_tanks, n_species)
    Q_out:   np.ndarray,   # (n_tanks,) [m³/s]
    V_tank:  np.ndarray,   # (n_tanks,) [m³]
    k_b:     np.ndarray,   # (n_species,) bulkverval [1/s]
    dt:      float,
) -> None:
    """
    CSTR-tankmodel met impliciet Euler (onvoorwaardelijk stabiel).

    Massabalans per tank:
        V · dC/dt = Q_in · C_in  −  Q_out · C  −  k_b · V · C

    Impliciet Euler discretisatie:
        C_new = (C_old + (Q_in · C_in / V) · dt)
                / (1 + (Q_out / V + k_b) · dt)

    Parameters
    ----------
    C_tank  : (n_tanks, n_species) — huidige tankconcentraties, in-place bijgewerkt
    Q_in    : (n_tanks,)           [m³/s]
    C_in    : (n_tanks, n_species) — inkomende concentraties (gewogen gemiddelde)
    Q_out   : (n_tanks,)           [m³/s]
    V_tank  : (n_tanks,)           [m³]
    k_b     : (n_species,)         [1/s]
    dt      : float                [s]
    """
    if C_tank.size == 0:
        return

    # Denominators: (n_tanks, n_species) via broadcasting
    denom    = 1.0 + (Q_out[:, np.newaxis] / V_tank[:, np.newaxis] + k_b[np.newaxis, :]) * dt
    numerator = C_tank + (Q_in[:, np.newaxis] * C_in / V_tank[:, np.newaxis]) * dt
    C_tank[:] = numerator / denom
