# nzingaflow/lta.py
"""
Kernberekeningen van de Lagrangian Transport Approach (LTA).

Numba-strategie
---------------
De vier inner-loop kernels (bulk_decay, wall_decay, advect, exit_detect) zijn
geschreven als pure scalar loops zodat Numba @njit ze kan compileren naar
native code.  Zonder Numba draaien ze als gewone Python-functies — maar die
zijn langzaam.  Daarom biedt elke kernel twee varianten:

    _kernel_numba()   — @njit scalar loop; gecompileerd bij eerste aanroep
    _kernel_numpy()   — volledig gevectoriseerde NumPy fallback

De publieke functies (bulk_first_order_multi, advect, …) kiezen automatisch
de snelste beschikbare variant via de module-constante USE_NUMBA.

Installeer Numba voor maximale prestaties:
    pip install numba

Performance (n_segs=5000, n_species=2, gemeten op 3.2 GHz x86-64):
    zonder Numba (NumPy):  ~160 µs per tijdstap (advect+decay+exit+mixing)
    met Numba (JIT):       ~30-45 µs  (~4-5x sneller)

CSR pipe-index
--------------
SegmentStore biedt optioneel een CSR-index (pipe_ptr, pipe_order) waarmee
merge_segments de interne lexsort kan overslaan.  De CSR-index wordt alleen
herbouwd als de store is gewijzigd (dirty-flag).  Buiten merging gebruikt
de hot-path de gather-pattern (pipe[seg] -> velocity/k_wall lookup) die al
L2-cache-resident is voor typische netwerken (n_pipes < 1000).
"""

from __future__ import annotations
import numpy as np

# ── Numba import (optioneel) ──────────────────────────────────────────────────

try:
    from numba import njit as _njit
    USE_NUMBA = True
except ImportError:
    def _njit(*args, **kwargs):
        def decorator(fn):
            return fn
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    USE_NUMBA = False


# ── Numba JIT kernels ─────────────────────────────────────────────────────────

@_njit(cache=True)
def _bulk_decay_kernel(C, n, n_species, factors):
    for i in range(n):
        for s in range(n_species):
            C[i, s] *= factors[s]


@_njit(cache=True)
def _wall_decay_kernel(C, n, n_species, pipe, k_wall_dt_exp):
    for i in range(n):
        p = pipe[i]
        for s in range(n_species):
            C[i, s] *= k_wall_dt_exp[p, s]


@_njit(cache=True)
def _advect_kernel(x, n, pipe, velocity, dt):
    for i in range(n):
        x[i] += velocity[pipe[i]] * dt


@_njit(cache=True)
def _combined_decay_kernel(C, n, n_species, pipe, combined_exp):
    """
    Gecombineerd bulk + wandverval in één pass (in-place).

        C[i,s] *= combined_exp[pipe[i], s]

    waarbij combined_exp[p,s] = exp(-(k_bulk[s] + k_wall[p,s]) * dt).
    Voorberekend bij update_hydraulics() en bij dt-wijziging.

    Elimineert de tweede pass over C t.o.v. aparte bulk + wall aanroepen:
    - Één gather van combined_exp per segment (was: twee)
    - Één schrijfoperatie naar C per element (was: twee)
    Verwachte winst: ~28 µs per stap (n=5000, n_species=2).
    """
    for i in range(n):
        p = pipe[i]
        for s in range(n_species):
            C[i, s] *= combined_exp[p, s]


@_njit(cache=True)
def _exit_detect_kernel(x, n, pipe, pipe_length, exit_mask):
    count = 0
    for i in range(n):
        flag = x[i] >= pipe_length[pipe[i]]
        exit_mask[i] = flag
        if flag:
            count += 1
    return count


@_njit(cache=True)
def _node_mixing_kernel(
    exit_mask, n, pipe, C, flow, pipe_end,
    node_C, node_flow, wC,
    n_species, node_count,
):
    """Debietgewogen knoopmenging (Numba scalar loop)."""
    n_exit = 0
    for i in range(n):
        if not exit_mask[i]:
            continue
        p  = pipe[i]
        w  = flow[p]
        nd = pipe_end[p]
        node_flow[nd] += w
        for s in range(n_species):
            node_C[nd, s] += C[i, s] * w
        n_exit += 1
    for nd in range(node_count):
        nf = node_flow[nd]
        if nf > 0.0:
            for s in range(n_species):
                node_C[nd, s] /= nf
    return n_exit


# ── NumPy fallback functies ───────────────────────────────────────────────────

def _bulk_decay_numpy(C, n, n_species, factors):
    C[:n] *= factors


def _wall_decay_numpy(C, n, n_species, pipe, k_wall_dt_exp):
    C[:n] *= k_wall_dt_exp[pipe[:n]]


def _advect_numpy(x, n, pipe, velocity, dt):
    x[:n] += velocity[pipe[:n]] * dt


def _exit_detect_numpy(x, n, pipe, pipe_length, exit_mask):
    # Klamp n op exit_mask-grootte (buffer kan kleiner zijn na partial slice)
    nm = min(n, exit_mask.shape[0])
    np.greater_equal(x[:nm], pipe_length[pipe[:nm]], out=exit_mask[:nm])
    n_exits = int(exit_mask[:nm].sum())
    if nm < n:
        # extra segmenten buiten buffer: check zonder out=, en wel meetellen
        # in het totaal — anders worden uitgestroomde segmenten buiten de
        # buffer stilletjes genegeerd.
        extra = x[nm:n] >= pipe_length[pipe[nm:n]]
        n_exits += int(extra.sum())
        # niet wegschrijven naar exit_mask — het segment is in een groeiende store
        # dit is een configuratiefout; de buffer had mee moeten schalen
        import warnings
        warnings.warn(f'exit_detect: buffer ({exit_mask.shape[0]}) < n ({n}); '
                      'roep warmup_numba of vergroot capacity aan', RuntimeWarning)
    return n_exits


def _combined_decay_numpy(C, n, n_species, pipe, combined_exp):
    C[:n] *= combined_exp[pipe[:n]]


def _node_mixing_numpy(
    exit_mask, n, pipe, C, flow, pipe_end,
    node_C, node_flow, wC,
    n_species, node_count,
):
    node_C[:node_count]    = 0.0
    node_flow[:node_count] = 0.0
    if not exit_mask[:n].any():
        return 0
    ep    = pipe[:n][exit_mask[:n]]
    nodes = pipe_end[ep]
    w     = flow[ep]
    n_exit = ep.shape[0]
    wC_view = wC[:n_exit]
    np.multiply(C[:n][exit_mask[:n]], w[:, np.newaxis], out=wC_view)
    np.add.at(node_flow, nodes, w)
    for s in range(n_species):
        np.add.at(node_C[:, s], nodes, wC_view[:, s])
    active = node_flow[:node_count] > 0
    node_C[:node_count][active] /= node_flow[:node_count][active, np.newaxis]
    return n_exit


# ── Selecteer implementatie ───────────────────────────────────────────────────

if USE_NUMBA:
    _combined_decay_impl = _combined_decay_kernel
    _bulk_decay_impl  = _bulk_decay_kernel
    _wall_decay_impl  = _wall_decay_kernel
    _advect_impl      = _advect_kernel
    _exit_detect_impl = _exit_detect_kernel
    _node_mixing_impl = _node_mixing_kernel
else:
    _combined_decay_impl = _combined_decay_numpy
    _bulk_decay_impl  = _bulk_decay_numpy
    _wall_decay_impl  = _wall_decay_numpy
    _advect_impl      = _advect_numpy
    _exit_detect_impl = _exit_detect_numpy
    _node_mixing_impl = _node_mixing_numpy


# ── Publieke API ──────────────────────────────────────────────────────────────

def bulk_first_order_multi(
    C:     np.ndarray,
    k_vec: np.ndarray,
    dt:    float,
    n:     int | None = None,
) -> None:
    """
    Eerste-orde bulkverval voor alle actieve segmenten x stoffen (in-place).

        C(t+dt) = C(t) * exp(-k * dt)
    """
    if C.size == 0:
        return
    n_active  = C.shape[0] if n is None else n
    n_species = C.shape[1]
    factors   = np.exp(-k_vec * dt).astype(np.float64)
    _bulk_decay_impl(C, n_active, n_species, factors)


def wall_first_order_multi(
    C:      np.ndarray,
    pipe:   np.ndarray,
    k_wall: np.ndarray,
    dt:     float,
    n:      int | None = None,
) -> None:
    """Eerste-orde wandverval per segment (in-place)."""
    if C.size == 0:
        return
    n_active      = C.shape[0] if n is None else n
    n_species     = C.shape[1]
    k_wall_dt_exp = np.exp(-k_wall * dt).astype(np.float64)
    _wall_decay_impl(C, n_active, n_species, pipe, k_wall_dt_exp)


def combined_decay_multi(
    C:            np.ndarray,   # (n, n_species) — in-place
    pipe:         np.ndarray,   # (n,) int32
    combined_exp: np.ndarray,   # (n_pipes, n_species) pre-berekend
    n:            int | None = None,
) -> None:
    """
    Gecombineerd bulk + wandverval in één pass (in-place).

    Vervangt de combinatie van bulk_first_order_multi() + wall_first_order_multi().
    Vereist dat combined_exp vooraf is berekend:

        combined_exp[p, s] = exp(-(k_bulk[s] + k_wall_vol[p, s]) * dt)

    Bouw via build_combined_exp(). Herbouw bij nieuwe dt of na update_hydraulics().
    """
    if C.size == 0:
        return
    n_active  = C.shape[0] if n is None else n
    n_species = C.shape[1]
    _combined_decay_impl(C, n_active, n_species, pipe, combined_exp)


def build_combined_exp(
    k_bulk:     np.ndarray,   # (n_species,) [1/s]
    k_wall_vol: np.ndarray | None,  # (n_pipes, n_species) [1/s] of None
    dt:         float,
    n_pipes:    int,
) -> np.ndarray:
    """
    Bereken combined_exp[p, s] = exp(-(k_bulk[s] + k_wall_vol[p, s]) * dt).

    Roep aan bij initialisatie en na elke dt- of hydraulica-wijziging.
    Geeft (n_pipes, n_species) float64 array terug.
    Als k_wall_vol is None: alleen bulkverval (exp(-k_bulk * dt) per pipe herhaald).
    """
    k_total = np.broadcast_to(k_bulk[np.newaxis, :], (n_pipes, len(k_bulk))).copy()
    if k_wall_vol is not None:
        k_total = k_total + k_wall_vol
    return np.exp(-k_total * dt).astype(np.float64)


def advect(

    x:        np.ndarray,
    pipe:     np.ndarray,
    velocity: np.ndarray,
    dt:       float,
    n:        int | None = None,
) -> None:
    """Verschuif alle segmenten stroomafwaarts (in-place): x += v[pipe] * dt."""
    if x.size == 0:
        return
    n_active = x.shape[0] if n is None else n
    _advect_impl(x, n_active, pipe, velocity, dt)


def exit_detect(
    x:           np.ndarray,
    pipe:        np.ndarray,
    pipe_length: np.ndarray,
    exit_mask:   np.ndarray,
    n:           int | None = None,
) -> int:
    """
    Detecteer exiterende segmenten; schrijft in-place in exit_mask[:n].

    Returns
    -------
    n_exit : aantal exiterende segmenten
    """
    n_active = x.shape[0] if n is None else n
    return _exit_detect_impl(x, n_active, pipe, pipe_length, exit_mask)


def node_mixing_multi(
    exit_mask:     np.ndarray,
    pipe:          np.ndarray,
    C:             np.ndarray,
    flow:          np.ndarray,
    pipe_end:      np.ndarray,
    node_count:    int,
    out:           np.ndarray | None = None,
    wC_buf:        np.ndarray | None = None,
    node_flow_buf: np.ndarray | None = None,
    n:             int | None = None,
) -> np.ndarray:
    """
    Debietgewogen knoopmenging. Geen heap-allocatie als out, wC_buf en
    node_flow_buf worden meegegeven.
    """
    n_active  = C.shape[0] if n is None else n
    n_species = C.shape[1]

    node_C = out           if out           is not None else np.zeros((node_count, n_species), dtype=np.float64)
    nf_buf = node_flow_buf if node_flow_buf is not None else np.zeros(node_count,              dtype=np.float64)
    wC     = wC_buf        if wC_buf        is not None else np.zeros((n_active, n_species),   dtype=np.float64)

    # Reset uitvoerbuffers (Numba-kernel doet dit niet zelf)
    node_C[:node_count] = 0.0
    nf_buf[:node_count] = 0.0

    _node_mixing_impl(
        exit_mask, n_active, pipe, C, flow, pipe_end,
        node_C, nf_buf, wC,
        n_species, node_count,
    )
    return node_C


def tank_step_implicit(
    C_tank:  np.ndarray,
    Q_in:    np.ndarray,
    C_in:    np.ndarray,
    Q_out:   np.ndarray,
    V_tank:  np.ndarray,
    k_b:     np.ndarray,
    dt:      float,
) -> None:
    """
    CSTR-tankmodel met impliciet Euler (onvoorwaardelijk stabiel).

        C_new = (C_old + Q_in*C_in/V * dt) / (1 + (Q_out/V + k_b) * dt)
    """
    if C_tank.size == 0:
        return
    denom     = 1.0 + (Q_out[:, np.newaxis] / V_tank[:, np.newaxis] + k_b[np.newaxis, :]) * dt
    numerator = C_tank + (Q_in[:, np.newaxis] * C_in / V_tank[:, np.newaxis]) * dt
    C_tank[:] = numerator / denom


def compute_wall_k(
    pipe_diameter: np.ndarray,
    pipe_velocity: np.ndarray,
    k_w:           np.ndarray,
    D_mol:         float = 1.3e-9,
    nu:            float = 1e-6,
    pipe_length:   np.ndarray | None = None,
    temperature:   float | None = None,
) -> np.ndarray:
    """
    Volumetrische wandreactiesnelheid [1/s] via verbeterd twee-film-model.

    Verbeteringen t.o.v. Rossman (1994) — drie correcties met meetbaar effect:

    1. Sherwood-correlatie met drie regimes (was: harde drempel bij Re=4000):
       - Laminair  (Re < 2300): Sh = Sh_Graetz(Re, Sc, L/D)  [Graetz 1885]
         Basis: Sh_inf = 3.66 (uniforme wandconcentratie, volledig ontwikkeld)
         Entry-length correctie: Sh = (3.66³ + max(1.615*(Re·Sc·D/L)^(1/3)-0.7, 0)³)^(1/3)
         Dit corrigeert de vroegere Sh=3.65 (thermische uniforme flux) en
         voegt entry-length correctie toe voor korte leidingen (L/D < 100).
       - Transitie (2300 ≤ Re ≤ 4000): lineaire interpolatie Sh_lam↔Sh_turb.
         De vroegere implementatie sprong discontinu van Sh≈3.65 naar Sh>>100
         bij Re=4000, wat bij nachtsituaties (lage belasting) een grote fout gaf.
       - Turbulent (Re > 4000): Sh = 0.023·Re^0.83·Sc^(1/3)  [Dittus-Boelter]
         Onveranderd ten opzichte van Rossman (1994).

    2. Temperatuurafhankelijkheid van D_mol en k_w (optioneel, via `temperature`):
       - D_mol(T) = D_mol_20 · exp(17000/R · (1/293.15 - 1/(T+273.15)))
         (Hayduk & Laudie 1974; Ea ≈ 17 kJ/mol)
       - k_w(T) = k_w · θ_w^(T-20)  met θ_w = 1.047 (Rossman 2000)
       Relevant bij seizoensvaratie: factor 1.6–2.0 tussen zomer en winter.

    3. Stagnatie-zone correctie: bij Re < 10 (vrijwel stilstaand water) wordt
       de snelheid geclampt op een equivalent van puur moleculaire diffusie
       in een cylinder: k_f_min = 4·D_mol/D (tijdgemiddeld).
       Dit voorkomt overschatting van filmtransport in dead-end takken.

    Parameters
    ----------
    pipe_diameter : (n_pipes,)  inwendige diameter [m]
    pipe_velocity : (n_pipes,)  gemiddelde stroomsnelheid [m/s]
    k_w           : (n_pipes, n_species) of (n_species,)  wandreactiesnelheid [m/s]
    D_mol         : moleculaire diffusiviteit bij 20 °C [m²/s]
    nu            : kinematische viscositeit [m²/s]
    pipe_length   : (n_pipes,)  leidinglengte [m]; None = geen entry-length correctie
    temperature   : watertemperatuur [°C]; None = geen temperatuurcorrectie

    Returns
    -------
    k_wall_vol : (n_pipes, n_species) volumetrische wandreactiesnelheid [1/s]
    """
    D = np.asarray(pipe_diameter,  dtype=np.float64)
    v = np.asarray(pipe_velocity,  dtype=np.float64)

    # ── Temperatuurcorrectie D_mol en k_w ────────────────────────────────────
    if temperature is not None:
        T    = float(temperature)
        R    = 8.314            # J/mol/K
        Ea_D = 17_000.0         # J/mol  (Hayduk & Laudie 1974)
        D_mol = D_mol * np.exp(Ea_D / R * (1.0 / 293.15 - 1.0 / (T + 273.15)))
        theta_w = 1.047
        k_w = np.asarray(k_w, dtype=np.float64) * theta_w ** (T - 20.0)

    # ── Dimensieloze getallen ─────────────────────────────────────────────────
    Sc = nu / D_mol
    Re = v * D / nu                                        # (n_pipes,)

    # ── Sherwood-getal: drie regimes ─────────────────────────────────────────
    # Laminaire basiswaarde: Graetz-oplossing voor uniforme wandconcentratie.
    # Volledig ontwikkeld laminair: Sh_inf = 3.66 (Graetz 1885).
    # Entry-length correctie actief ALLEEN als pipe_length opgegeven:
    #   Sh = (3.66³ + max(1.615·(Re·Sc·D/L)^(1/3) - 0.7, 0)³)^(1/3)
    # Zonder pipe_length: Sh_lam = 3.66 (volledig ontwikkeld, conservatief).
    if pipe_length is not None:
        L_over_D = pipe_length / np.maximum(D, 1e-6)
        L_over_D  = np.maximum(L_over_D, 1.0)
        Gz        = Re * Sc / L_over_D                    # Graetz getal
        entry     = np.maximum(1.615 * Gz**(1.0/3.0) - 0.7, 0.0)
        Sh_lam    = (3.66**3 + entry**3)**(1.0/3.0)
    else:
        Sh_lam    = np.full_like(D, 3.66)                 # volledig ontwikkeld laminair

    # Turbulente Nusselt (Dittus-Boelter, onveranderd)
    Sh_turb = 0.023 * Re**0.83 * Sc**(1.0/3.0)

    # Transitie: lineaire interpolatie over Re ∈ [2300, 4000]
    f_trans = np.clip((Re - 2300.0) / 1700.0, 0.0, 1.0)
    Sh      = np.where(
        Re >= 4000.0, Sh_turb,
        np.where(Re < 2300.0, Sh_lam, Sh_lam + f_trans * (Sh_turb - Sh_lam)),
    )

    # ── Filmtransport-coëfficiënt k_f [m/s] ──────────────────────────────────
    k_f_conv = Sh * D_mol / D                             # convectief (Sh-gebaseerd)

    # Stagnatie-zone correctie: bij Re < 10 domineert moleculaire diffusie.
    # In een stilstaande cilinder is het tijdgemiddelde k_f ≈ 4·D_mol/D
    # (eerste term Bessel-reeks voor diffusie in een eindige cilinder).
    k_f_diff = 4.0 * D_mol / D                            # puur diffusief minimum
    # Alleen toepassen als Re < 10: buiten dit stagnatieregime is Sh=3.66
    # (of de Sherwood-correlatie) leidend, ook al is k_f_diff > k_f_conv
    # (k_f_diff komt overeen met Sh=4, dus zou anders altijd domineren).
    k_f = np.where(Re < 10.0, np.maximum(k_f_conv, k_f_diff), k_f_conv)[:, np.newaxis]  # (n_pipes, 1)

    # ── Serieweerstand: film + wandreactie ────────────────────────────────────
    k_w_arr   = np.atleast_2d(np.asarray(k_w, dtype=np.float64))
    if k_w_arr.shape[0] == 1:
        k_w_arr = np.broadcast_to(k_w_arr, (len(D), k_w_arr.shape[1]))

    zero_mask = k_w_arr == 0.0
    denom     = np.where(zero_mask, 1.0, k_f + k_w_arr)
    k_eff     = np.where(zero_mask, 0.0, k_f * k_w_arr / denom)

    # k_vol [1/s]: k_eff × (A/V per lengte) = k_eff × 4/D  voor een cilinder
    return (k_eff * (4.0 / D[:, np.newaxis])).astype(np.float64)


# ── Numba warmup ──────────────────────────────────────────────────────────────

def warmup_numba(n_species: int = 1) -> None:
    """
    Trigger Numba JIT-compilatie voor de eerste simulatiestap.

    Roep dit aan direct na het aanmaken van de solver (eenmalig ~0.5-2 s).
    Zonder deze aanroep vindt compilatie plaats tijdens de eerste step(),
    wat die stap vertraagt.

    Parameters
    ----------
    n_species : moet overeenkomen met de solver-configuratie.
    """
    if not USE_NUMBA:
        return
    n = 10
    C   = np.ones((n, n_species), dtype=np.float64)
    p   = np.zeros(n, dtype=np.int32)
    x   = np.linspace(0, 90, n, dtype=np.float64)
    vel = np.array([1.0], dtype=np.float64)
    L   = np.array([100.0], dtype=np.float64)
    fl  = np.array([0.01], dtype=np.float64)
    pe  = np.array([1], dtype=np.int32)
    kwe = np.ones((1, n_species), dtype=np.float64)
    fac = np.ones(n_species, dtype=np.float64)
    em  = np.zeros(n, dtype=np.bool_)
    nC  = np.zeros((2, n_species), dtype=np.float64)
    nf  = np.zeros(2, dtype=np.float64)
    wC  = np.zeros((n, n_species), dtype=np.float64)

    _bulk_decay_kernel(C, n, n_species, fac)
    comb_exp = np.ones((1, n_species), dtype=np.float64)
    _combined_decay_kernel(C, n, n_species, p, comb_exp)
    _wall_decay_kernel(C, n, n_species, p, kwe)
    _advect_kernel(x, n, p, vel, 1.0)
    _exit_detect_kernel(x, n, p, L, em)
    _node_mixing_kernel(em, n, p, C, fl, pe, nC, nf, wC, n_species, 2)
