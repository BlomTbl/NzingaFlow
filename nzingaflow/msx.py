# nzingaflow/msx.py
"""
MsxReactionSystem — MSX-compatibele multi-species reactielaag voor NzingaFlow.

Implementeert de vier kernconcepten van EPANET-MSX 2.0:

    A. Willekeurig DAE-reactiesysteem
       RATE  : dC/dt = f(C_bulk, C_wall, params)   [kinetisch]
       EQUIL : 0 = g(C_bulk, C_wall, params)        [evenwicht]
       FORMULA: C = h(C_bulk, C_wall, params)        [afgeleide variabele]

    B. Oppervlaktesoorten (WALL species)
       Gebonden aan de leidingwand; bewegen NIET mee met het water.
       Koppelen aan bulksoorten via RATE-expressies die Av (= 4/D [1/m])
       als variabele gebruiken.

    C. Numerieke ODE-integratie
       solver='euler'    — voorwaarts Euler (snel, alleen niet-stijf)
       solver='rk4'      — klassieke Runge-Kutta 4e orde (vectorized)
       solver='rk45'     — adaptief RK45 via scipy (niet-stijf)
       solver='radau'    — Radau IIA via scipy (stijf: chloramine, biofilm)

    D. Expressie-taal (optioneel)
       Vergelijkingen kunnen als Python-string opgegeven worden; ze worden
       gecompileerd via sympy.lambdify naar snelle numpy-functies.
       Direct Python-callables zijn ook toegestaan en sneller.

Gebruik
-------
    from nzingaflow.msx import MsxReactionSystem

    rxn = MsxReactionSystem(
        bulk_species  = ['Cl2', 'NH3', 'NH2Cl'],
        wall_species  = ['BF'],                      # biofilm
        params        = {'k1': 1.5e-4, 'k2': 3e-3, 'k4': 0.01, 'k5': 0.005,
                         'BFmax': 100.0},
        pipe_rates = {
            'Cl2':   '-k1 * Cl2 * NH3 - k2 * Cl2 * BF * Av',
            'NH3':   '-k1 * Cl2 * NH3',
            'NH2Cl': 'k1 * Cl2 * NH3 - k3 * NH2Cl',
            'BF':    'k4 * NH2Cl * (BFmax - BF) - k5 * BF',
        },
        tank_rates = {
            'Cl2':   '-k1 * Cl2 * NH3',
            'NH3':   '-k1 * Cl2 * NH3',
            'NH2Cl': 'k1 * Cl2 * NH3 - k3 * NH2Cl',
        },
        solver = 'rk4',
    )

    solver = NzingaFlowSolver(
        'netwerk.inp',
        n_species = len(rxn.bulk_species),
        geochem   = rxn,          # plug in op de geochem-interface
    )

Notatie in expressie-strings
------------------------------
  Bulksoorten     : exacte naam (bijv. 'Cl2', 'NH2Cl')
  Oppervlaksoorten: exacte naam (bijv. 'BF', 'AS5s')
  Parameters      : exacte naam (bijv. 'k1', 'Smax')
  Gereserveerd    : 'Av'  = leidingoppervlak per volume [m²/m³] = 4/D
                    't'   = gesimuleerde tijd [s]
Eenheden
---------
  Dezelfde als de NzingaFlow-invoer. Geen automatische eenheidsconversie.
  RATE-expressies moeten [concentratie/s] teruggeven.
"""

from __future__ import annotations

import warnings
import numpy as np
from typing import Callable, Dict, List, Optional, Union

_SOLVERS = ('euler', 'rk4', 'rk45', 'radau')


class MsxReactionSystem:
    """
    Multi-species reactielaag compatibel met EPANET-MSX concepten.

    Plugt in op de NzingaFlowSolver via de `geochem`-parameter:
    de solver roept `apply_geochemistry()` en `apply_mixing()` aan
    op dezelfde manier als GeochemSolver.

    Parameters
    ----------
    bulk_species : list[str]
        Namen van de bulksoorten in dezelfde volgorde als de C-matrix
        van de SegmentStore. Lengte moet overeenkomen met n_species.
    wall_species : list[str]
        Namen van de oppervlaktesoorten. Lege lijst = geen wandsoorten.
    params : dict[str, float]
        Reactieparameters (constanten). Kunnen ook per leiding ingesteld
        worden via `set_pipe_param()`.
    pipe_rates : dict[str, str | callable]
        RATE-expressies voor leidingen. Sleutel = soort; waarde = string
        (wordt gecompileerd) of callable(state_dict) → float.
    pipe_equil : dict[str, str | callable]
        EQUIL-expressies voor leidingen (evenwichtssoorten).
        De expressie moet gelijk zijn aan 0; Newton-iteratie lost op.
    pipe_formulas : dict[str, str | callable]
        FORMULA-expressies: directe berekening (geen ODE/evenwicht).
    tank_rates : dict[str, str | callable]
        RATE-expressies voor tanks (geen wandsoorten in tanks).
    tank_formulas : dict[str, str | callable]
        FORMULA-expressies voor tanks.
    solver : str
        ODE-integratiemethode: 'euler', 'rk4', 'rk45', 'radau'.
        Gebruik 'rk4' voor niet-stijve systemen (standaard).
        Gebruik 'radau' voor stijve systemen (chloramine, biofilm).
    rtol, atol : float
        Toleranties voor adaptieve solvers (rk45, radau).
    """

    def __init__(
        self,
        bulk_species:   List[str],
        wall_species:   List[str] = (),
        params:         Dict[str, float] = None,
        pipe_rates:     Dict[str, Union[str, Callable]] = None,
        pipe_equil:     Dict[str, Union[str, Callable]] = None,
        pipe_formulas:  Dict[str, Union[str, Callable]] = None,
        tank_rates:     Dict[str, Union[str, Callable]] = None,
        tank_formulas:  Dict[str, Union[str, Callable]] = None,
        solver:         str = 'rk4',
        rtol:           float = 1e-3,
        atol:           float = 1e-6,
    ):
        if solver not in _SOLVERS:
            raise ValueError(f"solver moet één van {_SOLVERS} zijn, niet {solver!r}")

        self.bulk_species  = list(bulk_species)
        self.wall_species  = list(wall_species)
        self.n_bulk        = len(bulk_species)
        self.n_wall        = len(wall_species)
        self.params        = dict(params or {})
        self.solver        = solver
        self.rtol          = rtol
        self.atol          = atol

        # Per-leiding parameteroverrides: {pipe_idx: {param_name: value}}
        self._pipe_params: Dict[int, Dict[str, float]] = {}

        # C_wall: (n_pipes, n_wall_species)  — geïnitialiseerd bij eerste aanroep
        self._C_wall: Optional[np.ndarray] = None
        self._n_pipes: int = 0

        # Compileer expressies
        self._pipe_rate_fns   = self._compile_all(pipe_rates   or {}, 'pipe')
        self._pipe_equil_fns  = self._compile_all(pipe_equil   or {}, 'pipe')
        self._pipe_formula_fns= self._compile_all(pipe_formulas or {}, 'pipe')
        self._tank_rate_fns   = self._compile_all(tank_rates   or {}, 'tank')
        self._tank_formula_fns= self._compile_all(tank_formulas or {}, 'tank')

        # Index: soort → positie in bulk of wall array
        self._bulk_idx = {s: i for i, s in enumerate(bulk_species)}
        self._wall_idx = {s: i for i, s in enumerate(wall_species)}

        # Soorten met RATE-expressies (worden geïntegreerd)
        self._bulk_rate_names = [s for s in bulk_species  if s in self._pipe_rate_fns]
        self._wall_rate_names = [s for s in wall_species  if s in self._pipe_rate_fns]

    # ── Expressie-compilatie ─────────────────────────────────────────────────

    def _compile_all(
        self,
        exprs: Dict[str, Union[str, Callable]],
        context: str,
    ) -> Dict[str, Callable]:
        compiled = {}
        for name, expr in exprs.items():
            if callable(expr):
                compiled[name] = expr
            elif isinstance(expr, str):
                compiled[name] = self._compile_str(expr, name, context)
            else:
                raise TypeError(f"Expressie voor {name!r} moet str of callable zijn")
        return compiled

    def _compile_str(self, expr: str, species: str, context: str) -> Callable:
        """
        Compileer een expressie-string naar een Python-functie via sympy.

        De functie ontvangt een state-dict en retourneert een float.
        Alle soorten, parameters en 'Av' zijn beschikbaar als symbolen.
        """
        try:
            import sympy as _sp

            # Alle symbolische variabelen
            sym_names = (
                self.bulk_species + self.wall_species
                + list(self.params.keys())
                + ['Av', 't']
            )
            syms = {n: _sp.Symbol(n) for n in sym_names}

            parsed = _sp.sympify(expr, locals=syms)
            free   = {str(s) for s in parsed.free_symbols}

            # lambdify → snelle numpy-functie
            sym_list = [syms[n] for n in sym_names if n in free]
            lam = _sp.lambdify(sym_list, parsed, modules='numpy')
            free_names = [n for n in sym_names if n in free]

            def fn(state: dict) -> float:
                args = [state[n] for n in free_names]
                return float(lam(*args))

            fn.__name__ = f"msx_{context}_{species}"
            return fn

        except ImportError:
            # Fallback: eval() zonder sympy
            warnings.warn(
                "sympy niet beschikbaar; gebruik eval() voor expressies. "
                "Installeer sympy voor veiligere compilatie.",
                ImportWarning, stacklevel=3
            )
            code = compile(expr, f"<msx {species}>", 'eval')

            def fn_eval(state: dict) -> float:
                return float(eval(code, {"__builtins__": {}}, state))

            return fn_eval

    # ── Initialisatie wandsoorten ─────────────────────────────────────────────

    def _ensure_wall(self, n_pipes: int, n_wall: int) -> None:
        if self._C_wall is None or self._n_pipes != n_pipes:
            self._C_wall  = np.zeros((n_pipes, n_wall), dtype=np.float64)
            self._n_pipes = n_pipes

    # ── Per-leiding parameteroverride ─────────────────────────────────────────

    def set_pipe_param(self, pipe_idx: int, **kwargs: float) -> None:
        """
        Stel een of meer parameters in voor een specifieke leiding.

        Overeenkomst met MSX [PARAMETERS]-sectie.

        Voorbeeld
        ---------
            rxn.set_pipe_param(5, k_wall=2e-6)   # gietijzer leiding
            rxn.set_pipe_param(12, k_wall=5e-7)  # PVC leiding
        """
        self._pipe_params.setdefault(pipe_idx, {}).update(kwargs)

    def get_wall_concentrations(self) -> Optional[np.ndarray]:
        """Retourneer huidige wandconcentraties als (n_pipes, n_wall_species) array."""
        return self._C_wall.copy() if self._C_wall is not None else None

    def set_wall_concentrations(self, C_wall: np.ndarray) -> None:
        """Stel beginconcentraties in voor wandsoorten (n_pipes, n_wall_species)."""
        self._C_wall = np.asarray(C_wall, dtype=np.float64).copy()
        self._n_pipes = C_wall.shape[0]

    # ── State-dict bouwen ─────────────────────────────────────────────────────

    def _make_state(
        self,
        C_bulk: np.ndarray,   # (n_bulk,)
        C_wall: np.ndarray,   # (n_wall,)
        Av:     float,
        t:      float = 0.0,
        pipe_idx: Optional[int] = None,
    ) -> dict:
        """Bouw een toestandsdict voor expressie-evaluatie."""
        state = {s: float(C_bulk[i]) for i, s in enumerate(self.bulk_species)}
        state.update({s: float(C_wall[i]) for i, s in enumerate(self.wall_species)})
        state.update(self.params)
        if pipe_idx is not None and pipe_idx in self._pipe_params:
            state.update(self._pipe_params[pipe_idx])
        state['Av'] = float(Av)
        state['t']  = float(t)
        return state

    # ── ODE rechterhand ───────────────────────────────────────────────────────

    def _rhs_pipe(
        self,
        y:      np.ndarray,   # (n_bulk + n_wall,)
        Av:     float,
        t:      float,
        pipe_idx: Optional[int],
        rate_fns: Dict[str, Callable],
    ) -> np.ndarray:
        """
        Rechterhands-vector voor leidingsegment.
        y = [C_bulk..., C_wall...]
        """
        C_bulk = y[:self.n_bulk]
        C_wall = y[self.n_bulk:]
        state  = self._make_state(C_bulk, C_wall, Av, t, pipe_idx)

        dy = np.zeros_like(y)
        for name, fn in rate_fns.items():
            val = fn(state)
            if name in self._bulk_idx:
                dy[self._bulk_idx[name]] += val
            elif name in self._wall_idx:
                dy[self.n_bulk + self._wall_idx[name]] += val
        return dy

    def _rhs_tank(
        self,
        y:    np.ndarray,
        t:    float,
        rate_fns: Dict[str, Callable],
    ) -> np.ndarray:
        C_bulk = y[:self.n_bulk]
        C_wall = np.zeros(self.n_wall)
        state  = self._make_state(C_bulk, C_wall, Av=0.0, t=t)

        dy = np.zeros_like(y)
        for name, fn in rate_fns.items():
            if name in self._bulk_idx:
                dy[self._bulk_idx[name]] += fn(state)
        return dy

    # ── Integratie-stap ───────────────────────────────────────────────────────

    def _integrate(
        self,
        y0:  np.ndarray,
        dt:  float,
        rhs: Callable,   # signatuur: rhs(t: float, y: ndarray) -> ndarray
    ) -> np.ndarray:
        """
        Integreer van t=0 naar t=dt met gekozen methode.
        rhs-conventie: rhs(t, y) — overeenkomstig scipy.integrate.solve_ivp.
        """
        y0 = np.asarray(y0, dtype=np.float64)

        if self.solver == 'euler':
            return np.maximum(y0 + dt * rhs(0.0, y0), 0.0)

        elif self.solver == 'rk4':
            k1 = rhs(0.0,       y0)
            k2 = rhs(0.5*dt,    np.maximum(y0 + 0.5*dt*k1, 0.0))
            k3 = rhs(0.5*dt,    np.maximum(y0 + 0.5*dt*k2, 0.0))
            k4 = rhs(dt,        np.maximum(y0 +     dt*k3,  0.0))
            return np.maximum(y0 + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4), 0.0)

        else:
            from scipy.integrate import solve_ivp
            method = 'RK45' if self.solver == 'rk45' else 'Radau'
            sol = solve_ivp(
                rhs, [0.0, dt], y0,
                method=method,
                rtol=self.rtol, atol=self.atol,
                dense_output=False,
            )
            if not sol.success:
                warnings.warn(
                    f"ODE-integratie niet geconvergeerd: {sol.message}",
                    RuntimeWarning, stacklevel=3,
                )
            return np.maximum(sol.y[:, -1], 0.0)

    # ── Evenwichtsoplossing (Newton) ──────────────────────────────────────────

    def _solve_equil(
        self,
        C_bulk: np.ndarray,
        C_wall: np.ndarray,
        Av:     float,
        pipe_idx: Optional[int],
        max_iter: int = 20,
        tol:      float = 1e-8,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Newton-iteratie voor evenwichtssoorten.
        Elke EQUIL-expressie f(C) = 0; los op voor de betreffende soort.
        Eén-dimensionaal Newton (secant-methode per soort).
        """
        C_b = C_bulk.copy()
        C_w = C_wall.copy()

        for _ in range(max_iter):
            converged = True
            for name, fn in self._pipe_equil_fns.items():
                state = self._make_state(C_b, C_w, Av, pipe_idx=pipe_idx)
                f0 = fn(state)
                if abs(f0) < tol:
                    continue
                converged = False

                # Secant stap: perturbeer de variabele
                if name in self._bulk_idx:
                    idx = self._bulk_idx[name]
                    dx  = max(abs(C_b[idx]) * 1e-6, 1e-12)
                    C_b[idx] += dx
                    state2 = self._make_state(C_b, C_w, Av, pipe_idx=pipe_idx)
                    f1 = fn(state2)
                    dfdx = (f1 - f0) / dx if abs(f1 - f0) > 1e-20 else 1.0
                    C_b[idx] -= dx + f0 / dfdx
                    C_b[idx]  = max(C_b[idx], 0.0)   # concentratie ≥ 0
                elif name in self._wall_idx:
                    idx = self._wall_idx[name]
                    dx  = max(abs(C_w[idx]) * 1e-6, 1e-12)
                    C_w[idx] += dx
                    state2 = self._make_state(C_b, C_w, Av, pipe_idx=pipe_idx)
                    f1 = fn(state2)
                    dfdx = (f1 - f0) / dx if abs(f1 - f0) > 1e-20 else 1.0
                    C_w[idx] -= dx + f0 / dfdx
                    C_w[idx]  = max(C_w[idx], 0.0)

            if converged:
                break

        return C_b, C_w

    # ── Hoofd-API: apply_geochemistry ─────────────────────────────────────────

    def apply_geochemistry(
        self,
        store,
        dt:        float,
        pipe_diam: Optional[np.ndarray] = None,
        pipe_vel:  Optional[np.ndarray] = None,
        t:         float = 0.0,
    ) -> None:
        """
        Verwerk reacties voor alle actieve segmenten (React-stap).

        Compatibel met GeochemSolver.apply_geochemistry() interface:
        de NzingaFlowSolver roept deze methode aan in step().

        Parameters
        ----------
        store     : SegmentStore  (velden: .pipe, .C, .n)
        dt        : tijdstap [s]
        pipe_diam : (n_pipes,) diameter [m]; None → Av=0 (geen wandsoorten)
        pipe_vel  : niet gebruikt; aanwezig voor compatibiliteit
        t         : gesimuleerde tijd [s]
        """
        n       = store.n
        if n == 0:
            return

        n_pipes = int(store.pipe[:n].max()) + 1 if n > 0 else 1
        self._ensure_wall(n_pipes, self.n_wall)

        # Av per leiding [m²/m³] = 4/D voor een cilinder
        if pipe_diam is not None and self.n_wall > 0:
            Av_arr = 4.0 / np.maximum(pipe_diam, 1e-6)
        else:
            Av_arr = np.zeros(n_pipes)

        has_rates  = bool(self._pipe_rate_fns)
        has_equil  = bool(self._pipe_equil_fns)
        has_form   = bool(self._pipe_formula_fns)
        has_wall   = self.n_wall > 0

        # Groepeer segmenten per leiding voor efficiënte wandkoppeling
        pipe_arr = store.pipe[:n]

        for seg_i in range(n):
            pi     = int(pipe_arr[seg_i])
            Av     = float(Av_arr[pi]) if pi < len(Av_arr) else 0.0
            C_b    = store.C[seg_i].copy()
            C_w    = self._C_wall[pi].copy() if has_wall else np.zeros(0)

            # ── RATE-integratie ─────────────────────────────────────────────
            if has_rates:
                y0  = np.concatenate([C_b, C_w])
                _Av, _t, _pi, _rfns = Av, t, pi, self._pipe_rate_fns
                def _rhs_bound(_t_arg, y, _Av=_Av, _pi=_pi, _rfns=_rfns):
                    return self._rhs_pipe(np.asarray(y), _Av, 0.0, _pi, _rfns)
                y1  = self._integrate(y0, dt, _rhs_bound)
                C_b = np.maximum(y1[:self.n_bulk], 0.0)
                C_w = np.maximum(y1[self.n_bulk:],  0.0) if has_wall else C_w

            # ── EQUIL-iteratie ──────────────────────────────────────────────
            if has_equil:
                C_b, C_w = self._solve_equil(C_b, C_w, Av, pi)

            # ── FORMULA (afgeleide variabelen) ──────────────────────────────
            if has_form:
                state = self._make_state(C_b, C_w, Av, t, pi)
                for name, fn in self._pipe_formula_fns.items():
                    if name in self._bulk_idx:
                        C_b[self._bulk_idx[name]] = max(fn(state), 0.0)

            # ── Schrijf terug ────────────────────────────────────────────────
            store.C[seg_i] = C_b
            if has_wall:
                self._C_wall[pi] = C_w

    # ── Tank-reacties na knoopmenging ────────────────────────────────────────

    def apply_mixing(
        self,
        node_C:    np.ndarray,   # (node_count, n_bulk) — in-place
        node_flow: np.ndarray,   # (node_count,)
        dt:        float,
        t:         float = 0.0,
    ) -> None:
        """
        Verwerk bulk-reacties na knoopmenging (tank + evenwicht bij knopen).

        Compatibel met GeochemSolver.apply_mixing() interface.
        Alleen bulk-soorten; wandsoorten niet aanwezig bij knopen.
        """
        has_rates = bool(self._tank_rate_fns)
        has_form  = bool(self._tank_formula_fns)

        if not has_rates and not has_form:
            return

        node_count = node_C.shape[0]
        for ni in range(node_count):
            if node_flow[ni] <= 0.0:
                continue
            C_b = node_C[ni].copy()
            C_w = np.zeros(0)       # geen wandsoorten bij knopen

            if has_rates:
                y0 = C_b.copy()
                _tfns = self._tank_rate_fns
                def _rhs_tank_bound(_t_arg, y, _f=_tfns):
                    return self._rhs_tank(np.asarray(y), 0.0, _f)
                y1 = self._integrate(y0, dt, _rhs_tank_bound)
                C_b = np.maximum(y1, 0.0)

            if has_form:
                state = self._make_state(C_b, C_w, Av=0.0, t=t)
                for name, fn in self._tank_formula_fns.items():
                    if name in self._bulk_idx:
                        C_b[self._bulk_idx[name]] = max(fn(state), 0.0)

            node_C[ni] = C_b

    # ── Reset ────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset wandsoorten naar nul (bij herstart simulatie)."""
        if self._C_wall is not None:
            self._C_wall[:] = 0.0

    # ── Repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"<MsxReactionSystem "
            f"bulk={self.bulk_species} "
            f"wall={self.wall_species} "
            f"solver={self.solver!r} "
            f"pipe_rates={list(self._pipe_rate_fns.keys())} "
            f"equil={list(self._pipe_equil_fns.keys())}>"
        )


# ── Voorgeconfigureerde reactiesystemen ──────────────────────────────────────

def chloramine_decay_msx(
    k_f:    float = 2.5e-4,   # NH2Cl-vervalconstante [1/s]
    k_ox:   float = 5.0e-5,   # oxidatie HOCl+NH3 [1/(mg/L·s)]
    solver: str   = 'rk4',
) -> MsxReactionSystem:
    """
    Drie-stof chloramine-verval (Vikesland 2001, vereenvoudigd):
        HOCl + NH3 → NH2Cl   (snel)
        NH2Cl     → producten (langzaam)

    Soorten: [HOCl, NH3, NH2Cl]
    """
    return MsxReactionSystem(
        bulk_species = ['HOCl', 'NH3', 'NH2Cl'],
        params       = {'k_ox': k_ox, 'k_f': k_f},
        pipe_rates   = {
            'HOCl':  '-k_ox * HOCl * NH3',
            'NH3':   '-k_ox * HOCl * NH3',
            'NH2Cl': 'k_ox * HOCl * NH3 - k_f * NH2Cl',
        },
        tank_rates   = {
            'HOCl':  '-k_ox * HOCl * NH3',
            'NH3':   '-k_ox * HOCl * NH3',
            'NH2Cl': 'k_ox * HOCl * NH3 - k_f * NH2Cl',
        },
        solver = solver,
    )


def chlorine_nom_msx(
    k_bulk: float = 3e-4,    # chloor+NOM bulk [1/(mg/L·s)]
    k_wall: float = 1e-5,    # wandreactie chloor [m/s]
    solver: str   = 'rk4',
) -> MsxReactionSystem:
    """
    Twee-stof chloor-NOM model:
        Cl2 reageert met NOM in bulk en aan de wand.

    Soorten: [Cl2, NOM]
    """
    return MsxReactionSystem(
        bulk_species = ['Cl2', 'NOM'],
        params       = {'k_b': k_bulk, 'k_w_ms': k_wall},
        pipe_rates   = {
            'Cl2': '-k_b * Cl2 * NOM - k_w_ms * Cl2 * Av',
            'NOM': '-k_b * Cl2 * NOM',
        },
        tank_rates   = {
            'Cl2': '-k_b * Cl2 * NOM',
            'NOM': '-k_b * Cl2 * NOM',
        },
        solver = solver,
    )


def arsenic_oxidation_msx(
    Ka:   float = 10.0,    # arseniet-oxidatiesnelheid [L/(µg·h)] → [L/(µg·s)]
    Kb:   float = 0.1,     # NH2Cl-verval [1/h] → [1/s]
    K1:   float = 5.0,     # adsorptiesnelheid  [L/(µg·h)]
    K2:   float = 1.0,     # desorptiesnelheid  [1/h]
    Smax: float = 50.0,    # max oppervlakconcentratie [µg/m²]
    solver: str = 'radau', # stijf door adsorptie-evenwicht
) -> MsxReactionSystem:
    """
    Arseen-oxidatie + adsorptie (Zhang 2004, MSX voorbeeld 1):
        AS3 + NH2CL → AS5          (oxidatie)
        AS5 ⇌ AS5s                 (adsorptie aan wand)

    Soorten bulk:  [AS3, AS5, NH2CL]
    Soorten wand:  [AS5s]
    Eenheden:      µg/L voor bulk, µg/m² voor wand
    """
    # Eenheden: Ka en K1 in handboek per uur → omzetten naar per seconde
    Ka_s = Ka / 3600.0
    Kb_s = Kb / 3600.0
    K1_s = K1 / 3600.0
    K2_s = K2 / 3600.0

    return MsxReactionSystem(
        bulk_species = ['AS3', 'AS5', 'NH2CL'],
        wall_species = ['AS5s'],
        params       = {'Ka': Ka_s, 'Kb': Kb_s, 'K1': K1_s,
                        'K2': K2_s, 'Smax': Smax,
                        'Ks': K1_s / K2_s},
        pipe_rates   = {
            'AS3':  '-Ka * AS3 * NH2CL',
            'AS5':  'Ka * AS3 * NH2CL - Av * (K1 * (Smax - AS5s) * AS5 - K2 * AS5s)',
            'NH2CL':'-Kb * NH2CL',
            'AS5s': 'K1 * (Smax - AS5s) * AS5 - K2 * AS5s',
        },
        pipe_formulas= {},   # AStot optioneel: voeg toe als extra bulk-soort
        tank_rates   = {
            'AS3':   '-Ka * AS3 * NH2CL',
            'AS5':   'Ka * AS3 * NH2CL',
            'NH2CL': '-Kb * NH2CL',
        },
        solver = solver,
    )
