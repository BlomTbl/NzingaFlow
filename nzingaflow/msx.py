# nzingaflow/msx.py
"""
MsxReactionSystem — MSX-compatibele multi-species reactielaag voor NzingaFlow.

Twee verbeteringen t.o.v. de vorige versie, gebaseerd op EPANET-MSX broncode:

1. ROS2-solver (ros2.c, Verwer et al. 1999 / L. Rossman US EPA)
   Rosenbrock 2(1) met adaptieve stapgrootte. Geschikt voor stijve
   systemen (chloramine, arsenaat-adsorptie) zonder scipy-afhankelijkheid.
   Vervangt 'radau' als aanbevolen keuze voor stijve kinetiek.

2. Ingebouwde expressie-parser (mathexpr.c, Rossman/Shang/Uber US EPA)
   Tokenizer + postfix-evaluator. Vervangt sympy/eval() volledig.
   Geen externe afhankelijkheden voor string-expressies.
   Ondersteunt: + - * / ^ () abs sgn sqrt exp log log10
                sin cos tan cot asin acos atan acot
                sinh cosh tanh coth step

ODE-solvers:
  'euler'  — voorwaarts Euler (snel, niet-stijf)
  'rk4'    — klassieke RK4 (standaard, niet-stijf)
  'ros2'   — Rosenbrock 2(1) (stijf, geen scipy)  <- nieuw
  'rk45'   — adaptief RK45 via scipy
  'radau'  — Radau IIA via scipy (achterwaarts compatibel)
"""

from __future__ import annotations
import math
import warnings
import numpy as np
from typing import Callable, Dict, List, Optional

_SOLVERS = ('euler', 'rk4', 'ros2', 'rk45', 'radau')


# ══════════════════════════════════════════════════════════════════════════════
#  EXPRESSIE-PARSER  (gebaseerd op EPANET-MSX mathexpr.c)
# ══════════════════════════════════════════════════════════════════════════════

_OP_LPAREN=1; _OP_RPAREN=2; _OP_ADD=3; _OP_SUB=4; _OP_MUL=5; _OP_DIV=6
_OP_NUM=7; _OP_VAR=8; _OP_NEG=9
_OP_COS=10; _OP_SIN=11; _OP_TAN=12; _OP_COT=13; _OP_ABS=14; _OP_SGN=15
_OP_SQRT=16; _OP_LOG=17; _OP_EXP=18; _OP_ASIN=19; _OP_ACOS=20; _OP_ATAN=21
_OP_ACOT=22; _OP_SINH=23; _OP_COSH=24; _OP_TANH=25; _OP_COTH=26
_OP_LOG10=27; _OP_STEP=28; _OP_POW=31

_MATH_FUNCS = {
    'COS':_OP_COS,'SIN':_OP_SIN,'TAN':_OP_TAN,'COT':_OP_COT,
    'ABS':_OP_ABS,'SGN':_OP_SGN,'SQRT':_OP_SQRT,'LOG':_OP_LOG,
    'EXP':_OP_EXP,'ASIN':_OP_ASIN,'ACOS':_OP_ACOS,'ATAN':_OP_ATAN,
    'ACOT':_OP_ACOT,'SINH':_OP_SINH,'COSH':_OP_COSH,'TANH':_OP_TANH,
    'COTH':_OP_COTH,'LOG10':_OP_LOG10,'STEP':_OP_STEP,
}


class _Parser:
    """Tokenizer + recursive-descent parser → postfix lijst van (opcode,fval,ivar)."""

    def __init__(self, formula: str, var_names: list):
        self._vi   = {n.upper(): i for i, n in enumerate(var_names)}
        self._s    = formula
        self._pos  = 0
        self._len  = len(formula)
        self._err  = False
        self._bc   = 0
        self._prev = 0
        self._cur  = 0
        self._fval = 0.0
        self._ivar = -1

    def _digit(self, c): return '0' <= c <= '9'
    def _letter(self, c): return c.isalpha() or c == '_'

    def _get_token(self):
        s = self._s; p = self._pos; n = self._len
        start = p
        while p < n and (self._letter(s[p]) or self._digit(s[p])): p += 1
        self._pos = p
        return s[start:p]

    def _get_number(self):
        s = self._s; p = self._pos; n = self._len; start = p
        while p < n and self._digit(s[p]): p += 1
        if p < n and s[p] == '.':
            p += 1
            while p < n and self._digit(s[p]): p += 1
        if p < n and s[p].upper() == 'E':
            p += 1
            if p < n and s[p] in ('+','-'): p += 1
            while p < n and self._digit(s[p]): p += 1
        self._pos = p
        return float(s[start:p])

    def _get_operand(self):
        """Herkent enkelteken-operatoren. Geeft (code, advance) terug."""
        c = self._s[self._pos]
        if c == '(': return _OP_LPAREN
        if c == ')': return _OP_RPAREN
        if c == '+': return _OP_ADD
        if c == '*': return _OP_MUL
        if c == '/': return _OP_DIV
        if c == '^': return _OP_POW
        if c == '-': return _OP_SUB
        return 0

    def _lex(self):
        s = self._s; n = self._len
        while self._pos < n and s[self._pos] == ' ': self._pos += 1
        if self._pos >= n: return 0

        code = self._get_operand()

        if code == _OP_SUB:
            # Negatief getal? Alleen als vorig token begin of '(' was EN
            # het volgende teken een cijfer is
            if (self._pos+1 < n and
                    self._digit(s[self._pos+1]) and
                    self._cur in (0, _OP_LPAREN)):
                self._pos += 1          # sla '-' over
                self._fval = -self._get_number()   # leest cijfers, zet pos
                code = _OP_NUM
                # pos staat nu NA het getal — geen extra pos+=1 nodig
            else:
                self._pos += 1          # gewone aftrekking: sla '-' over
        elif code != 0:
            self._pos += 1              # enkelteken-operator: sla over
        else:
            # Geen operand-teken: letter, cijfer of fout
            if self._letter(s[self._pos]):
                tok = self._get_token().upper()
                if tok in _MATH_FUNCS:
                    code = _MATH_FUNCS[tok]
                elif tok in self._vi:
                    self._ivar = self._vi[tok]; code = _OP_VAR
                else:
                    self._err = True; return 0
            elif self._digit(s[self._pos]):
                self._fval = self._get_number(); code = _OP_NUM
            else:
                self._err = True; return 0

        self._prev = self._cur; self._cur = code
        return code

    def _single_op(self, lex):
        nodes = []
        if lex[0] == _OP_LPAREN:
            self._bc += 1; nodes = self._tree()
        else:
            if lex[0] < _OP_NUM or lex[0] == _OP_NEG or lex[0] > 30:
                self._err = True; return []
            op = lex[0]
            if op == _OP_NUM:   nodes = [(op, self._fval, -1)]
            elif op == _OP_VAR: nodes = [(op, 0.0, self._ivar)]
            else:
                lex[0] = self._lex()
                if lex[0] != _OP_LPAREN: self._err = True; return []
                self._bc += 1
                inner = self._tree()
                nodes = inner + [(op, 0.0, -1)]
        lex[0] = self._lex()
        return nodes

    def _op(self, lex):
        lex[0] = self._lex()
        neg = False
        if self._prev in (0, _OP_LPAREN):
            if lex[0] == _OP_SUB:   neg = True; lex[0] = self._lex()
            elif lex[0] == _OP_ADD: lex[0] = self._lex()
        left = self._single_op(lex)
        while lex[0] in (_OP_MUL, _OP_DIV, _OP_POW):
            op = lex[0]; lex[0] = self._lex()
            right = self._single_op(lex)
            left = left + right + [(op, 0.0, -1)]
        if neg: left = left + [(_OP_NEG, 0.0, -1)]
        return left

    def _tree(self):
        lex = [0]; left = self._op(lex)
        while True:
            if lex[0] in (0, _OP_RPAREN):
                if lex[0] == _OP_RPAREN: self._bc -= 1
                break
            if lex[0] not in (_OP_ADD, _OP_SUB): self._err = True; break
            op = lex[0]; right = self._op(lex)
            left = left + right + [(op, 0.0, -1)]
        return left

    def compile(self):
        nodes = self._tree()
        if self._err or self._bc != 0:
            raise ValueError(
                f"Expressie-syntaxfout: {self._s!r}  "
                f"(brackets={self._bc}, err={self._err})")
        return nodes


def _eval_postfix(nodes: list, var_values: list) -> float:
    """Stack-evaluator van postfix-expressie. Identiek aan mathexpr_eval() (mathexpr.c)."""
    stack = [0.0] * 64; sp = 0
    for opcode, fvalue, ivar in nodes:
        if   opcode == _OP_NUM:  sp += 1; stack[sp] = fvalue
        elif opcode == _OP_VAR:  sp += 1; stack[sp] = var_values[ivar]
        elif opcode == _OP_ADD:  stack[sp-1] += stack[sp]; sp -= 1
        elif opcode == _OP_SUB:  stack[sp-1] -= stack[sp]; sp -= 1
        elif opcode == _OP_MUL:  stack[sp-1] *= stack[sp]; sp -= 1
        elif opcode == _OP_DIV:
            r = stack[sp]; sp -= 1
            stack[sp] = stack[sp] / r if r != 0.0 else 0.0
        elif opcode == _OP_POW:
            r = stack[sp]; sp -= 1; b = stack[sp]
            stack[sp] = math.exp(r*math.log(b)) if b > 0 else 0.0
        elif opcode == _OP_NEG:  stack[sp] = -stack[sp]
        elif opcode == _OP_ABS:  stack[sp] = abs(stack[sp])
        elif opcode == _OP_SGN:
            v = stack[sp]; stack[sp] = 1.0 if v>0 else (-1.0 if v<0 else 0.0)
        elif opcode == _OP_SQRT: stack[sp] = math.sqrt(max(stack[sp],0.0))
        elif opcode == _OP_LOG:
            v = stack[sp]; stack[sp] = math.log(v) if v>0 else 0.0
        elif opcode == _OP_LOG10:
            v = stack[sp]; stack[sp] = math.log10(v) if v>0 else 0.0
        elif opcode == _OP_EXP:  stack[sp] = math.exp(stack[sp])
        elif opcode == _OP_SIN:  stack[sp] = math.sin(stack[sp])
        elif opcode == _OP_COS:  stack[sp] = math.cos(stack[sp])
        elif opcode == _OP_TAN:  stack[sp] = math.tan(stack[sp])
        elif opcode == _OP_COT:
            v = stack[sp]; stack[sp] = 1.0/math.tan(v) if v!=0 else 0.0
        elif opcode == _OP_ASIN: stack[sp] = math.asin(max(-1.,min(1.,stack[sp])))
        elif opcode == _OP_ACOS: stack[sp] = math.acos(max(-1.,min(1.,stack[sp])))
        elif opcode == _OP_ATAN: stack[sp] = math.atan(stack[sp])
        elif opcode == _OP_ACOT: stack[sp] = math.pi/2 - math.atan(stack[sp])
        elif opcode == _OP_SINH: stack[sp] = math.sinh(stack[sp])
        elif opcode == _OP_COSH: stack[sp] = math.cosh(stack[sp])
        elif opcode == _OP_TANH: stack[sp] = math.tanh(stack[sp])
        elif opcode == _OP_COTH:
            v = stack[sp]
            e = math.exp(min(2*v, 700))
            stack[sp] = (e+1)/(e-1) if e != 1.0 else 0.0
        elif opcode == _OP_STEP: stack[sp] = 0.0 if stack[sp] <= 0.0 else 1.0
    return stack[sp]


def _compile_str(formula: str, var_names: list, species: str) -> Callable:
    """Compileer expressiestring naar callable via ingebouwde parser (geen sympy/eval)."""
    try:
        postfix = _Parser(formula, var_names).compile()
    except ValueError as e:
        raise ValueError(
            f"Fout in expressie voor {species!r}: {e}\n"
            f"  Formule    : {formula!r}\n"
            f"  Bekende namen: {var_names}"
        ) from e

    def fn(state: dict) -> float:
        return _eval_postfix(postfix, [state.get(v, 0.0) for v in var_names])
    fn.__name__ = f"msx_{species}"
    fn._formula = formula
    return fn


# ══════════════════════════════════════════════════════════════════════════════
#  ROS2 — Rosenbrock 2(1) stijve ODE-integrator
#  Gebaseerd op EPANET-MSX ros2.c (Verwer et al., SIAM J. Sci. Comput. 1999)
# ══════════════════════════════════════════════════════════════════════════════

_UROUND = 2.3e-16
_G_ROS2 = 1.0 + 1.0 / math.sqrt(2.0)   # γ = 1 + 1/√2


def _ros2_integrate(y0, dt, rhs, atol=1e-6, rtol=1e-3):
    """
    Rosenbrock 2(1) met adaptieve stapgrootte.
    Identieke algoritme als ros2_integrate() in EPANET-MSX ros2.c.

    - Jacobian via eindige differenties (perturbatie √UROUND·max(|y|, 1e-6))
    - LU via numpy.linalg.solve
    - Foutschatting: RMSE((y2-y1)/ytol)
    - Stapfactor: 0.9/√err, begrensd [0.1, 10]
    - Concentraties < UROUND worden op 0 gezet na acceptatie
    """
    n = len(y0)
    y = y0.copy().astype(np.float64)
    g = _G_ROS2
    t = 0.0; tnext = float(dt)

    # Initiële stapgrootte
    f0 = rhs(0.0, y); h = dt
    for j in range(n):
        ytol = atol + rtol * abs(y[j])
        if f0[j] != 0.0: h = min(h, ytol / abs(f0[j]))
    h = max(1e-8, min(h, dt))

    ghinv1 = 0.0; is_rej = False; f_cur = f0

    while t < tnext:
        if 0.1 * abs(h) <= abs(t) * _UROUND:
            h = tnext - t  # forceer voltooiing
        tplus = min(t + h, tnext); h_eff = tplus - t

        # Jacobian (alleen bij geaccepteerde stap)
        if not is_rej:
            f_cur = rhs(t, y)
            J = np.zeros((n, n))
            for j in range(n):
                eps = math.sqrt(_UROUND) * max(abs(y[j]), 1e-6)
                yp = y.copy(); yp[j] += eps
                J[:, j] = (rhs(t, yp) - f_cur) / eps

        # A = γ/h·I - J  (met correctie voor stapgrootte-wijziging)
        ghinv = -1.0 / (g * h_eff)
        dghinv = ghinv - ghinv1
        if not is_rej:
            A = -J.copy()
            for j in range(n): A[j, j] += ghinv
        else:
            for j in range(n): A[j, j] += dghinv
        ghinv1 = ghinv

        # Stadium 1: (γ/h·I - J)·k1 = f(y)
        try:
            k1 = np.linalg.solve(A, f_cur * ghinv)
        except np.linalg.LinAlgError:
            h = max(0.5*h_eff, 1e-8); is_rej = True; continue

        # Stadium 2: (γ/h·I - J)·k2 = f(y+h·k1) - 2·k1
        y1 = y + h_eff * k1
        f1 = rhs(tplus, y1)
        try:
            k2 = np.linalg.solve(A, (f1 - 2.0*k1) * ghinv)
        except np.linalg.LinAlgError:
            h = max(0.5*h_eff, 1e-8); is_rej = True; continue

        # 2e-orde oplossing
        y2 = y + 1.5*h_eff*k1 + 0.5*h_eff*k2

        # Foutschatting (RMSE)
        err = 0.0
        for j in range(n):
            ytol = atol + rtol * abs(y2[j])
            ej = abs(y2[j] - y1[j]) / ytol
            err += ej * ej
        err = max(math.sqrt(err / n), _UROUND)

        # Stapfactor (identiek aan ros2.c)
        factor = 0.9 / math.sqrt(err)
        facmax = 1.0 if is_rej else 10.0
        factor = min(max(factor, 0.1), facmax)
        h_new  = min(factor * h_eff, tnext - tplus + h_eff)

        if err > 1.0:
            h = max(0.5*h_eff, 1e-8); is_rej = True
        else:
            for j in range(n):
                y[j] = y2[j] if y2[j] > _UROUND else 0.0
            t = tplus; h = h_new; is_rej = False

    return np.maximum(y, 0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  MsxReactionSystem
# ══════════════════════════════════════════════════════════════════════════════

class MsxReactionSystem:
    """
    Multi-species reactielaag compatibel met EPANET-MSX concepten.
    Plugt in op NzingaFlowSolver via geochem=rxn.

    Parameters
    ----------
    bulk_species  : list[str]   — bulksoorten (volgorde = C-matrix index)
    wall_species  : list[str]   — wandsoorten (lege lijst = geen)
    params        : dict        — reactieparameters (constanten)
    pipe_rates    : dict        — RATE-expressies leidingen (str of callable)
    pipe_equil    : dict        — EQUIL-expressies leidingen
    pipe_formulas : dict        — FORMULA-expressies leidingen
    tank_rates    : dict        — RATE-expressies tanks
    tank_formulas : dict        — FORMULA-expressies tanks
    solver        : str         — 'euler'|'rk4'|'ros2'|'rk45'|'radau'
    rtol, atol    : float       — toleranties voor adaptieve solvers
    """

    def __init__(
        self,
        bulk_species,
        wall_species   = (),
        params         = None,
        pipe_rates     = None,
        pipe_equil     = None,
        pipe_formulas  = None,
        tank_rates     = None,
        tank_formulas  = None,
        solver         = 'rk4',
        rtol           = 1e-3,
        atol           = 1e-6,
    ):
        if solver not in _SOLVERS:
            raise ValueError(f"solver moet één van {_SOLVERS} zijn, niet {solver!r}")

        self.bulk_species = list(bulk_species)
        self.wall_species = list(wall_species)
        self.n_bulk       = len(bulk_species)
        self.n_wall       = len(wall_species)
        self.params       = dict(params or {})
        self.solver       = solver
        self.rtol         = rtol
        self.atol         = atol

        self._pipe_params: Dict[int, Dict[str, float]] = {}
        self._C_wall:  Optional[np.ndarray] = None
        self._n_pipes: int = 0

        # Variabelenamen voor de parser
        self._var_names: List[str] = (
            self.bulk_species + self.wall_species +
            list(self.params.keys()) + ['Av', 't']
        )

        self._pipe_rate_fns    = self._compile_all(pipe_rates    or {})
        self._pipe_equil_fns   = self._compile_all(pipe_equil    or {})
        self._pipe_formula_fns = self._compile_all(pipe_formulas or {})
        self._tank_rate_fns    = self._compile_all(tank_rates    or {})
        self._tank_formula_fns = self._compile_all(tank_formulas or {})

        self._bulk_idx = {s: i for i, s in enumerate(bulk_species)}
        self._wall_idx = {s: i for i, s in enumerate(wall_species)}

    def _compile_all(self, exprs):
        compiled = {}
        for name, expr in exprs.items():
            if callable(expr):
                compiled[name] = expr
            elif isinstance(expr, str):
                compiled[name] = _compile_str(expr, self._var_names, name)
            else:
                raise TypeError(
                    f"Expressie voor {name!r} moet str of callable zijn")
        return compiled

    def set_pipe_param(self, pipe_idx: int, **kwargs: float) -> None:
        """Stel parameteroverrides in voor één leiding (MSX [PARAMETERS])."""
        self._pipe_params.setdefault(pipe_idx, {}).update(kwargs)

    def get_wall_concentrations(self):
        return self._C_wall.copy() if self._C_wall is not None else None

    def set_wall_concentrations(self, C_wall: np.ndarray) -> None:
        self._C_wall  = np.asarray(C_wall, dtype=np.float64).copy()
        self._n_pipes = C_wall.shape[0]

    def reset(self) -> None:
        if self._C_wall is not None:
            self._C_wall[:] = 0.0

    def _ensure_wall(self, n_pipes: int) -> None:
        if self._C_wall is None or self._n_pipes != n_pipes:
            self._C_wall  = np.zeros((n_pipes, self.n_wall), dtype=np.float64)
            self._n_pipes = n_pipes

    def _make_state(self, C_bulk, C_wall, Av, t=0.0, pipe_idx=None):
        state = {s: float(C_bulk[i]) for i, s in enumerate(self.bulk_species)}
        state.update({s: float(C_wall[i]) for i, s in enumerate(self.wall_species)})
        state.update(self.params)
        if pipe_idx is not None and pipe_idx in self._pipe_params:
            state.update(self._pipe_params[pipe_idx])
        state['Av'] = float(Av)
        state['t']  = float(t)
        return state

    def _rhs_pipe(self, y, Av, t, pipe_idx, rate_fns):
        C_bulk = y[:self.n_bulk]; C_wall = y[self.n_bulk:]
        state  = self._make_state(C_bulk, C_wall, Av, t, pipe_idx)
        dy = np.zeros_like(y)
        for name, fn in rate_fns.items():
            val = fn(state)
            if name in self._bulk_idx:   dy[self._bulk_idx[name]] += val
            elif name in self._wall_idx: dy[self.n_bulk + self._wall_idx[name]] += val
        return dy

    def _rhs_tank(self, y, t, rate_fns):
        C_bulk = y[:self.n_bulk]
        state  = self._make_state(C_bulk, np.zeros(self.n_wall), 0.0, t)
        dy = np.zeros_like(y)
        for name, fn in rate_fns.items():
            if name in self._bulk_idx:
                dy[self._bulk_idx[name]] += fn(state)
        return dy

    def _integrate(self, y0, dt, rhs):
        y0 = np.asarray(y0, dtype=np.float64)
        if self.solver == 'euler':
            return np.maximum(y0 + dt * rhs(0.0, y0), 0.0)
        elif self.solver == 'rk4':
            k1 = rhs(0.0,    y0)
            k2 = rhs(0.5*dt, np.maximum(y0+0.5*dt*k1, 0.0))
            k3 = rhs(0.5*dt, np.maximum(y0+0.5*dt*k2, 0.0))
            k4 = rhs(dt,     np.maximum(y0+    dt*k3, 0.0))
            return np.maximum(y0 + (dt/6.0)*(k1+2*k2+2*k3+k4), 0.0)
        elif self.solver == 'ros2':
            return _ros2_integrate(y0, dt, rhs, atol=self.atol, rtol=self.rtol)
        else:  # rk45 / radau
            from scipy.integrate import solve_ivp
            method = 'RK45' if self.solver == 'rk45' else 'Radau'
            sol = solve_ivp(rhs, [0.0, dt], y0, method=method,
                            rtol=self.rtol, atol=self.atol)
            if not sol.success:
                warnings.warn(f"ODE niet geconvergeerd ({method}): {sol.message}",
                              RuntimeWarning, stacklevel=3)
            return np.maximum(sol.y[:, -1], 0.0)

    def _solve_equil(self, C_bulk, C_wall, Av, pipe_idx,
                     max_iter=20, tol=1e-8):
        C_b = C_bulk.copy(); C_w = C_wall.copy()
        for _ in range(max_iter):
            converged = True
            for name, fn in self._pipe_equil_fns.items():
                state = self._make_state(C_b, C_w, Av, pipe_idx=pipe_idx)
                f0 = fn(state)
                if abs(f0) < tol: continue
                converged = False
                if name in self._bulk_idx:
                    idx = self._bulk_idx[name]
                    dx  = max(abs(C_b[idx])*1e-6, 1e-12)
                    C_b[idx] += dx
                    f1 = fn(self._make_state(C_b, C_w, Av, pipe_idx=pipe_idx))
                    dfdx = (f1-f0)/dx if abs(f1-f0)>1e-20 else 1.0
                    C_b[idx] = max(C_b[idx]-dx-f0/dfdx, 0.0)
                elif name in self._wall_idx:
                    idx = self._wall_idx[name]
                    dx  = max(abs(C_w[idx])*1e-6, 1e-12)
                    C_w[idx] += dx
                    f1 = fn(self._make_state(C_b, C_w, Av, pipe_idx=pipe_idx))
                    dfdx = (f1-f0)/dx if abs(f1-f0)>1e-20 else 1.0
                    C_w[idx] = max(C_w[idx]-dx-f0/dfdx, 0.0)
            if converged: break
        return C_b, C_w

    def apply_geochemistry(self, store, dt, pipe_diam=None,
                           pipe_vel=None, t=0.0):
        """Verwerk reacties voor alle actieve segmenten (compatibel met GeochemSolver)."""
        n = store.n
        if n == 0: return
        n_pipes = int(store.pipe[:n].max()) + 1
        self._ensure_wall(n_pipes)

        Av_arr = (4.0 / np.maximum(pipe_diam, 1e-6)
                  if pipe_diam is not None and self.n_wall > 0
                  else np.zeros(n_pipes))

        has_rates = bool(self._pipe_rate_fns)
        has_equil = bool(self._pipe_equil_fns)
        has_form  = bool(self._pipe_formula_fns)
        has_wall  = self.n_wall > 0
        pipe_arr  = store.pipe[:n]

        for seg_i in range(n):
            pi  = int(pipe_arr[seg_i])
            Av  = float(Av_arr[pi]) if pi < len(Av_arr) else 0.0
            C_b = store.C[seg_i].copy()
            C_w = self._C_wall[pi].copy() if has_wall else np.zeros(0)

            if has_rates:
                y0 = np.concatenate([C_b, C_w])
                _Av=Av; _pi=pi; _rfns=self._pipe_rate_fns
                def _rhs(_t, y, _Av=_Av, _pi=_pi, _rfns=_rfns):
                    return self._rhs_pipe(np.asarray(y), _Av, 0.0, _pi, _rfns)
                y1  = self._integrate(y0, dt, _rhs)
                C_b = np.maximum(y1[:self.n_bulk], 0.0)
                C_w = np.maximum(y1[self.n_bulk:], 0.0) if has_wall else C_w

            if has_equil:
                C_b, C_w = self._solve_equil(C_b, C_w, Av, pi)

            if has_form:
                state = self._make_state(C_b, C_w, Av, t, pi)
                for name, fn in self._pipe_formula_fns.items():
                    if name in self._bulk_idx:
                        C_b[self._bulk_idx[name]] = max(fn(state), 0.0)

            store.C[seg_i] = C_b
            if has_wall: self._C_wall[pi] = C_w

    def apply_mixing(self, node_C, node_flow, dt, t=0.0):
        """Tank-reacties na knoopmenging (compatibel met GeochemSolver)."""
        has_rates = bool(self._tank_rate_fns)
        has_form  = bool(self._tank_formula_fns)
        if not has_rates and not has_form: return

        for ni in range(node_C.shape[0]):
            if node_flow[ni] <= 0.0: continue
            C_b = node_C[ni].copy()
            if has_rates:
                _tfns = self._tank_rate_fns
                def _rhs_t(_t, y, _f=_tfns):
                    return self._rhs_tank(np.asarray(y), 0.0, _f)
                C_b = self._integrate(C_b, dt, _rhs_t)
            if has_form:
                state = self._make_state(C_b, np.zeros(0), 0.0, t)
                for name, fn in self._tank_formula_fns.items():
                    if name in self._bulk_idx:
                        C_b[self._bulk_idx[name]] = max(fn(state), 0.0)
            node_C[ni] = C_b

    def __repr__(self):
        return (
            f"<MsxReactionSystem "
            f"bulk={self.bulk_species} wall={self.wall_species} "
            f"solver={self.solver!r} "
            f"pipe_rates={list(self._pipe_rate_fns.keys())} "
            f"equil={list(self._pipe_equil_fns.keys())}>"
        )


# ── Voorgeconfigureerde reactiesystemen ───────────────────────────────────────

def chloramine_decay_msx(k_f=2.5e-4, k_ox=5.0e-5, solver='ros2'):
    """HOCl + NH3 → NH2Cl (Vikesland 2001). Soorten: [HOCl, NH3, NH2Cl]."""
    return MsxReactionSystem(
        bulk_species=['HOCl','NH3','NH2Cl'],
        params={'k_ox':k_ox,'k_f':k_f},
        pipe_rates={
            'HOCl':  '-k_ox * HOCl * NH3',
            'NH3':   '-k_ox * HOCl * NH3',
            'NH2Cl': 'k_ox * HOCl * NH3 - k_f * NH2Cl',
        },
        tank_rates={
            'HOCl':  '-k_ox * HOCl * NH3',
            'NH3':   '-k_ox * HOCl * NH3',
            'NH2Cl': 'k_ox * HOCl * NH3 - k_f * NH2Cl',
        },
        solver=solver,
    )


def chlorine_nom_msx(k_bulk=3e-4, k_wall=1e-5, solver='rk4'):
    """Cl2 + NOM bulk- en wandreactie. Soorten: [Cl2, NOM]."""
    return MsxReactionSystem(
        bulk_species=['Cl2','NOM'],
        params={'k_b':k_bulk,'k_w_ms':k_wall},
        pipe_rates={
            'Cl2': '-k_b * Cl2 * NOM - k_w_ms * Cl2 * Av',
            'NOM': '-k_b * Cl2 * NOM',
        },
        tank_rates={
            'Cl2': '-k_b * Cl2 * NOM',
            'NOM': '-k_b * Cl2 * NOM',
        },
        solver=solver,
    )


def arsenic_oxidation_msx(Ka=10.0, Kb=0.1, K1=5.0, K2=1.0, Smax=50.0,
                          solver='ros2'):
    """AS3-oxidatie + adsorptie (Zhang 2004). Bulk: [AS3,AS5,NH2CL] Wall: [AS5s]."""
    Ka_s=Ka/3600; Kb_s=Kb/3600; K1_s=K1/3600; K2_s=K2/3600
    return MsxReactionSystem(
        bulk_species=['AS3','AS5','NH2CL'],
        wall_species=['AS5s'],
        params={'Ka':Ka_s,'Kb':Kb_s,'K1':K1_s,'K2':K2_s,
                'Smax':Smax,'Ks':K1_s/K2_s},
        pipe_rates={
            'AS3':   '-Ka * AS3 * NH2CL',
            'AS5':   'Ka * AS3 * NH2CL - Av * (K1 * (Smax - AS5s) * AS5 - K2 * AS5s)',
            'NH2CL': '-Kb * NH2CL',
            'AS5s':  'K1 * (Smax - AS5s) * AS5 - K2 * AS5s',
        },
        tank_rates={
            'AS3':   '-Ka * AS3 * NH2CL',
            'AS5':   'Ka * AS3 * NH2CL',
            'NH2CL': '-Kb * NH2CL',
        },
        solver=solver,
    )
