# Changelog

## [1.1.0] — 2026

### Improved wall reaction model — three corrections with measurable effect

**`lta.py` — `compute_wall_k()` rewritten**

1. **Sherwood correlation: three regimes** (was: hard threshold at Re=4000)
   - Laminar (Re < 2300): Graetz solution with entry-length correction
     `Sh = (3.66³ + max(1.615·(Re·Sc·D/L)^(1/3) − 0.7, 0)³)^(1/3)`
     Base value Sh=3.66 (uniform wall concentration) instead of 3.65 (uniform flux).
     Entry-length correction active when `pipe_length` is provided.
   - Transition (2300 ≤ Re ≤ 4000): linear interpolation Sh_lam ↔ Sh_turb.
     The previous hard threshold at Re=4000 caused a factor 1.5× error on k_wall_vol
     at night-time conditions (v~0.03 m/s in DN100).
   - Turbulent (Re > 4000): Dittus-Boelter `0.023·Re^0.83·Sc^(1/3)` — unchanged.

2. **Temperature dependence** (new, optional via `temperature=[°C]`)
   - `D_mol(T) = D_mol_20 · exp(17000/R · (1/T₀ − 1/T))`  [Hayduk & Laudie 1974]
   - `k_wall(T) = k_wall · 1.047^(T−20)`  [Rossman 2000]
   - Effect: factor 0.59× (5°C) to 1.38× (30°C) on k_wall_vol.
   - Relevant for seasonal variation or ground temperature data.

3. **Stagnation zone correction** (new, automatic)
   - At zero flow (Re → 0), molecular diffusion dominates over convection.
   - Minimum film transport: `k_f_min = 4·D_mol/D` (purely diffusive, Bessel series).
   - Implementation: `k_f = max(k_f_convective, k_f_diffusive)`.
   - Backward compatible: at turbulent velocities k_f_convective is always larger.

**New API parameters for `NzingaFlowSolver.__init__()`:**
- `temperature: float | None = None` — water temperature [°C]; None = Rossman 1994-compatible
- `leakage_fraction: float = 0.0` — fraction of pipe flow lost as volume loss per pipe
- `wall_mode: str = 'two_film'` — interpretation of k_wall:
  - `'two_film'` (default): EPANET-compatible series resistance `k_eff = k_f·k_w/(k_f+k_w)`
  - `'direct'`: k_wall is already k_eff; no film resistance applied

**Leakage modelling in `solver.py` — `step()`:**
- Each segment loses proportional volume per timestep: `V(t+dt) = V(t)·exp(−λ·dt)`
- `λ = leakage_fraction / pipe_residence_time` (scale-independent)
- Concentration remains constant (conservative mixing model, no contaminant ingress)
- Typical value for NL networks: `leakage_fraction=0.10–0.15`

---

### MSX multi-species reaction layer — new module `nzingaflow/msx.py`

Implements the four core concepts of EPANET-MSX 2.0:

- **`RATE`** — kinetic ODE: `dC/dt = f(C_bulk, C_wall, params)`
- **`EQUIL`** — equilibrium condition: `0 = g(C_bulk, C_wall, params)`
- **`FORMULA`** — derived variable: `C = h(C_bulk, C_wall, params)`
- **`WALL` species** — wall-bound species (e.g. biofilm); do not move with the water

**`MsxReactionSystem` — new class:**
- Expressions as Python strings or direct callables; strings compiled via `sympy.lambdify`
- Four numerical ODE solvers: `euler`, `rk4`, `rk45` (scipy), `radau` (scipy, stiff)
- `Av = 4/D [m²/m³]` automatically available in all pipe expressions
- Per-pipe parameter override via `set_pipe_param(pipe_idx, **kwargs)`
- Fully compatible with the `geochem` interface of `NzingaFlowSolver`

**Ready-made models:**
- `chloramine_decay_msx(k_f, k_ox, solver)` — HOCl + NH3 → NH2Cl (Vikesland 2001); 3 species
- `chlorine_nom_msx(k_bulk, k_wall, solver)` — Cl2 + NOM bulk/wall; 2 species
- `arsenic_oxidation_msx(Ka, Kb, K1, K2, Smax, solver)` — AS3→AS5 + wall adsorption (Zhang 2004); 3 bulk + 1 wall species

**Backward compatibility:** fully preserved. All new parameters have defaults that
reproduce the v1.0.0 behaviour.


## [1.0.0] — 2025

### First stable release

**Core functionality**
- Vectorized LTA solver via `NzingaFlowSolver` + `EPSRunner`
- Multi-species support (n_species > 1)
- Wall reactions via two-film model (Rossman 1994)
- CSTR tank model with implicit Euler
- Full geochemistry via PhreeqPython (`GeochemSolver`)
- Flow reversal detection and correction

**Performance**
- Numba JIT kernels for advect, decay, exit-detect, node-mixing, merging (~4–5×)
- Combined bulk+wall decay in one memory pass (`combined_decay_multi`)
- Pre-allocated working buffers: no heap allocations in hot path
- CSR pipe-index in `SegmentStore`: lexsort skipped when cached
- Vectorized `_handle_exits` via `_node_type`/`_node_out0` routing cache (~734×)
- Numba merging kernels with early-exit (~29×)

**Correctness fixes**
- `MassBalanceTracker`: wall loss estimate now uses pre-step mass
- `_apply_flow_reversal`: Python loop replaced by vectorized `np.isin`
- `booster_inject`: volume unit corrected (flow × dt → m³)
- `parse_inp`: conversion factors for MGD/IMGD/AFD/CFS corrected
- `SpeciesMap.__post_init__`: `assert` → `ValueError`
- `HydraulicModel._to_cms`: marked as deprecated

**Package**
- `py.typed` marker (PEP 561)
- Fully documented public API
- 10 scientific validation tests (V1–V10)
- CI via GitHub Actions
