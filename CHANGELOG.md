# Changelog

## [1.2.2] — 2026-07-28

### MSX native library bridge — bundled binaries + critical bugfixes

`msxlibrary.py` (`MsxSimulation`/`MsxNativeLib`) was not actually functional
since its introduction in v1.2.0: no `libepanetmsx`/`epanetmsx.dll` was
available on the system, and even with the library present the bridge still
failed due to several underlying bugs. Both are now fixed.

**Bundled binaries (`nzingaflow/lib/`)**
- `libepanetmsx.so` + `libepanet2_msx.so` (Linux x86-64) and `epanetmsx.dll` +
  `epanet2_msx.dll` (Windows x86-64), built from the official EPANET-MSX
  2.0 source (github.com/USEPA/EPANETMSX). macOS is not yet built (see
  `nzingaflow/lib/README.txt` for build instructions).
- `_default_lib_path()` now also searches `lib/`, platform-aware.

**Bugfixes in `msxlibrary.py`**
- `MSXstep` used `c_long` instead of `c_double` for its time parameters —
  an ABI mismatch against the MSX 2.0 C API (potentially memory-corrupting
  on Windows, silently wrong time values elsewhere).
- `_epanet_open()` was a no-op; `ENopen()` was never called before
  `MSXopen()`, even though MSX depends on it for the shared network state
  (see the official CLI reference, `msxmain.c`). The EPANET network is now
  actually opened via a new `en_open()`/`en_close()` binding.
- Node/link counts and names went through `MSXgetcount`/`MSXgetID`, which
  don't support that object type (only SPECIES/CONSTANT/PARAMETER/PATTERN) —
  this raised `MSX error 515` everywhere. Rerouted through the correct
  EPANET-layer calls (`ENgetcount`, `ENgetnodeid`, `ENgetlinkid`,
  `ENgetnodeindex`, `ENgetlinkindex`), with a dedicated error translator
  (`ENgeterror`) since EN and MSX error codes use different numbering.
- **Library name collision with epynet:** the epanet2 library that
  `libepanetmsx` links against shared its name (and, on Linux, its SONAME)
  with epynet's own bundled epanet2 library. Running both in the same
  process (as NzingaFlow does, since it depends on epynet) made the dynamic
  linker / Windows loader silently resolve to the wrong, already-loaded
  copy. Fixed with a unique name (`libepanet2_msx.so` / `epanet2_msx.dll`)
  for the companion library.

**Docs:** the earlier reference to EPANET-MSX **1.1** (the source of the
`MSXstep` bug) has been corrected to **2.0** in both the module docstring
and the READMEs.

**Tests:** new `tests/test_msxlibrary.py`, run against the official arsenic
oxidation example network (USEPA EPANETMSX Examples/), including regression
tests for the bugs listed above.

**Backward compatibility:** `MsxSimulation`/`MsxNativeLib`'s public API is
unchanged; this is a bugfix and bundling release only.

## [1.2.1] — 2026-06-22

### Valve support in transport topology — `include_valves` parameter

**`hydraulics.py` — `HydraulicModel`**
- New parameter `include_valves: bool = False` on `HydraulicModel.__init__()`.
  When `True`, all EPANET valve types (PRV, PSV, PBV, FCV, TCV, GPV, PCV) are
  added to the transport topology alongside pipes.
- `_get_links()` updated: appends `list(self.net.valves)` when `include_valves=True`.
- `summary()` updated: active-links label now reflects `+valves` when the flag is set.

**`solver.py` — `NzingaFlowSolver`**
- New parameter `include_valves: bool = False` on `NzingaFlowSolver.__init__()`,
  forwarded to `HydraulicModel`.

**Background**

EPANET always solves hydraulics correctly for valves; NzingaFlow previously excluded
them from the Lagrangian transport topology. This meant:
- No residence time or decay calculated across valve elements.
- Valves that were the *sole connection* between two network segments caused those
  segments to be topologically disconnected in NzingaFlow.

With `include_valves=True` valves are treated as short pipe elements; flow and
velocity are taken directly from the EPANET solution.

**Backward compatibility:** default is `False`; existing behaviour is unchanged.

```python
# Enable valve transport
solver = NzingaFlowSolver("network.inp", include_valves=True)

# Or directly via HydraulicModel
hyd = HydraulicModel("network.inp", include_valves=True)
```

### Bugfix — `include_pumps=True` crashed on `diameter`

**`hydraulics.py` — `HydraulicModel.get_topology()`**
- `diam_raw` now uses `getattr(lnk, 'diameter', 0.0)` instead of `lnk.diameter`,
  analogous to the existing `length` fallback for valves.

**Background**

epynet `Pump` objects have no `diameter` static property (EPANET does not
define a diameter for pumps). Any network containing a real pump raised
`AttributeError: ('Nonexistant Attribute', 'diameter')` as soon as
`include_pumps=True` was used. Pumps are now treated as zero-area elements
(consistent with how valves are already treated as zero-length elements),
so `get_topology()` no longer raises.

Regression tests for both the valve-transport path and this pump fix were
added in `tests/test_epynet_networks.py` (real EPANET `.inp` networks loaded
via epynet, marked `requires_epynet`).

---

## [1.2.0] — 2026-04-24

### New module `nzingaflow/msxlibrary.py` — direct binding to the EPANET-MSX native library

Provides three layers above the `epanetmsx.dll` / `libepanetmsx.so` C API:

**`MsxNativeLib` — ctypes wrapper (layer 1)**
- Thin wrapper around all `MSX_*` C functions (EPANET-MSX 1.1, revision 11/01/10).
- Automatic signature binding via `_bind_signatures()`; error codes → `MsxError` when `strict=True`.
- Automatic library detection (`_default_lib_path()`) for Windows, Linux, and macOS.
- Full coverage: open/close, hydraulics, quality, step-by-step simulation, sources,
  patterns, constants, parameters, initial qualities.

**`MsxNetworkState` / dataclasses (layer 2)**
- `MsxNetworkState` — snapshot after `load()`: species, constants, sources, initial qualities, patterns.
- `MsxSpecies` — per-species metadata (index, name, bulk/wall, units, tolerances).
- `MsxSourceRecord` — source definition per (node, species) pair.
- `MsxSimulationResult` — time series `(T × N × S)` node_quality and `(T × L × S)` link_quality;
  helper methods `node_concentrations()`, `link_concentrations()`, `to_dataframe()`.

**`MsxSimulation` — orchestrator (layer 3)**
- High-level interface: `load()` → `run()` → `MsxSimulationResult`.
- Context manager (`with MsxSimulation(...) as sim:`).
- Write helpers: `update_initial_quality()`, `configure_source()`, `update_constant()`,
  `add_time_pattern()`.
- `hyd_file` parameter for reusing a previously saved hydraulics file.

**Convenience function `run_msx(inp, msx)`**
- Full simulation in a single call; returns `MsxSimulationResult`.

**Relationship to existing `msx.py`:**
- `msx.py` (`MsxReactionSystem`) — pure-Python ODE solver; plugs in via `geochem=` in `NzingaFlowSolver`.
  No native library required; suited for Lagrangian transport with custom reaction expressions.
- `msxlibrary.py` (`MsxSimulation`) — direct bridge to the official EPANET-MSX C solver;
  reads `.msx` files unchanged; requires `libepanetmsx.so` / `epanetmsx.dll`.
  Suited when an existing `.msx` file must be used or when the MSX solver handles time integration.

**New exports in `nzingaflow/__init__.py`:**
`MsxNativeLib`, `MsxNetworkState`, `MsxSimulation`, `MsxSimulationResult`,
`MsxSpecies`, `MsxSourceRecord`, `MsxError`, `run_msx`.

---

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
- Expressions as Python strings or direct callables; strings compiled via built-in parser (no sympy/eval needed)
- Five numerical ODE solvers: `euler`, `rk4`, `ros2` (stiff, no scipy), `rk45` (scipy), `radau` (scipy, legacy)
- `Av = 4/D [m²/m³]` automatically available in all pipe expressions
- Per-pipe parameter override via `set_pipe_param(pipe_idx, **kwargs)`
- Fully compatible with the `geochem` interface of `NzingaFlowSolver`

**Built-in expression parser** (based on EPANET-MSX `mathexpr.c`, Rossman/Shang/Uber — US EPA):
- Tokenizer + postfix evaluator; no external dependencies for string expressions
- Supported: `+ - * / ^ ()` and functions `abs sgn sqrt exp log log10 sin cos tan cot asin acos atan acot sinh cosh tanh coth step`
- Compile-time validation: unknown variable names raise `ValueError` with a clear message

**ROS2 stiff solver** (based on EPANET-MSX `ros2.c`, Verwer et al. 1999 / Rossman — US EPA):
- Rosenbrock 2(1) with adaptive step size control
- Jacobian via finite differences; LU decomposition via `numpy.linalg.solve`
- Recommended for stiff systems (chloramine kinetics, arsenic adsorption) — no scipy required
- Replaces `radau` as default for stiff pre-configured models

**Ready-made models:**
- `chloramine_decay_msx(k_f, k_ox, solver='ros2')` — HOCl + NH3 → NH2Cl (Vikesland 2001); 3 species
- `chlorine_nom_msx(k_bulk, k_wall, solver='rk4')` — Cl2 + NOM bulk/wall; 2 species
- `arsenic_oxidation_msx(Ka, Kb, K1, K2, Smax, solver='ros2')` — AS3→AS5 + wall adsorption (Zhang 2004); 3 bulk + 1 wall species

**Backward compatibility:** fully preserved. All new parameters have defaults that
reproduce the v1.0.0 behaviour. `solver='radau'` still works for existing code.


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
