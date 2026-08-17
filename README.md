🇳🇱 [Nederlands](README_NL.md) &nbsp;|&nbsp; 🇬🇧 [English](README.md)
# NzingaFlow

**Lagrangian Transport Water Quality Simulator for EPANET Networks**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.24-orange)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.2.2-informational)](CHANGELOG.md)

NzingaFlow simulates water quality in drinking water distribution networks using the **Lagrangian Transport Approach (LTA)**. Chemical species are transported as discrete segments carried along with the flow — eliminating the numerical diffusion inherent to Eulerian methods. All core calculations are fully vectorized with NumPy and optionally accelerated with Numba JIT compilation.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
  - [NzingaFlowSolver](#nzingaflowsolver)
  - [EPSRunner](#epsrunner)
  - [HydraulicModel](#hydraulicmodel)
  - [SegmentStore](#segmentstore)
  - [LTA Core Functions](#lta-core-functions)
  - [GeochemSolver & SpeciesMap](#geochemsolver--speciesmap)
  - [MsxReactionSystem](#msxreactionsystem)
  - [MsxSimulation & MsxNativeLib](#msxsimulation--msxnativelib)
  - [Stability Functions](#stability-functions)
  - [merge\_segments](#merge_segments)
- [Advanced Usage](#advanced-usage)
- [Wall Reactions](#wall-reactions)
- [FAQ](#faq)
- [Glossary](#glossary)

---

## Features

- **Vectorized LTA solver** — no Python loops over segments; all calculations via NumPy
- **Numba JIT kernels** — optional ~4–5× speedup via `pip install numba`
- **Multi-species** — multiple species simultaneously in a single `C` matrix `(n_segments × n_species)`
- **Combined decay** — bulk and wall decay fused into one memory pass (`combined_decay_multi`)
- **Wall reactions** — two-film model per pipe with three-regime Sherwood correlation (v1.1.0)
- **Temperature correction** — Arrhenius/Hayduk-Laudie correction on D_mol and k_wall (v1.1.0)
- **Leakage modelling** — proportional volume loss per segment without contaminant ingress (v1.1.0)
- **MSX reaction system** — EPANET-MSX 2.0 compatible multi-species reaction layer (v1.1.0)
- **MSX native library bridge** — direct ctypes binding to `libepanetmsx` (EPANET-MSX 2.0); runs an existing `.msx` file unchanged through the official C solver; native libraries bundled for Linux/Windows (v1.2.2)
- **Valve topology support** — PRV, PSV, TCV, FCV, GPV and PCV included in transport via `include_valves=True` (v1.2.1)
- **Tanks** — CSTR model with implicit Euler integration (unconditionally stable)
- **Extended Period Simulation (EPS)** — automatic hydraulic updates every `hyd_dt` seconds
- **Full geochemistry** — optional PhreeqPython/PHREEQC integration
- **CFL stability check** — automatic warning and `recommended_dt()`
- **Mass balance tracking** — via `MassBalanceTracker` (opt-in)
- **Flow reversal** — detection and correct segment mirroring
- **Pre-allocated buffers** — zero heap allocations in the hot path
- **CSR pipe-index** — cached sorted index in `SegmentStore`; lexsort skipped when valid

### Performance Benchmarks

| Operation | Scale | Without Numba | With Numba |
|---|---|---|---|
| `combined_decay_multi()` | 5,000 segs, 200 pipes, 2 species | ~63 µs | ~12 µs |
| `advect()` | 5,000 segs | ~15 µs | ~4 µs |
| `node_mixing_multi()` | 5,000 segs, 150 nodes | ~32 µs | ~8 µs |
| `merge_segments()` | 5,000 segs, 2 species | ~2,200 µs | ~90 µs |
| **Full time step** | 5,000 segs, 200 pipes, 2 species | **~280 µs** | **~60–90 µs** |

---

## Installation

### From PyPI

```bash
# Minimal (NumPy only)
pip install nzingaflow

# With Numba JIT (~4–5× faster)
pip install "nzingaflow[numba]"

# With full geochemistry (PhreeqPython/PHREEQC)
pip install "nzingaflow[geochem]"

# Everything including dev tools
pip install "nzingaflow[all]"
```

### From source

```bash
git clone https://github.com/nzingaflow/nzingaflow.git
cd nzingaflow
pip install -e ".[dev]"
```

> **Requirements:** Python ≥ 3.10, NumPy ≥ 1.24, epynet ≥ 2.0

---

## Quick Start

### Simple simulation (single species)

```python
import numpy as np
from nzingaflow import NzingaFlowSolver, EPSRunner

solver = NzingaFlowSolver("network.inp", n_species=1)

# Optional: trigger Numba JIT compilation before the simulation (~1 s once)
solver.warmup_numba()

runner = EPSRunner(
    solver,
    qual_dt=5.0,        # water quality time step [s]
    hyd_dt=300.0,       # hydraulic update interval [s]
    duration=86400.0,   # total simulation duration [s]
)

results = runner.run(
    decay_k=np.array([0.0003]),              # bulk decay constant [1/s]
    inject_schedule={
        "R1": [(0, 86400, np.array([1.0]))]  # 1 mg/L for 24 hours
    },
    verbose=True,
)
# results.shape == (n_steps, node_count, 1)
```

### Multi-species

```python
solver = NzingaFlowSolver("network.inp", n_species=2)
solver.warmup_numba()   # optional: trigger JIT compilation before the simulation
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)

results = runner.run(
    decay_k=np.array([0.0003, 0.0]),         # chlorine decays; tracer is conservative
    inject_schedule={
        "R1": [(0, 86400, np.array([1.0, 100.0]))]
    },
)
# results[:, :, 0]  →  chlorine per node per time step
# results[:, :, 1]  →  tracer per node per time step
```

### With wall reactions

```python
n_pipes = len(solver.pipe_ids)
k_wall = np.full((n_pipes, 1), 1e-5)   # [m/s] — uniform across all pipes

solver = NzingaFlowSolver("network.inp", n_species=1, k_wall=k_wall)
solver.warmup_numba()   # optional: trigger JIT compilation before the simulation
```

### With temperature correction and leakage (new in v1.1.0)

```python
solver = NzingaFlowSolver(
    "network.inp",
    n_species=1,
    k_wall=np.full((n_pipes, 1), 1e-5),
    temperature=12.0,          # [°C] — activates Arrhenius/Hayduk-Laudie correction
    leakage_fraction=0.12,     # 12% pipe loss (typical for distribution networks)
)
solver.warmup_numba()   # optional: trigger JIT compilation before the simulation
```

### With MSX multi-species reactions (new in v1.1.0)

```python
from nzingaflow.msx import chloramine_decay_msx

rxn = chloramine_decay_msx(k_f=2.5e-4, k_ox=5e-5, solver='ros2')

solver = NzingaFlowSolver(
    "network.inp",
    n_species=len(rxn.bulk_species),  # 3: HOCl, NH3, NH2Cl
    geochem=rxn,
)
solver.warmup_numba()   # optional: trigger JIT compilation before the simulation
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(decay_k=np.zeros(3))
```

### With the EPANET-MSX native library bridge (new in v1.2.2)

Use this to run an existing `.msx` file unchanged through the official MSX C solver,
without rewriting reaction expressions in Python.

```python
from nzingaflow import MsxSimulation, run_msx

# Simplest usage: everything in one call
result = run_msx("network.inp", "network.msx")
df = result.to_dataframe("CL2", element="node")   # DataFrame: time [h] x node names

# More control via context manager
with MsxSimulation("network.inp", "network.msx") as sim:
    state = sim.load()
    print([s.name for s in state.bulk_species()])

    sim.update_initial_quality(node_values={"R1": {"CL2": 1.0}})
    sim.configure_source("R1", "CL2", kind="CONCEN", level=1.0)

    result = sim.run()

print(result.time_hours())                   # time axis [h]
print(result.node_concentrations("CL2"))     # array (T x N)
```

> Native library (`libepanetmsx`/`epanet2`) ships bundled in `nzingaflow/lib/` —
> Linux x86-64 and Windows x86-64 included. macOS is not (yet) built; build it
> yourself via EPANETMSX's own `CMakeLists.txt` (supports Linux/macOS/Windows).

### With geochemistry (PhreeqPython)

```python
from nzingaflow import NzingaFlowSolver, EPSRunner
from nzingaflow.geochemistry import full_water_chemistry

geo = full_water_chemistry()
solver = NzingaFlowSolver("network.inp", n_species=6, geochem=geo)
solver.warmup_numba()   # optional: trigger JIT compilation before the simulation
runner = EPSRunner(solver, qual_dt=10.0, hyd_dt=300.0, duration=86400.0)

# Species: [Cl2, pH, Alk, Ca, Fe, Mn]
results = runner.run(
    decay_k=np.zeros(6),
    inject_schedule={
        "R1": [(0, 86400, np.array([1.0, 7.5, 2.5, 60.0, 0.1, 0.05]))]
    },
)
```

---

## Architecture

### LTA Principle

Each time step `dt`, the solver executes the following phases:

```
1. Combined decay  C *= combined_exp[pipe]   where combined_exp[p,s] = exp(-(k_bulk[s] + k_wall_vol[p,s]) * dt)
2. Advection       x += v[pipe] * dt
3. Exit detection  x >= L[pipe]  →  segment leaves pipe
4. Node mixing     flow-weighted  →  node_C (n_nodes × n_species)
5. Tank model      CSTR implicit Euler  →  node_C updated
6. Pipe routing    vectorized: through-nodes updated in one pass; splits in small loop
7. Merging         adjacent segs with |ΔC| < tol  →  merged (every merge_interval steps)
```

Steps 1–4 are executed by Numba JIT kernels when Numba is installed, with no intermediate array allocations.

### Module Overview

| Module | Class / Function | Responsibility |
|---|---|---|
| `solver.py` | `NzingaFlowSolver` | Full LTA solver; integrates all components |
| `eps.py` | `EPSRunner` | EPS time loop, injection, progress reporting |
| `hydraulics.py` | `HydraulicModel` | EPANET coupling via epynet |
| `segments.py` | `SegmentStore` | Structure-of-Arrays storage + CSR pipe-index |
| `lta.py` | `combined_decay_multi`, `advect`, … | Vectorized core calculations with Numba JIT |
| `merging.py` | `merge_segments` | Parallel-reduction segment merging with Numba JIT |
| `stability.py` | `recommended_dt`, `MassBalanceTracker` | CFL check and mass balance |
| `geochemistry.py` | `GeochemSolver`, `SpeciesMap` | PhreeqPython integration |
| `msx.py` | `MsxReactionSystem` | Pure-Python MSX reaction layer: ODE solvers + expression parser (v1.1.0) |
| `msxlibrary.py` | `MsxSimulation`, `MsxNativeLib` | Direct ctypes bridge to `libepanetmsx` (EPANET-MSX 2.0); loads `.msx` files unchanged; native libs bundled (v1.2.2) |
| `units.py` | — | EPANET unit-system conversions and enums shared by `hydraulics.py` and `parse_inp.py` |

### SegmentStore (Structure-of-Arrays)

```
pipe   [int32,   capacity]          pipe index per segment
x      [float64, capacity]          position along pipe [m]
C      [float64, capacity × n_sp]   concentrations per species
volume [float64, capacity]          segment volume [m³]
n                                   number of active segments
```

Pre-allocation avoids heap allocations during the simulation. Capacity overflow triggers automatic doubling with an `on_resize` callback that keeps all solver buffers in sync. A lazy CSR pipe-index (`build_csr()`) caches the sorted-by-pipe order used by `merge_segments`; the sort is skipped when the index is still valid.

---

## API Reference

### NzingaFlowSolver

```python
NzingaFlowSolver(
    inp_path:         str,
    n_species:        int   = 1,
    capacity:         int   = 200_000,
    k_wall:           ndarray | None = None,   # (n_pipes, n_species) or (n_species,) [m/s]
    D_mol:            float = 1.3e-9,          # molecular diffusivity [m²/s]
    nu:               float = 1e-6,            # kinematic viscosity [m²/s]
    track_mass:       bool  = False,
    geochem                 = None,            # GeochemSolver or MsxReactionSystem instance
    temperature:      float | None = None,     # water temperature [°C] (v1.1.0)
    leakage_fraction: float = 0.0,             # fractional pipe flow loss (v1.1.0)
    wall_mode:        str   = 'two_film',      # 'two_film' or 'direct' (v1.1.0)
    include_valves:   bool  = False,           # include valves in transport topology (v1.2.1)
)
```

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `inp_path` | `str` | Path to the EPANET `.inp` file |
| `n_species` | `int` | Number of chemical species to simulate |
| `capacity` | `int` | Initial SegmentStore capacity (auto-resize) |
| `k_wall` | `ndarray \| None` | Wall reaction rate [m/s]; `None` = no wall reactions |
| `D_mol` | `float` | Molecular diffusivity [m²/s] |
| `nu` | `float` | Kinematic viscosity [m²/s] |
| `track_mass` | `bool` | Enable mass balance tracking |
| `geochem` | `GeochemSolver \| MsxReactionSystem \| None` | Replaces first-order bulk decay with full geochemistry or MSX reactions |
| `temperature` | `float \| None` | Water temperature [°C]. Activates Arrhenius correction on D_mol (`Ea≈17 kJ/mol`) and θ=1.047 correction on k_wall (Rossman 2000). `None` = Rossman 1994-compatible (no correction). *(v1.1.0)* |
| `leakage_fraction` | `float` | Fraction of pipe flow lost as leakage (0.0–1.0). Each segment loses proportional volume per timestep; concentration remains constant. Typically 0.05–0.20 for distribution networks. *(v1.1.0)* |
| `wall_mode` | `str` | `'two_film'` (default): EPANET-compatible series resistance `k_eff = k_f·k_w/(k_f+k_w)`. `'direct'`: k_wall is already k_eff — use when k_wall comes from direct calibration rather than EPANET. *(v1.1.0)* |
| `include_valves` | `bool` | When `True`, valves (PRV, PSV, PBV, FCV, TCV, GPV, PCV) are included in the transport topology. EPANET always solves hydraulics for valves correctly; this option ensures NzingaFlow also computes residence time and species transport across them. Default `False` for backward compatibility. *(v1.2.1)* |

**Methods**

| Method | Return type | Description |
|---|---|---|
| `step(dt, decay_k, merge_interval=10, merge_tol=1e-6, check_cfl=False)` | `ndarray (node_count, n_species)` | Execute one water quality time step |
| `update_hydraulics(simtime=0)` | `None` | Recompute hydraulics; detects flow reversals and rebuilds routing caches |
| `warmup_numba()` | `None` | Trigger Numba JIT compilation before the simulation (call once after init) |
| `inject(node_uid, C_vector, volume)` | `None` | Inject from a node, distributed proportionally over outgoing pipes |
| `inject_pipe(pipe_uid, C_vector, volume, x=0.0)` | `None` | Inject directly into a pipe at position `x` [m] |
| `booster_inject(node_uid, C_set, flow_frac=1.0)` | `None` | Fix concentration at a set value on outgoing pipes |
| `check_stability(dt, decay_k, warn=True)` | `dict` | CFL and reaction stability check |
| `recommended_dt(decay_k)` | `float` | Maximum stable time step [s] |
| `mass_balance()` | `dict \| None` | Mass balance report (requires `track_mass=True`) |
| `geochem_report(C_vec)` | `dict` | LSI, saturation indices, pH (requires `geochem`) |

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `node_count` | `int` | Number of nodes |
| `n_species` | `int` | Number of species |
| `node_index` | `dict[str, int]` | Name → 0-based index |
| `pipe_index` | `dict[str, int]` | EPANET name → pipe index |
| `node_names` | `list[str]` | Node names in index order |
| `pipe_ids` | `list[str]` | Pipe names in index order |
| `pipe_length` | `ndarray (n_pipes,)` | Pipe lengths [m] |
| `pipe_area` | `ndarray (n_pipes,)` | Cross-sectional areas [m²] |
| `segments` | `SegmentStore` | Active segments |

---

### EPSRunner

```python
EPSRunner(
    solver:   NzingaFlowSolver,
    qual_dt:  float,   # water quality time step [s]
    hyd_dt:   float,   # hydraulic update interval [s]
    duration: float,   # total simulation duration [s]
)
```

#### `EPSRunner.run()`

```python
run(
    decay_k,                            # (n_species,) bulk [1/s]
    inject_schedule:  dict  = None,     # {node: [(t_start, t_end, C_vec), ...]}
    inject_fn:        callable = None,  # dynamic injection callback
    booster_schedule: dict  = None,     # {node: [(t_start, t_end, C_set), ...]}
    merge_interval:   int   = 10,
    merge_tol:        float = 1e-6,
    check_cfl:        bool  = True,
    verbose:          bool  = False,
) -> ndarray  # shape: (n_steps, node_count, n_species)
```

**`inject_schedule` format**

```python
inject_schedule = {
    "R1": [
        (0,    3600,  np.array([1.0])),   # 1 mg/L for the first hour
        (3600, 86400, np.array([0.5])),   # 0.5 mg/L thereafter
    ]
}
```

**`inject_fn` callback**

```python
def my_injection(t: float, solver: NzingaFlowSolver):
    if t < 1800:
        solver.inject("R1", [1.0], volume=0.05)

results = runner.run(decay_k=..., inject_fn=my_injection)
```

> `inject_fn` takes precedence over `inject_schedule` when both are provided.

#### `EPSRunner.time_axis()`

```python
t_hours = runner.time_axis(unit="h")   # "s", "min", or "h"
```

---

### HydraulicModel

Low-level EPANET coupling via epynet. Normally instantiated internally by `NzingaFlowSolver`.

```python
HydraulicModel(inp_path: str, include_pumps: bool = False, include_valves: bool = False)
```

| Method | Description |
|---|---|
| `solve(simtime=0)` | Solve hydraulics for one EPS time step [s]; memoized — re-solving the same `simtime` is a no-op |
| `close()` | Close the underlying EPANET hydraulic session (`EN_closeH`). Idempotent; a later `solve()` reopens it automatically |
| `get_topology()` | Returns `(pipe_start, pipe_end, pipe_length, pipe_area, node_count, pipe_ids, node_names)` — cached |
| `get_topology_with_reversal()` | Topology with pipe_start/end corrected for actual flow direction |
| `get_hydraulic_state()` | Returns `(flow [m³/s], velocity [m/s], reversed_mask)` — always SI units |
| `summary()` | Diagnostic overview of the network as a string |

**Session management**

`HydraulicModel` keeps its EPANET hydraulic session (`EN_openH`) open between `solve()` calls for performance. Call `close()` when done with the model (e.g. at the end of an EPS run), or use it as a context manager:

```python
with HydraulicModel("network.inp") as hm:
    hm.solve()
    ...
# hm.close() is called automatically
```

---

### SegmentStore

Structure-of-Arrays storage with fixed pre-allocation.

```python
SegmentStore(capacity: int = 100_000, n_species: int = 1)
```

| Attribute / Method | Description |
|---|---|
| `pipe[:n]` | `ndarray (capacity,) int32` — pipe index per segment |
| `x[:n]` | `ndarray (capacity,) float64` — position along pipe [m] |
| `C[:n]` | `ndarray (capacity, n_species) float64` — concentrations |
| `volume[:n]` | `ndarray (capacity,) float64` — segment volume [m³] |
| `n` | `int` — number of active segments |
| `add(pipe, x, volume, C_vector)` | Add a segment; auto-resize on capacity overflow |
| `remove(indices)` | Remove via swap-with-last in O(1) |
| `build_csr(n_pipes)` | Build or return cached CSR pipe-index `(pipe_order, pipe_ptr)` |
| `pipe_a`, `x_a`, `C_a`, `volume_a` | Properties returning views `[:n]` |
| `on_resize` | Callback `(new_capacity) → None`; called when capacity doubles |

---

### LTA Core Functions

All functions in `lta.py` are fully vectorized with NumPy and JIT-compiled with Numba when available.

```python
from nzingaflow.lta import (
    combined_decay_multi,   # bulk + wall decay fused in one pass  ← use this
    build_combined_exp,     # pre-compute combined decay factors
    bulk_first_order_multi, # bulk decay only
    wall_first_order_multi, # wall decay only
    compute_wall_k,
    advect,
    exit_detect,
    node_mixing_multi,
    tank_step_implicit,
    warmup_numba,           # trigger JIT compilation
    USE_NUMBA,              # True if Numba is available
)
```

| Function | Signature | Description |
|---|---|---|
| `build_combined_exp` | `(k_bulk, k_wall_vol, dt, n_pipes) → (n_pipes, n_species)` | Pre-compute `exp(-(k_bulk + k_wall_vol) * dt)`; call once per hydraulic update or dt change |
| `combined_decay_multi` | `(C, pipe, combined_exp, n=None) → None` | `C *= combined_exp[pipe]` in-place; one memory pass for both bulk and wall decay |
| `bulk_first_order_multi` | `(C, k_vec, dt, n=None) → None` | `C *= exp(-k_vec * dt)` in-place; broadcasts over all segments |
| `wall_first_order_multi` | `(C, pipe, k_wall, dt, n=None) → None` | Wall decay per segment in-place |
| `compute_wall_k` | `(diam, velocity, k_w, D_mol, nu) → (n_pipes, n_species) [1/s]` | Volumetric wall reaction rate via two-film model |
| `advect` | `(x, pipe, velocity, dt, n=None) → None` | `x += velocity[pipe] * dt` in-place |
| `exit_detect` | `(x, pipe, pipe_length, exit_mask, n=None) → int` | Fill `exit_mask` in-place; returns number of exiting segments |
| `node_mixing_multi` | `(exit_mask, pipe, C, flow, pipe_end, node_count, out=None, wC_buf=None, node_flow_buf=None, n=None) → ndarray` | Flow-weighted mixing; zero allocations when pre-allocated buffers are provided |
| `tank_step_implicit` | `(C_tank, Q_in, C_in, Q_out, V_tank, k_b, dt) → None` | CSTR implicit Euler in-place |
| `warmup_numba` | `(n_species=1) → None` | Trigger JIT compilation; call once after creating the solver |

> **Performance tip:** use `combined_decay_multi` together with `build_combined_exp` instead of separate `bulk_first_order_multi` + `wall_first_order_multi` calls. This halves the number of memory passes over `C` and eliminates the per-step `exp()` computation.

---

### GeochemSolver & SpeciesMap

PhreeqPython integration that replaces first-order bulk decay with full PHREEQC geochemistry.

> Requires: `pip install nzingaflow[geochem]` or `pip install phreeqpython`

#### SpeciesMap

```python
SpeciesMap(
    species_names: list[str],         # human-readable names
    phreeqc_names: list[str | None],  # PHREEQC element name; None = no coupling
    units:         list[str],         # "mg/L", "mol/L", "meq/L", "mmol/L"
    is_pH:         list[bool] = [],   # True if this species represents pH
)
```

Example:

```python
smap = SpeciesMap(
    species_names=["Cl2",  "pH",  "Alk"],
    phreeqc_names=["Cl",   None,  "Alk"],
    units        =["mg/L", "",    "meq/L"],
    is_pH        =[False,  True,  False],
)
```

Supported PHREEQC elements: `Cl`, `Ca`, `Mg`, `Na`, `K`, `Fe`, `Mn`, `N(5)` (NO₃), `N(3)` (NO₂), `N(-3)` (NH₄), `Alk`, `C(4)` (TIC/CO₂), `S(6)` (SO₄), `P` (phosphate).

#### GeochemSolver

```python
GeochemSolver(
    species_map:         SpeciesMap,
    background_solution: dict | None = None,
    kinetics_script:     str  | None = None,   # PHREEQC KINETICS block
    equilibrium_phases:  dict | None = None,   # {phase: (SI_target, amount)}
    reaction_script:     str  | None = None,   # raw PHREEQC script (advanced)
    db_path:             str  | None = None,   # path to .dat database
    batch_size:          int  = 500,
    min_C_threshold:     float = 1e-9,
)
```

| Method | Description |
|---|---|
| `apply_geochemistry(store, dt, pipe_diam=None, pipe_vel=None)` | Replaces `combined_decay_multi()`; runs PHREEQC reactions per segment in batches |
| `apply_mixing(node_C, node_flow, dt)` | Chemical equilibrium after node mixing (pH correction) |
| `make_injection_solution(C_vector)` | Convert a concentration vector to a PHREEQC solution dict |
| `langelier_index(C_vec)` | Langelier Saturation Index for CaCO₃ (> 0 = scaling, < 0 = corrosive) |
| `saturation_indices(C_vec)` | Saturation indices for Calcite, Dolomite, Goethite, and others |
| `reset()` | Reset PHREEQC instance on simulation restart |

#### Pre-configured recipes

```python
from nzingaflow.geochemistry import chlorine_decay_geochem, full_water_chemistry

# Chlorine + pH (n_species=2)
geo = chlorine_decay_geochem(k_bulk_per_day=0.5)

# Chlorine, pH, alkalinity, calcium, iron, manganese (n_species=6)
geo = full_water_chemistry()
```

---

### MsxReactionSystem

MSX-compatible multi-species reaction layer implementing the four EPANET-MSX 2.0 concepts: `RATE`, `EQUIL`, `FORMULA`, and surface species (`WALL`). Pass an `MsxReactionSystem` instance as the `geochem` argument to `NzingaFlowSolver`.

> Requires: `pip install nzingaflow` (no extra dependency; scipy required for `rk45`/`radau` solvers)

```python
from nzingaflow.msx import MsxReactionSystem

rxn = MsxReactionSystem(
    bulk_species,          # list of bulk species names
    wall_species=[],       # list of wall-bound species names
    params={},             # {name: value} reaction parameters
    pipe_rates={},         # {species: expression} in pipes
    pipe_formulas={},      # {species: expression} derived variables
    tank_rates={},         # {species: expression} in tanks
    solver='rk4',          # 'euler', 'rk4', 'ros2' (stiff, no scipy), 'rk45', 'radau'
)
```

**Expression notation**

| Symbol | Description |
|---|---|
| Species name | Bulk concentration, e.g. `Cl2`, `NH3` |
| Parameter name | Scalar constant, e.g. `k1`, `BFmax` |
| `Av` | Pipe surface area per volume [m²/m³] = 4/D — automatically available |
| `t` | Simulated time [s] |

**Numerical solvers**

| Solver | Type | scipy needed | Recommended for |
|---|---|---|---|
| `euler` | Explicit, 1st order | No | Fast; non-stiff only |
| `rk4` | Explicit, 4th order | No | Default for most systems |
| `ros2` | Rosenbrock 2(1), adaptive | **No** | **Stiff systems** (chloramine, biofilm) |
| `rk45` | Adaptive RK45 | Yes | Non-stiff; variable step size |
| `radau` | Implicit Radau IIA | Yes | Stiff (legacy; use `ros2` instead) |

**Expression parser:** string expressions are compiled by a built-in tokenizer/evaluator based on EPANET-MSX `mathexpr.c` (Rossman/Shang/Uber — US EPA NRMRL). No sympy or eval() needed. Compile-time validation raises `ValueError` for unknown variable names.

**Wall species**

Wall species (e.g. biofilm `BF`) are bound to the pipe wall and do not move with the water. They couple to bulk species via `RATE` expressions using `Av`. Each segment carries its own wall concentration, initialised to zero.

```python
rxn = MsxReactionSystem(
    bulk_species = ['Cl2', 'NH3', 'NH2Cl'],
    wall_species = ['BF'],
    params       = {'k1': 1.5e-4, 'k2': 3e-3, 'k4': 0.01,
                    'k5': 0.005, 'BFmax': 100.0},
    pipe_rates   = {
        'Cl2':   '-k1 * Cl2 * NH3 - k2 * Cl2 * BF * Av',
        'NH3':   '-k1 * Cl2 * NH3',
        'NH2Cl': 'k1 * Cl2 * NH3 - k3 * NH2Cl',
        'BF':    'k4 * NH2Cl * (BFmax - BF) - k5 * BF',
    },
    tank_rates   = {
        'Cl2':   '-k1 * Cl2 * NH3',
        'NH3':   '-k1 * Cl2 * NH3',
        'NH2Cl': 'k1 * Cl2 * NH3 - k3 * NH2Cl',
    },
    solver = 'rk4',
)
```

**Methods**

| Method | Description |
|---|---|
| `apply_geochemistry(store, dt, pipe_diam, pipe_vel)` | Integrate reactions for all segments; replaces `combined_decay_multi()` |
| `apply_mixing(node_C, node_flow, dt)` | Integrate tank reactions after node mixing |
| `get_wall_concentrations()` | Return current wall concentration array `(n_segments, n_wall_species)` |
| `set_wall_concentrations(C_wall)` | Override wall concentrations (e.g. for warm-start) |
| `set_pipe_param(pipe_idx, **kwargs)` | Override parameters for a single pipe |
| `reset()` | Reset wall concentrations to zero |

**Ready-made MSX models**

```python
from nzingaflow.msx import (
    chloramine_decay_msx,    # HOCl + NH3 → NH2Cl (Vikesland 2001); species: [HOCl, NH3, NH2Cl]
    chlorine_nom_msx,        # Cl2 + NOM bulk/wall; species: [Cl2, NOM]
    arsenic_oxidation_msx,   # AS3→AS5, wall adsorption (Zhang 2004); species: [AS3, AS5, NH2CL] + wall [AS5s]
)

rxn = chloramine_decay_msx(k_f=2.5e-4, k_ox=5e-5, solver='ros2')
rxn = chlorine_nom_msx(k_bulk=3e-4, k_wall=1e-5, solver='rk4')
rxn = arsenic_oxidation_msx(Ka=10.0, Kb=0.1, K1=5.0, K2=1.0, Smax=50.0, solver='radau')
```

---

### MsxSimulation & MsxNativeLib

Direct binding to the official EPANET-MSX C library via ctypes. Three layers on top of the `MSX_*` C API (EPANET-MSX 2.0).

> Native library (`libepanetmsx`/`epanet2`) ships bundled in `nzingaflow/lib/` —
> Linux x86-64 and Windows x86-64 included. For macOS: build it yourself via the
> `CMakeLists.txt` from [github.com/USEPA/EPANETMSX](https://github.com/USEPA/EPANETMSX)
> (supports Linux/macOS/Windows).

#### `MsxSimulation` — orchestrator (layer 3)

```python
MsxSimulation(
    inp_path:  str,           # path to the EPANET .inp file
    msx_path:  str,           # path to the EPANET-MSX .msx file
    lib_path:  str | None = None,  # explicit path to libepanetmsx; None = auto-detect
    strict:    bool = True,   # raise MsxError on non-zero return codes
)
```

**Methods**

| Method | Return type | Description |
|---|---|---|
| `load()` | `MsxNetworkState` | Open the MSX file and build a full network snapshot |
| `run(save_to_file=False, hyd_file=None)` | `MsxSimulationResult` | Run the full simulation; optionally from a previously saved hydraulics file |
| `update_initial_quality(node_values=None, link_values=None)` | `None` | Adjust initial concentrations before `run()`: `{name: {species: value}}` |
| `configure_source(node, species, kind, level, pattern_name=None)` | `None` | Configure a source; `kind` = `'CONCEN'`, `'MASS'`, `'SETPOINT'`, `'FLOWPACED'` or `'NOSOURCE'` |
| `update_constant(name, value)` | `None` | Change a named constant in the .msx file |
| `add_time_pattern(name, multipliers)` | `None` | Add a new time pattern with the given multipliers |
| `close()` | `None` | Close the library and free memory |

**Context manager**

```python
with MsxSimulation("network.inp", "network.msx") as sim:
    state = sim.load()
    result = sim.run()
# sim.close() is called automatically
```

#### `MsxSimulationResult`

```python
result.time_s                        # ndarray — timestamps [s]
result.time_hours()                  # ndarray — timestamps [h]
result.node_quality                  # ndarray (T, N, S) — node concentrations
result.link_quality                  # ndarray (T, L, S) — link concentrations
result.node_names                    # list[str]
result.link_names                    # list[str]
result.species                       # list[MsxSpecies]

result.node_concentrations("CL2")   # ndarray (T, N) — single species from node_quality
result.link_concentrations("CL2")   # ndarray (T, L) — single species from link_quality
result.to_dataframe("CL2", element="node")  # pandas DataFrame: index=time_h, columns=node names
```

#### `MsxNetworkState`

Snapshot of the network after `load()`. Contains species, constants, sources, and patterns.

```python
state.species               # list[MsxSpecies]
state.constants             # dict[str, float]
state.sources               # list[MsxSourceRecord]
state.node_initq            # ndarray (N, S) — initial node quality
state.link_initq            # ndarray (L, S) — initial link quality
state.patterns              # dict[int, list[float]]

state.species_by_name("CL2")   # → MsxSpecies
state.bulk_species()            # → list[MsxSpecies]
state.wall_species()            # → list[MsxSpecies]
```

#### `MsxNativeLib` — ctypes wrapper (layer 1)

For advanced use: direct access to all `MSX_*` C functions.

```python
from nzingaflow import MsxNativeLib

lib = MsxNativeLib(lib_path=None, strict=True)
lib.open("network.msx")
lib.solve_hydraulics()
lib.solve_quality()

n_sp = lib.object_count(3)               # ObjectType.SPECIES = 3
name = lib.object_id(3, 1)              # name of species 1
conc = lib.concentration(0, 1, 1)       # NODE=0, node 1, species 1

lib.set_constant(idx, value)
lib.set_initial_quality(obj_type, idx, sp_idx, value)
lib.set_source(node, species, type, level, pattern)
lib.close()
```

#### Which one: `msx.py` or `msxlibrary.py`?

| Situation | Recommended module |
|---|---|
| Writing reaction expressions yourself in Python | `msx.py` — `MsxReactionSystem` |
| Reusing an existing `.msx` file | `msxlibrary.py` — `MsxSimulation` |
| No native library available | `msx.py` (no dependency) |
| The MSX C solver should handle time integration | `msxlibrary.py` |
| Coupling with `NzingaFlowSolver` via `geochem=` | `msx.py` — `MsxReactionSystem` |
| Standalone MSX simulation without LTA | `msxlibrary.py` — `run_msx()` |

---

### Stability Functions

```python
from nzingaflow import recommended_dt, check_dt
```

#### `recommended_dt()`

```python
recommended_dt(
    pipe_length:  ndarray,          # (n_pipes,) [m]
    velocity:     ndarray,          # (n_pipes,) [m/s]
    k_bulk:       ndarray,          # (n_species,) [1/s]
    k_wall_vol:   ndarray | None,   # (n_pipes, n_species) [1/s]
    safety:       float = 0.9,
    min_dt:       float = 0.1,
    max_dt:       float = 3600.0,
) -> float
```

Computes `min(dt_CFL, dt_rxn) × safety` where:
- **CFL:** `dt_CFL = min(L / v)` over all pipes
- **Reaction:** `dt_rxn = 0.1 / k_max` (10× safety margin)

#### `check_dt()` — returns a report dict

```python
report = solver.check_stability(dt=5.0, decay_k=np.array([0.001]))
# {
#   "cfl_ok":      bool,
#   "reaction_ok": bool,
#   "stable":      bool,
#   "dt_rec":      float,
#   "cfl_ratio":   float,   # dt / dt_CFL  (< 1 is safe)
#   "violations":  list[str],
# }
```

#### `MassBalanceTracker`

Enable with `track_mass=True`:

```python
solver = NzingaFlowSolver("network.inp", n_species=1, track_mass=True)
# ... run simulation ...
mb = solver.mass_balance()
# {
#   "injected":      ndarray,   # mass injected per species
#   "in_system":     ndarray,   # mass currently in segments
#   "bulk_decay":    ndarray,   # cumulative bulk decay loss
#   "wall_decay":    ndarray,   # cumulative wall decay loss
#   "outflow":       ndarray,   # mass that left via sink nodes
#   "balance_error": ndarray,   # relative error per species
#   "ok":            bool,      # True if error < 1%
# }
```

---

### `merge_segments()`

```python
from nzingaflow import merge_segments

n_removed = merge_segments(
    store,               # SegmentStore
    tol=1e-6,            # concentration tolerance [mg/L]
    min_volume=1e-12,    # segments smaller than this are always removed [m³]
    n_pipes=0,           # pipe count for CSR index (0 = auto-detect)
)
```

Algorithm: iterative parallel reduction (O(n log n)); Numba JIT kernels with early-exit candidate detection. Called automatically by the solver every `merge_interval` steps.

---

## Advanced Usage

### Numba warmup

```python
solver = NzingaFlowSolver("network.inp", n_species=2)
solver.warmup_numba()   # ~1 s once; subsequent steps are fast

runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(decay_k=np.array([0.001, 0.0]))
```

Or use the standalone function:

```python
from nzingaflow import warmup_numba, warmup_numba_merging, USE_NUMBA

print(f"Numba available: {USE_NUMBA}")
warmup_numba(n_species=2)
warmup_numba_merging(n_species=2)
```

### Time-varying injection with a callback

```python
def injection(t: float, solver: NzingaFlowSolver):
    """Inject only when flow at R1 exceeds 10 L/s."""
    flow, _ = solver._get_hydraulics()
    out_pipes = solver._node_outpipes[solver.node_index["R1"]]
    q_tot = sum(flow[p] for p in out_pipes)
    if q_tot > 0.01:
        solver.inject("R1", C_vector=[1.0], volume=q_tot * 5.0)

results = runner.run(decay_k=np.array([0.0003]), inject_fn=injection)
```

### Booster stations

```python
booster_schedule = {
    "B1": [(6*3600, 18*3600, np.array([0.8]))]  # 0.8 mg/L from hour 6 to 18
}

results = runner.run(
    decay_k=np.array([0.0003]),
    inject_schedule={"R1": [(0, 86400, np.array([1.0]))]},
    booster_schedule=booster_schedule,
)
```

### Determining the recommended time step

```python
solver = NzingaFlowSolver("network.inp", n_species=1)
dt_opt = solver.recommended_dt(decay_k=np.array([0.001]))
print(f"Recommended time step: {dt_opt:.1f} s")

runner = EPSRunner(solver, qual_dt=dt_opt, hyd_dt=300.0, duration=86400.0)
```

### Geochemical report

```python
C_sample = np.array([0.5, 7.8, 3.0, 75.0, 0.05, 0.02])
report = solver.geochem_report(C_sample)

print(f"pH:  {report['pH']:.2f}")
print(f"LSI: {report['lsi']:.3f}  (>0 = scaling, <0 = corrosive)")
for mineral, si in report['saturation_indices'].items():
    print(f"  {mineral}: SI = {si:.3f}")
```

### Mass balance verification

```python
solver = NzingaFlowSolver("network.inp", n_species=1, track_mass=True)
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=3600.0)
results = runner.run(
    decay_k=np.array([0.001]),
    inject_schedule={"R1": [(0, 3600, np.array([1.0]))]},
)

mb = solver.mass_balance()
print(f"Injected:    {mb['injected'][0]:.4f} kg")
print(f"In system:   {mb['in_system'][0]:.4f} kg")
print(f"Bulk decay:  {mb['bulk_decay'][0]:.4f} kg")
print(f"Outflow:     {mb['outflow'][0]:.4f} kg")
print(f"Error:       {mb['balance_error'][0]*100:.3f}%  OK={mb['ok']}")
```

### Direct injection into a pipe

```python
solver.inject_pipe(
    pipe_uid="P100",
    C_vector=np.array([2.0]),
    volume=0.01,
    x=solver.pipe_length[solver.pipe_index["P100"]] / 2,  # midpoint of pipe
)
```

### Low-level: using combined decay directly

```python
from nzingaflow.lta import build_combined_exp, combined_decay_multi

# Build once per hydraulic update (or when dt changes)
cexp = build_combined_exp(
    k_bulk=np.array([0.001, 0.0]),
    k_wall_vol=solver._k_wall_vol,   # (n_pipes, n_species) [1/s]
    dt=5.0,
    n_pipes=len(solver.pipe_ids),
)

# Apply every time step — one memory pass, no exp() call
combined_decay_multi(solver.segments.C, solver.segments.pipe, cexp,
                     n=solver.segments.n)
```

---

## Wall Reactions

NzingaFlow implements the two-film model from Rossman (1994), extended in v1.1.0 with a three-regime Sherwood correlation, temperature dependence, and stagnation correction.

### Film transport model

```
Re    = v · D / ν
Sh    = 0.023 · Re^0.83 · Sc^(1/3)                                (turbulent, Re > 4000)
Sh    = (3.66³ + max(1.615·(Re·Sc·D/L)^(1/3) − 0.7, 0)³)^(1/3)  (laminar, Re < 2300)
Sh    = linear interpolation Sh_lam ↔ Sh_turb                     (transition, 2300 ≤ Re ≤ 4000)
k_f   = max(Sh · D_mol / D, 4·D_mol/D)   [m/s]   (stagnation floor)
k_eff = k_f · k_w / (k_f + k_w)                  (series resistance, wall_mode='two_film')
k_vol = k_eff · 4 / D                    [1/s]    (cylinder A/V = 4/D)
```

**Changes from v1.0.0:** the previous hard threshold at Re=4000 caused a factor 1.5× error at night-time flows in DN100 pipes. The stagnation floor prevents underestimation at near-zero velocity.

### Temperature dependence (optional, v1.1.0)

Pass `temperature` [°C] to `NzingaFlowSolver` to activate:

```
D_mol(T) = D_mol_20 · exp(17000/R · (1/T₀ − 1/T))   [Hayduk & Laudie 1974]
k_wall(T) = k_wall · 1.047^(T−20)                    [Rossman 2000]
```

Effect: factor 0.59× at 5°C to 1.38× at 30°C on k_wall_vol.

### Specifying k_wall per pipe

```python
n_pipes = len(solver.pipe_ids)

# Uniform
k_wall = np.full((n_pipes, 1), 5e-6)   # [m/s]

# Variable per pipe (e.g. based on pipe material)
k_wall = np.zeros((n_pipes, 1))
k_wall[0:20] = 1e-5   # cast iron
k_wall[20:]  = 2e-6   # PVC

solver = NzingaFlowSolver("network.inp", n_species=1, k_wall=k_wall)

# 'direct' mode: k_wall is already the effective rate (no film resistance)
solver = NzingaFlowSolver("network.inp", n_species=1, k_wall=k_wall, wall_mode='direct')
```

> `k_wall = 0` disables wall reactions regardless of flow velocity.

---

## FAQ

**Q: The simulation raises a CFL warning. What should I do?**

Use `solver.recommended_dt()` to compute the optimal time step and pass it as `qual_dt` to `EPSRunner`.

**Q: The mass balance error exceeds 1%. What are possible causes?**

- The time step is too large (CFL violation)
- Wall reactions are active — the mass balance estimate is an approximation for wall decay
- Geochemistry is active: non-linear reactions are not tracked by the linear mass balance

**Q: How do I get the best performance?**

1. Install Numba: `pip install nzingaflow[numba]`
2. Call `solver.warmup_numba()` once after creating the solver, before the simulation loop
3. Use `combined_decay_multi` + `build_combined_exp` instead of separate bulk/wall calls (done automatically by the solver)

**Q: How do I add a new PHREEQC element?**

```python
smap = SpeciesMap(
    species_names=["Cl2","pH","Alk","Ca","Fe","Mn","SO4"],
    phreeqc_names=["Cl", None,"Alk","Ca","Fe","Mn","S(6)"],
    units        =["mg/L","","meq/L","mg/L","mg/L","mg/L","mg/L"],
    is_pH        =[False,True,False,False,False,False,False],
)
```

**Q: When should I use MsxReactionSystem instead of GeochemSolver?**

Use `MsxReactionSystem` when your reactions can be expressed as ordinary differential equations (first- or higher-order kinetics, biofilm growth, chloramine decay). It is faster than PhreeqPython and requires no extra dependencies. Use `GeochemSolver` when you need full thermodynamic equilibrium, mineral dissolution/precipitation, or pH-buffering via PHREEQC.

**Q: When should I use MsxSimulation (msxlibrary) instead of MsxReactionSystem (msx.py)?**

Use `MsxSimulation` when you want to run an existing `.msx` file unchanged through the official EPANET-MSX C solver — useful when the reaction equations already live in a `.msx` file, or when you want to compare results against the reference MSX implementation. `MsxSimulation` works as a standalone simulator and does not couple with `NzingaFlowSolver`.

Use `MsxReactionSystem` when you want to define reactions in Python and couple them to the NzingaFlow Lagrangian solver via `geochem=rxn`. This requires no native library and is more flexible for parameter studies.

**Q: How do I model pipe leakage?**

Pass `leakage_fraction` to `NzingaFlowSolver`. A value of `0.12` means 12% of the pipe flow is lost. Each segment loses volume proportionally per timestep; concentrations are unchanged (conservative mixing model — no contaminant ingress from groundwater). Typical values for Dutch distribution networks are 0.05–0.20.

**Q: Are pumps simulated?**

Pumps are excluded by default (`include_pumps=False`). You can include them via `HydraulicModel("net.inp", include_pumps=True)`, though this is only meaningful when residence time inside the pump is relevant.

**Q: Are valves included in the transport calculation?**

Not by default (`include_valves=False`). EPANET always solves hydraulics for valves correctly, but NzingaFlow traditionally excluded them from the Lagrangian transport topology. With `include_valves=True` all EPANET valve types (PRV, PSV, PBV, FCV, TCV, GPV, PCV) are treated as short pipe elements, so residence time and species transport are computed across them as well. This is particularly relevant when a valve is the sole connection between two network segments.

```python
solver = NzingaFlowSolver("network.inp", include_valves=True)
```

**Q: How do I interpret the Langelier Saturation Index (LSI)?**

| LSI | Meaning | Risk |
|---|---|---|
| > +0.5 | Strong CaCO₃ precipitation tendency | Scaling, blockage |
| 0 to +0.5 | Slight supersaturation | Protective calcium carbonate layer |
| −0.5 to 0 | Slight undersaturation | Low corrosion risk |
| < −0.5 | Strong undersaturation | Corrosion of metal pipes |

---

## Glossary

| Term | Description |
|---|---|
| **CFL** | Courant-Friedrichs-Lewy condition: stability requirement `dt ≤ L/v` for advection |
| **CSTR** | Continuously Stirred Tank Reactor: perfect mixing tank model |
| **CSR** | Compressed Sparse Row: index format used by `SegmentStore.build_csr()` to cache per-pipe segment order |
| **EPS** | Extended Period Simulation: time-varying simulation with periodically updated hydraulics |
| **LTA** | Lagrangian Transport Approach: transport described in the reference frame of the fluid |
| **LSI** | Langelier Saturation Index: measure of CaCO₃ saturation in water |
| **MSX** | Multi-Species eXtension: EPANET-MSX compatible reaction layer for arbitrary kinetic systems |
| **MSX native bridge** | Direct ctypes binding to `libepanetmsx`; implemented in `msxlibrary.py` |
| **PHREEQC** | Geochemical modelling software by the USGS |
| **SoA** | Structure of Arrays: memory layout where each property occupies a separate array |
| **`k_bulk`** | Bulk decay constant [1/s]: first-order decay in the water column |
| **`k_wall`** | Wall reaction rate [m/s]: chemical reaction at the inner pipe surface |
| **`combined_exp`** | Combined decay factor table `(n_pipes × n_species)` = `exp(-(k_bulk + k_wall_vol) × dt)`; pre-computed by `build_combined_exp()` |
| **`leakage_fraction`** | Fraction of pipe flow lost as leakage per timestep; segments shrink without concentration change |
| **`temperature`** | Water temperature [°C] used for Arrhenius/Hayduk-Laudie corrections on diffusivity and k_wall |
| **segment** | Lagrangian fluid parcel with fixed volume, position, and concentration vector |

---

## Changelog

### Unreleased

- **Hydraulics refactor + `units.py`.** `HydraulicModel` internals reworked around `Topology`/`HydraulicState` dataclasses, with a cached topology/pipe list and a read-once hydraulic snapshot after `solve()`. Public API unchanged except two additions: `close()` and context-manager support (`with HydraulicModel(...) as hm:`) for explicit EPANET session cleanup (`EN_closeH`).
- Replaced the internal hydraulic solver with a session-reusing implementation (`EN_openH()` stays open across `solve()` calls; `EN_INITFLOW` used for correct cold-start behaviour), improving EPS performance.
- New module `nzingaflow/units.py`: centralises EPANET unit-system conversions and enums (fixes AFD/MLD and diameter/velocity US↔SI conversion bugs), now shared by `hydraulics.py` and `parse_inp.py`.
- `parse_inp.py` updated to use typed node/link classes and the corrected unit conversions.
- New tests: `tests/gen_grid_network.py`; extended `tests/test_epynet_networks.py` with session-reuse/regression and memoization checks.

### 1.2.2

- **MSX native library bridge — bundled binaries + critical bugfixes.** `msxlibrary.py` (`MsxSimulation`/`MsxNativeLib`) was not actually functional since its introduction in 1.2.0: no native library was available on the system, and even with one present the bridge still failed due to several underlying bugs.
- Native binaries now ship in `nzingaflow/lib/`: `libepanetmsx.so`/`libepanet2_msx.so` (Linux x86-64) and `epanetmsx.dll`/`epanet2_msx.dll` (Windows x86-64), built from the official EPANET-MSX 2.0 source (github.com/USEPA/EPANETMSX). macOS not yet built — see `nzingaflow/lib/README.txt`.
- Fixed: `MSXstep` used `c_long` instead of `c_double` for its time parameters — an ABI mismatch against the MSX 2.0 C API (potentially memory-corrupting on Windows, silently wrong time values elsewhere).
- Fixed: `_epanet_open()` was a no-op; `ENopen()` was never called before `MSXopen()`, even though MSX depends on it for the shared network state (see the official CLI reference, `msxmain.c`). The EPANET network is now actually opened via a new `en_open()`/`en_close()` binding.
- Fixed: node/link counts and names went through `MSXgetcount`/`MSXgetID`, which don't support that object type (only SPECIES/CONSTANT/PARAMETER/PATTERN) — this raised `MSX error 515` everywhere. Rerouted through the correct EPANET-layer calls (`ENgetcount`, `ENgetnodeid`, `ENgetlinkid`, `ENgetnodeindex`, `ENgetlinkindex`), with a dedicated error translator (`ENgeterror`) since EN and MSX error codes use different numbering.
- Fixed — **library name collision with epynet**: the epanet2 library that `libepanetmsx` links against shared its name (and, on Linux, its SONAME) with epynet's own bundled epanet2 library. Running both in the same process (as NzingaFlow does, since it depends on epynet) made the dynamic linker/Windows loader silently resolve to the wrong, already-loaded copy. Fixed with a unique name (`libepanet2_msx.so` / `epanet2_msx.dll`) for the companion library.
- Docs: the earlier reference to EPANET-MSX **1.1** (the source of the `MSXstep` bug) has been corrected to **2.0** throughout.
- New `tests/test_msxlibrary.py`, run against the official arsenic oxidation example network (USEPA EPANETMSX Examples/), including regression tests for the bugs listed above.
- Backward compatible: `MsxSimulation`/`MsxNativeLib`'s public API is unchanged; this is a bugfix and bundling release only.

### 1.2.1

- **Valve support in transport topology** (`include_valves` parameter): all EPANET valve types (PRV, PSV, PBV, FCV, TCV, GPV, PCV) can now be included alongside pipes in the Lagrangian transport topology via `include_valves=True` on `HydraulicModel` and `NzingaFlowSolver`. Previously NzingaFlow always excluded valves, meaning no residence time/decay was computed across valve elements, and valves that were the sole connection between two network segments caused a topological disconnection. Default `False`; backward compatible.
- **Bugfix — `include_pumps=True` crashed on `diameter`**: `HydraulicModel.get_topology()` used `lnk.diameter` without a fallback, while epynet `Pump` objects have no `diameter` static property. Any network with a real pump raised an `AttributeError` as soon as `include_pumps=True` was used. Fixed with the same `getattr(..., 0.0)` fallback already used for valve length; pumps are now treated as zero-area elements.
- Regression tests for both points added in `tests/test_epynet_networks.py` (real EPANET `.inp` networks loaded via epynet, marked `requires_epynet`).

### 1.2.0

- **New module `nzingaflow/msxlibrary.py`** — direct ctypes binding to the official EPANET-MSX C library (`libepanetmsx.so` / `epanetmsx.dll`), with three layers: `MsxNativeLib` (thin ctypes wrapper), dataclasses `MsxNetworkState`/`MsxSpecies`/`MsxSourceRecord`/`MsxSimulationResult` (structured snapshots and time series), and `MsxSimulation` (high-level orchestrator with `load()`/`run()`, context manager, and write helpers `update_initial_quality`, `configure_source`, `update_constant`, `add_time_pattern`).
- **`run_msx(inp, msx)`** — full simulation in a single call.
- Use `msx.py` (`MsxReactionSystem`) for pure-Python reactions plugged in via `geochem=`; use `msxlibrary.py` (`MsxSimulation`) to run an existing `.msx` file unchanged through the official MSX C solver.
- Automatic library detection for Windows, Linux, and macOS.
- New exports in `nzingaflow/__init__.py`: `MsxNativeLib`, `MsxNetworkState`, `MsxSimulation`, `MsxSimulationResult`, `MsxSpecies`, `MsxSourceRecord`, `MsxError`, `run_msx`.
- Backward compatible: existing `MsxReactionSystem` code is unaffected.

### 1.1.0

- **Three-regime Sherwood correlation** in `compute_wall_k()`: laminar (Graetz + entry-length), transition (linear interpolation), turbulent (Dittus-Boelter). Eliminates factor 1.5× error at night-time flows in DN100 pipes.
- **Temperature correction** (`temperature=` parameter): Arrhenius correction on D_mol (Hayduk & Laudie 1974) and θ=1.047 correction on k_wall (Rossman 2000). Effect: 0.59× at 5°C to 1.38× at 30°C.
- **Stagnation floor** in `compute_wall_k()`: minimum film transport `k_f_min = 4·D_mol/D` at near-zero velocity.
- **Leakage modelling** (`leakage_fraction=` parameter): proportional volume loss per segment per timestep without contaminant ingress.
- **`wall_mode` parameter**: `'two_film'` (default, EPANET-compatible) or `'direct'` (k_wall already k_eff).
- **`MsxReactionSystem`** (`nzingaflow.msx`): EPANET-MSX 2.0 compatible reaction layer supporting `RATE`, `EQUIL`, `FORMULA`, and wall species. Five numerical solvers: `euler`, `rk4`, `ros2` (stiff, no scipy), `rk45`, `radau`.
- **Built-in expression parser** (based on EPANET-MSX `mathexpr.c`): no sympy or eval() required; compile-time validation.
- **ROS2 stiff solver** (based on EPANET-MSX `ros2.c`, Verwer et al. 1999): adaptive Rosenbrock 2(1) without scipy dependency.
- **Ready-made MSX models**: `chloramine_decay_msx`, `chlorine_nom_msx`, `arsenic_oxidation_msx`.
- All new parameters are backward-compatible with defaults that reproduce v1.0.0 behaviour.

### 1.0.0

- Initial release
- Vectorized LTA, multi-species, wall reactions, tanks, EPS
- GeochemSolver with PhreeqPython integration
- MassBalanceTracker, CFL stability check
- Parallel-reduction segment merging (O(n log n))
- Flow reversal detection and segment mirroring
- Numba JIT kernels for all hot-path operations (~4–5× speedup)
- Combined bulk + wall decay in one memory pass (`combined_decay_multi` / `build_combined_exp`)
- Vectorized junction routing via `_node_type` / `_node_out0` cache (~734× vs Python loop)
- CSR pipe-index in `SegmentStore` (cached lexsort for `merge_segments`)
- Pre-allocated solver buffers; zero heap allocations in hot path
- `exit_detect()` as explicit pre-allocated function
- `warmup_numba()` / `warmup_numba_merging()` for predictable startup latency
- `on_resize` callback on `SegmentStore` to keep external buffers in sync
