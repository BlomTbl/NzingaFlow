🇳🇱 [Nederlands](README_NL.md) &nbsp;|&nbsp; 🇬🇧 [English](README.md)
# InzingaFlow

**Lagrangian Transport Water Quality Simulator for EPANET Networks**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.21-orange)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-informational)](CHANGELOG.md)

InzingaFlow simulates water quality in drinking water distribution networks using the **Lagrangian Transport Approach (LTA)**. Chemical species are transported as discrete segments carried along with the flow — eliminating the numerical diffusion inherent to Eulerian methods. All core calculations are fully vectorized with NumPy.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Reference](#api-reference)
  - [InzingaFlowSolver](#inzingaflowsolver)
  - [EPSRunner](#epsrunner)
  - [HydraulicModel](#hydraulicmodel)
  - [SegmentStore](#segmentstore)
  - [LTA Core Functions](#lta-core-functions)
  - [GeochemSolver & SpeciesMap](#geochemsolver--speciesmap)
  - [Stability Functions](#stability-functions)
  - [merge\_segments](#merge_segments)
- [Advanced Usage](#advanced-usage)
- [Wall Reactions](#wall-reactions)
- [FAQ](#faq)
- [Glossary](#glossary)

---

## Features

- **Vectorized LTA solver** — no Python loops over segments; all calculations via NumPy
- **Multi-species** — multiple species simultaneously in a single `C` matrix `(n_segments × n_species)`
- **Wall reactions** — two-film model per pipe (Dittus-Boelter / Rossman 1994)
- **Tanks** — CSTR model with implicit Euler integration (unconditionally stable)
- **Extended Period Simulation (EPS)** — automatic hydraulic updates every `hyd_dt` seconds
- **Full geochemistry** — optional PhreeqPython/PHREEQC integration
- **CFL stability check** — automatic warning and `recommended_dt()`
- **Mass balance tracking** — via `MassBalanceTracker` (opt-in)
- **Flow reversal** — detection and correct segment mirroring

### Performance Benchmarks

| Operation | Scale | Time |
|---|---|---|
| `apply_combined_decay()` | 50,000 segs, 6,000 pipes, 3 species | ~0.7 ms |
| `advect()` | 50,000 segs | ~0.2 ms |
| `node_mixing_multi()` | 50,000 segs, 2,500 exits, 5,000 nodes | ~0.2 ms |
| `merge_segments()` | 1,000 segs, 3 species | ~260 µs |

---

## Installation

```bash
pip install numpy epynet

# Optional: full geochemistry via PhreeqPython
pip install phreeqpython
```

> **Requirements:** Python ≥ 3.9, NumPy ≥ 1.21, epynet ≥ 2.0 (2025 release)

---

## Quick Start

### Simple simulation (single species)

```python
import numpy as np
from inzingaflow import InzingaFlowSolver, EPSRunner

solver = InzingaFlowSolver("network.inp", n_species=1)

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
solver = InzingaFlowSolver("network.inp", n_species=2)
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

solver = InzingaFlowSolver("network.inp", n_species=1, k_wall=k_wall)
```

### With geochemistry (PhreeqPython)

```python
from inzingaflow import InzingaFlowSolver, EPSRunner
from inzingaflow.geochemistry import full_water_chemistry

geo = full_water_chemistry()
solver = InzingaFlowSolver("network.inp", n_species=6, geochem=geo)
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
1. Decay          C *= exp(-(k_bulk + k_wall_vol) * dt)   [combined decay_pipe table]
2. Advection      x += v[pipe] * dt
3. Exit detection x >= L[pipe]  →  segment leaves pipe
4. Node mixing    flow-weighted  →  node_C (n_nodes × n_species)
5. Tank model     CSTR implicit Euler  →  node_C updated
6. Splitting      segment reaches junction  →  split proportionally by flow
7. Merging        adjacent segs with |ΔC| < tol  →  merged
```

### Module Overview

| Module | Class / Function | Responsibility |
|---|---|---|
| `solver.py` | `InzingaFlowSolver` | Full LTA solver; integrates all components |
| `eps.py` | `EPSRunner` | EPS time loop, injection, progress reporting |
| `hydraulics.py` | `HydraulicModel` | EPANET coupling via epynet |
| `segments.py` | `SegmentStore` | Structure-of-Arrays storage |
| `lta.py` | `advect`, `node_mixing_multi`, … | Vectorized core calculations |
| `merging.py` | `merge_segments` | Parallel-reduction segment merging |
| `stability.py` | `recommended_dt`, `MassBalanceTracker` | CFL check and mass balance |
| `geochemistry.py` | `GeochemSolver`, `SpeciesMap` | PhreeqPython integration |

### SegmentStore (Structure-of-Arrays)

```
pipe   [int32,   capacity]          pipe index per segment
x      [float64, capacity]          position along pipe [m]
C      [float64, capacity × n_sp]   concentrations per species
volume [float64, capacity]          segment volume [m³]
n                                   number of active segments
```

Pre-allocation avoids heap allocations during the simulation. Capacity overflow triggers automatic doubling.

---

## API Reference

### InzingaFlowSolver

```python
InzingaFlowSolver(
    inp_path:   str,
    n_species:  int   = 1,
    capacity:   int   = 200_000,
    k_wall:     ndarray | None = None,   # (n_pipes, n_species) or (n_species,) [m/s]
    D_mol:      float = 1.3e-9,          # molecular diffusivity [m²/s]
    nu:         float = 1e-6,            # kinematic viscosity [m²/s]
    track_mass: bool  = False,
    geochem           = None,            # GeochemSolver instance
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
| `geochem` | `GeochemSolver \| None` | Replaces first-order bulk decay with full PHREEQC geochemistry |

**Methods**

| Method | Return type | Description |
|---|---|---|
| `step(dt, decay_k, merge_interval=10, merge_tol=1e-6, check_cfl=False)` | `ndarray (node_count, n_species)` | Execute one water quality time step |
| `update_hydraulics(simtime=0)` | `None` | Recompute hydraulics; detects flow reversals automatically |
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
    solver:   InzingaFlowSolver,
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
def my_injection(t: float, solver: InzingaFlowSolver):
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

Low-level EPANET coupling via epynet. Normally instantiated internally by `InzingaFlowSolver`.

```python
HydraulicModel(inp_path: str, include_pumps: bool = False)
```

| Method | Description |
|---|---|
| `solve(simtime=0)` | Solve hydraulics for one EPS time step [s] |
| `get_topology()` | Returns `(pipe_start, pipe_end, pipe_length, pipe_area, node_count, pipe_ids, node_names)` — cached |
| `get_topology_with_reversal()` | Topology with pipe_start/end corrected for actual flow direction |
| `get_hydraulic_state()` | Returns `(flow [m³/s], velocity [m/s], reversed_mask)` — always SI units |
| `summary()` | Diagnostic overview of the network as a string |

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
| `pipe_a`, `x_a`, `C_a`, `volume_a` | Properties returning views `[:n]` |

---

### LTA Core Functions

All functions in `lta.py` are fully vectorized with NumPy.

```python
from inzingaflow.lta import (
    bulk_first_order_multi,
    wall_first_order_multi,
    combined_decay_factors,
    apply_combined_decay,
    compute_wall_k,
    advect,
    node_mixing_multi,
    tank_step_implicit,
)
```

| Function | Signature | Description |
|---|---|---|
| `bulk_first_order_multi` | `(C, k_vec, dt) → None` | `C *= exp(-k_vec * dt)` in-place; broadcasts over all segments |
| `wall_first_order_multi` | `(C, pipe, k_wall, dt) → None` | Wall decay per segment in-place |
| `combined_decay_factors` | `(k_bulk, k_wall_vol, dt, n_pipes) → (n_pipes, n_species)` | Combined decay factors — computed once per hydraulic update |
| `apply_combined_decay` | `(C, pipe, decay_pipe) → None` | `C *= decay_pipe[pipe]` in-place |
| `compute_wall_k` | `(diam, velocity, k_w, D_mol, nu) → (n_pipes, n_species) [1/s]` | Volumetric wall reaction rate via two-film model |
| `advect` | `(x, pipe, velocity, dt) → None` | `x += velocity[pipe] * dt` in-place |
| `node_mixing_multi` | `(exit_mask, pipe, C, flow, pipe_end, node_count) → (node_count, n_species)` | Flow-weighted mixing via `bincount` |
| `tank_step_implicit` | `(C_tank, Q_in, C_in, Q_out, V_tank, k_b, dt) → None` | CSTR implicit Euler in-place |

---

### GeochemSolver & SpeciesMap

PhreeqPython integration that replaces first-order bulk decay with full PHREEQC geochemistry.

> Requires: `pip install phreeqpython`

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
| `apply_geochemistry(store, dt, pipe_diam=None, pipe_vel=None)` | Replaces `bulk_first_order_multi()`; runs PHREEQC reactions per segment in batches |
| `apply_mixing(node_C, node_flow, dt)` | Chemical equilibrium after node mixing (pH correction) |
| `make_injection_solution(C_vector)` | Convert a concentration vector to a PHREEQC solution dict |
| `langelier_index(C_vec)` | Langelier Saturation Index for CaCO₃ (> 0 = scaling, < 0 = corrosive) |
| `saturation_indices(C_vec)` | Saturation indices for Calcite, Dolomite, Goethite, and others |
| `reset()` | Reset PHREEQC instance on simulation restart |

#### Pre-configured recipes

```python
from inzingaflow.geochemistry import chlorine_decay_geochem, full_water_chemistry

# Chlorine + pH (n_species=2)
geo = chlorine_decay_geochem(k_bulk_per_day=0.5)

# Chlorine, pH, alkalinity, calcium, iron, manganese (n_species=6)
geo = full_water_chemistry()
```

---

### Stability Functions

```python
from inzingaflow import recommended_dt, check_dt
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
solver = InzingaFlowSolver("network.inp", n_species=1, track_mass=True)
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
from inzingaflow import merge_segments

n_removed = merge_segments(
    store,               # SegmentStore
    tol=1e-6,            # concentration tolerance [mg/L]
    min_volume=1e-12,    # segments smaller than this are always removed [m³]
)
```

Algorithm: iterative parallel reduction (O(n log n)); fully vectorized. Called automatically by the solver every `merge_interval` steps.

---

## Advanced Usage

### Time-varying injection with a callback

```python
def injection(t: float, solver: InzingaFlowSolver):
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
solver = InzingaFlowSolver("network.inp", n_species=1)
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
solver = InzingaFlowSolver("network.inp", n_species=1, track_mass=True)
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

---

## Wall Reactions

InzingaFlow implements the two-film model from Rossman (1994):

```
Re    = v · D / ν
Sh    = 0.023 · Re^0.83 · Sc^(1/3)    (turbulent, Re > 4000)
Sh    = 3.65                            (laminar, Re ≤ 4000)
k_f   = Sh · D_mol / D    [m/s]        film transport coefficient
k_eff = k_f · k_w / (k_f + k_w)       series resistance
k_vol = k_eff · 4 / D     [1/s]        volumetric rate (cylinder A/V = 4/D)
```

### Specifying k_wall per pipe

```python
n_pipes = len(solver.pipe_ids)

# Uniform
k_wall = np.full((n_pipes, 1), 5e-6)   # [m/s]

# Variable per pipe (e.g. based on pipe material)
k_wall = np.zeros((n_pipes, 1))
k_wall[0:20] = 1e-5   # cast iron
k_wall[20:]  = 2e-6   # PVC

solver = InzingaFlowSolver("network.inp", n_species=1, k_wall=k_wall)
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

**Q: How do I add a new PHREEQC element?**

```python
smap = SpeciesMap(
    species_names=["Cl2","pH","Alk","Ca","Fe","Mn","SO4"],
    phreeqc_names=["Cl", None,"Alk","Ca","Fe","Mn","S(6)"],
    units        =["mg/L","","meq/L","mg/L","mg/L","mg/L","mg/L"],
    is_pH        =[False,True,False,False,False,False,False],
)
```

**Q: Are pumps simulated?**

Pumps are excluded by default (`include_pumps=False`). You can include them via `HydraulicModel("net.inp", include_pumps=True)`, though this is only meaningful when residence time inside the pump is relevant.

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
| **EPS** | Extended Period Simulation: time-varying simulation with periodically updated hydraulics |
| **LTA** | Lagrangian Transport Approach: transport described in the reference frame of the fluid |
| **LSI** | Langelier Saturation Index: measure of CaCO₃ saturation in water |
| **PHREEQC** | Geochemical modelling software by the USGS |
| **SoA** | Structure of Arrays: memory layout where each property occupies a separate array |
| **`k_bulk`** | Bulk decay constant [1/s]: first-order decay in the water column |
| **`k_wall`** | Wall reaction rate [m/s]: chemical reaction at the inner pipe surface |
| **`decay_pipe`** | Combined decay factor table `(n_pipes × n_species)` = `exp(-(k_bulk + k_wall_vol) × dt)` |
| **segment** | Lagrangian fluid parcel with fixed volume, position, and concentration vector |

---

## Changelog

### 1.0.0

- Initial release
- Vectorized LTA, multi-species, wall reactions, tanks, EPS
- GeochemSolver with PhreeqPython integration
- MassBalanceTracker, CFL stability check
- Parallel-reduction segment merging (O(n log n))
- Flow reversal detection and segment mirroring
