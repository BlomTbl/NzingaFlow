# InzingaFlow

Lagrangian Transport Quality Simulator for EPANET Drinking Water Networks

InzingaFlow calculates water quality through pipe networks using the Lagrangian Transport Algorithm (LTA). Hydraulics (flow rates, velocities) are provided by EPANET via epynet; the quality calculation is performed entirely within InzingaFlow—not within EPANET. Optionally, full geochemistry is supported via PhreeqPython/PHREEQC.

---

## Installation

### Basic installation (conservative transport, first-order decay)

```bash
pip install inzingaflow
```

### With geochemistry (PhreeqPython/PHREEQC)

```bash
pip install "inzingaflow[geochem]"
```

### From source code

```bash
git clone https://github.com/inzingaflow/inzingaflow
cd inzingaflow
pip install -e ".[geochem,dev]"
```

---

## Quickstart

### Conservative transport (tracer)

```python
import numpy as np
from inzingaflow import InzingaFlowSolver, EPSRunner

# Load network and create solver
solver = InzingaFlowSolver("network.inp", n_species=1)

# Automatically determine the stable time step
qual_dt = solver.recommended_dt(decay_k=np.array([0.0]))
print(f"Recommended time step: {qual_dt:.1f} s")

# Set EPS runner
runner = EPSRunner(
solver,
qual_dt = qual_dt,
hyd_dt = 300.0, # hydraulic time step [s]
duration = 86400.0, # simulation duration [s]
)

# Run simulation
results = runner.run(
decay_k = np.array([0.0]),
inject_schedule = {"R1": [(0, 86400, np.array([1.0]))]},
verbose = True,
)
# results.shape == (n_steps, node_count, 1)
```

### Chlorine with first-order decay

```python
solver = InzingaFlowSolver("network.inp", n_species=1)
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)

results = runner.run(
decay_k = np.array([2.0 / 86400]), # k = 2.0 / day
inject_schedule = {"R1": [(0, 86400, np.array([0.8]))]}, # 0.8 mg/L Cl₂
)
```

### Complete geochemistry via PhreeqPython

```python
import numpy as np
from inzingaflow import InzingaFlowSolver, EPSRunner
from inzingaflow.geochemistry import full_water_chemistry

# Ready-to-use GeochemSolver: Cl₂ + pH + Alk + Ca + Fe + Mn
geo = full_water_chemistry( 
background = {"temp": 12, "pH": 7.8, "Alkalinity": 3e-3, "Ca": 1.5e-3}, 
equilibrium_minerals = ["Calcite"],
)

solver = InzingaFlowSolver("network.inp", n_species=6, geochem=geo)
runner = EPSRunner(solver, qual_dt=10.0, hyd_dt=300.0, duration=86400.0)

results = runner.run(
decay_k = np.zeros(6),
inject_schedule = {
"R1": [(0, 86400, np.array([0.8, 7.8, 3.0, 60.0, 0.05, 0.02]))]
# Cl₂ pH Alk[meq/L] Ca Fe Mn [mg/L]
},
)
```

### Booster injection (e.g., chlorine booster)

```python
runner.run(
decay_k = np.array([2.0 / 86400]),
booster_schedule = {
"B1": [(3600, 86400, np.array([0.4]))] # Set C to 0.4 mg/L
},
)
```

---

## Architecture

```
InzingaFlowSolver ← main API: manages segments, hydraulics, tanks
│
├── HydraulicModel ← EPANET ↔ epynet connection; provides numpy arrays
├── SegmentStore ← SoA storage; O(1) add/remove; Auto-resize
│
├── lta.py ← vectorized kernel:
│ ├── bulk_first_order_multi() first-order bulk decay
│ ├── wall_first_order_multi() wall decay (two-film model)
│ ├── advect() segment advection
│ ├── node_mixing_multi() flow-weighted node mixing
│ └── tank_step_implicit() CSTR tank (implicit Euler)
│
├── merging.py ← segment merging (parallel reduction, O(n log n))
├── stability.py ← CFL check, recommended dt, mass balance
│
└── geochemistry.py ← PhreeqPython integration (optional)
├── GeochemSolver PHREEQC per segment (kinetics + equilibrium)
├── SpeciesMap connection InzingaFlow substances ↔ PHREEQC elements
├── chlorine_decay_geochem() ready-made recipe: Cl₂ + pH
└── full_water_chemistry() ready-made recipe: Cl₂+pH+Alk+Ca+Fe+Mn

EPSRunner ← EPS loop: hyd updates, injection, progress
```

---

## Difference with Victoria

| Aspect | InzingaFlow | Victoria |
|---|---|---|
| **Algorithm** | LTA (vectorized NumPy) | LTA (BFS, Python objects) |
| **Geochemistry moment** | Eager — each time step per segment | Lazy — when queried via fraction mixing |
| **Node mixing** | Linear + PHREEQC correction (step 5b) | PHREEQC `mix_solutions()` always |
| **Multi-species** | Yes, NumPy matrix `C[n, n_species]` | Via PHREEQC solution references |
| **Performance** | High (vectorized) | Lower (Python loop per parcel) |
| **Geochemistry** | Optional via `GeochemSolver` | Always via `PhreeqPython` |

---

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- epynet ≥ 0.6
- phreeqpython ≥ 1.5 *(for geochemistry only)*

---

## License

MIT