🇳🇱 [Nederlands](README_NL.md) &nbsp;|&nbsp; 🇬🇧 [English](README.md)
# InzingaFlow

**Lagrangian Transport Kwaliteitssimulator voor EPANET-netwerken**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.21-orange)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.0.0-informational)](CHANGELOG.md)

InzingaFlow simuleert waterchemische kwaliteit in drinkwaterdistributienetwerken via de **Lagrangian Transport Approach (LTA)**. Stoffen reizen als discrete segmenten mee met de waterstroming — zonder de numerieke diffusie van Euleriaanse methoden. Alle kernberekeningen zijn volledig gevectoriseerd met NumPy.

---

## Inhoudsopgave

- [Kenmerken](#kenmerken)
- [Installatie](#installatie)
- [Snelstart](#snelstart)
- [Architectuur](#architectuur)
- [API-referentie](#api-referentie)
  - [InzingaFlowSolver](#inzingaflowsolver)
  - [EPSRunner](#epsrunner)
  - [HydraulicModel](#hydraulicmodel)
  - [SegmentStore](#segmentstore)
  - [LTA-kernfuncties](#lta-kernfuncties)
  - [GeochemSolver & SpeciesMap](#geochemsolver--speciesmap)
  - [Stabiliteitsfuncties](#stabiliteitsfuncties)
  - [merge\_segments](#merge_segments)
- [Geavanceerd gebruik](#geavanceerd-gebruik)
- [Wandreacties](#wandreacties)
- [Veelgestelde vragen](#veelgestelde-vragen)
- [Woordenlijst](#woordenlijst)

---

## Kenmerken

- **Vectorized LTA-solver** — geen Python-lussen over segmenten; alle berekeningen via NumPy
- **Multi-species** — meerdere stoffen simultaan in één `C`-matrix `(n_segmenten × n_stoffen)`
- **Wandreacties** — twee-film-model per leiding (Dittus-Boelter / Rossman 1994)
- **Tanks** — CSTR-model met impliciet Euler (onvoorwaardelijk stabiel)
- **Extended Period Simulation (EPS)** — automatische hydraulica-updates elke `hyd_dt` seconden
- **Volledige geochemie** — optionele PhreeqPython/PHREEQC-integratie
- **CFL-stabiliteitscontrole** — automatische waarschuwing en `recommended_dt()`
- **Massabalansregistratie** — via `MassBalanceTracker` (opt-in)
- **Flow reversal** — detectie en correcte segmentspiegeling

### Prestatiecijfers

| Operatie | Grootte | Tijd |
|---|---|---|
| `apply_combined_decay()` | 50 000 segs, 6000 leidingen, 3 stoffen | ~0.7 ms |
| `advect()` | 50 000 segs | ~0.2 ms |
| `node_mixing_multi()` | 50 000 segs, 2500 exits, 5000 knopen | ~0.2 ms |
| `merge_segments()` | 1000 segs, 3 stoffen | ~260 µs |

---

## Installatie

```bash
pip install numpy epynet

# Optioneel: geochemie via PhreeqPython
pip install phreeqpython
```

> **Vereisten:** Python ≥ 3.9, NumPy ≥ 1.21, epynet ≥ 2.0 (2025-versie)

---

## Snelstart

### Eenvoudige simulatie (één stof)

```python
import numpy as np
from inzingaflow import InzingaFlowSolver, EPSRunner

solver = InzingaFlowSolver("netwerk.inp", n_species=1)

runner = EPSRunner(
    solver,
    qual_dt=5.0,       # kwaliteitstijdstap [s]
    hyd_dt=300.0,      # hydraulica-update interval [s]
    duration=86400.0,  # simulatieduur [s]
)

results = runner.run(
    decay_k=np.array([0.0003]),           # bulkvervalconstante [1/s]
    inject_schedule={
        "R1": [(0, 86400, np.array([1.0]))]  # 1 mg/L gedurende 24 uur
    },
    verbose=True,
)
# results.shape == (n_stappen, node_count, 1)
```

### Multi-species

```python
solver = InzingaFlowSolver("netwerk.inp", n_species=2)
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)

results = runner.run(
    decay_k=np.array([0.0003, 0.0]),     # chloor vervalt; tracer conservatief
    inject_schedule={
        "R1": [(0, 86400, np.array([1.0, 100.0]))]
    },
)
# results[:, :, 0]  →  chloor per knoop per tijdstap
# results[:, :, 1]  →  tracer per knoop per tijdstap
```

### Met wandreacties

```python
n_pipes = len(solver.pipe_ids)
k_wall = np.full((n_pipes, 1), 1e-5)    # [m/s] — uniform over alle leidingen

solver = InzingaFlowSolver("netwerk.inp", n_species=1, k_wall=k_wall)
```

### Met geochemie (PhreeqPython)

```python
from inzingaflow import InzingaFlowSolver, EPSRunner
from inzingaflow.geochemistry import full_water_chemistry

geo = full_water_chemistry()
solver = InzingaFlowSolver("netwerk.inp", n_species=6, geochem=geo)
runner = EPSRunner(solver, qual_dt=10.0, hyd_dt=300.0, duration=86400.0)

# Stoffen: [Cl2, pH, Alk, Ca, Fe, Mn]
results = runner.run(
    decay_k=np.zeros(6),
    inject_schedule={
        "R1": [(0, 86400, np.array([1.0, 7.5, 2.5, 60.0, 0.1, 0.05]))]
    },
)
```

---

## Architectuur

### LTA-principe

Elke tijdstap `dt` doorloopt de solver de volgende fasen:

```
1. Verval         C *= exp(-(k_bulk + k_wall_vol) * dt)   [gecombineerde decay_pipe-tabel]
2. Advectie       x += v[pipe] * dt
3. Exit-detectie  x >= L[pipe]  →  segment verlaat leiding
4. Knoopmenging   debietgewogen  →  node_C (n_nodes × n_species)
5. Tankmodel      CSTR impliciet Euler  →  node_C bijgewerkt
6. Splitsing      segment naar splitsingknoop  →  proportioneel gesplitst
7. Merging        aangrenzende segs met |ΔC| < tol  →  samengevoegd
```

### Modulaire opbouw

| Module | Klasse / Functie | Verantwoordelijkheid |
|---|---|---|
| `solver.py` | `InzingaFlowSolver` | Volledige LTA-solver; koppeling van alle onderdelen |
| `eps.py` | `EPSRunner` | EPS-tijdlus, injectie, voortgang |
| `hydraulics.py` | `HydraulicModel` | EPANET-koppeling via epynet |
| `segments.py` | `SegmentStore` | Structure-of-Arrays opslag |
| `lta.py` | `advect`, `node_mixing_multi`, … | Vectorized kernberekeningen |
| `merging.py` | `merge_segments` | Parallel-reductie segmentmerging |
| `stability.py` | `recommended_dt`, `MassBalanceTracker` | CFL-controle en massabalans |
| `geochemistry.py` | `GeochemSolver`, `SpeciesMap` | PhreeqPython-integratie |

### SegmentStore (Structure-of-Arrays)

```
pipe   [int32,   capacity]          pipe-index per segment
x      [float64, capacity]          positie langs leiding [m]
C      [float64, capacity × n_sp]   concentraties per stof
volume [float64, capacity]          segmentvolume [m³]
n                                   aantal actieve segmenten
```

Pre-allocatie vermijdt heap-allocaties tijdens de simulatie. Capaciteitsoverschrijding triggert automatische verdubbeling.

---

## API-referentie

### InzingaFlowSolver

```python
InzingaFlowSolver(
    inp_path:   str,
    n_species:  int   = 1,
    capacity:   int   = 200_000,
    k_wall:     ndarray | None = None,   # (n_pipes, n_species) of (n_species,) [m/s]
    D_mol:      float = 1.3e-9,          # moleculaire diffusiviteit [m²/s]
    nu:         float = 1e-6,            # kinematische viscositeit [m²/s]
    track_mass: bool  = False,
    geochem           = None,            # GeochemSolver instantie
)
```

**Parameters**

| Parameter | Type | Omschrijving |
|---|---|---|
| `inp_path` | `str` | Pad naar EPANET `.inp` bestand |
| `n_species` | `int` | Aantal te simuleren chemische stoffen |
| `capacity` | `int` | Initiële SegmentStore-capaciteit (auto-resize) |
| `k_wall` | `ndarray \| None` | Wandreactiesnelheid [m/s]; `None` = geen wandreacties |
| `D_mol` | `float` | Moleculaire diffusiviteit [m²/s] |
| `nu` | `float` | Kinematische viscositeit [m²/s] |
| `track_mass` | `bool` | Activeer massabalansregistratie |
| `geochem` | `GeochemSolver \| None` | Vervangt eerste-orde bulkverval door PHREEQC-geochemie |

**Methoden**

| Methode | Retourtype | Omschrijving |
|---|---|---|
| `step(dt, decay_k, merge_interval=10, merge_tol=1e-6, check_cfl=False)` | `ndarray (node_count, n_species)` | Voer één kwaliteitstijdstap uit |
| `update_hydraulics(simtime=0)` | `None` | Herbereken hydraulica; detecteert flow reversals |
| `inject(node_uid, C_vector, volume)` | `None` | Injecteer vanuit knoop, proportioneel over uitgaande leidingen |
| `inject_pipe(pipe_uid, C_vector, volume, x=0.0)` | `None` | Injecteer direct in leiding op positie `x` [m] |
| `booster_inject(node_uid, C_set, flow_frac=1.0)` | `None` | Stel concentratie vast op uitgaande leidingen |
| `check_stability(dt, decay_k, warn=True)` | `dict` | CFL + reactie-stabiliteitscheck |
| `recommended_dt(decay_k)` | `float` | Maximale stabiele tijdstap [s] |
| `mass_balance()` | `dict \| None` | Massabalansrapport (vereist `track_mass=True`) |
| `geochem_report(C_vec)` | `dict` | LSI, verzadigingsindices, pH (vereist `geochem`) |

**Attributen**

| Attribuut | Type | Omschrijving |
|---|---|---|
| `node_count` | `int` | Aantal knooppunten |
| `n_species` | `int` | Aantal stoffen |
| `node_index` | `dict[str, int]` | Naam → 0-based index |
| `pipe_index` | `dict[str, int]` | EPANET-naam → pipe-index |
| `node_names` | `list[str]` | Knoopnamen in index-volgorde |
| `pipe_ids` | `list[str]` | Leidingnamen in index-volgorde |
| `pipe_length` | `ndarray (n_pipes,)` | Leidinglengtes [m] |
| `pipe_area` | `ndarray (n_pipes,)` | Dwarsdoorsneden [m²] |
| `segments` | `SegmentStore` | Actieve segmenten |

---

### EPSRunner

```python
EPSRunner(
    solver:   InzingaFlowSolver,
    qual_dt:  float,   # kwaliteitstijdstap [s]
    hyd_dt:   float,   # hydraulica-update interval [s]
    duration: float,   # totale simulatieduur [s]
)
```

#### `EPSRunner.run()`

```python
run(
    decay_k,                            # (n_species,) bulk [1/s]
    inject_schedule:  dict  = None,     # {knoop: [(t_start, t_end, C_vec), ...]}
    inject_fn:        callable = None,  # dynamische injectie-callback
    booster_schedule: dict  = None,     # {knoop: [(t_start, t_end, C_set), ...]}
    merge_interval:   int   = 10,
    merge_tol:        float = 1e-6,
    check_cfl:        bool  = True,
    verbose:          bool  = False,
) -> ndarray  # shape: (n_stappen, node_count, n_species)
```

**`inject_schedule` formaat**

```python
inject_schedule = {
    "R1": [
        (0,    3600,  np.array([1.0])),   # 1 mg/L in eerste uur
        (3600, 86400, np.array([0.5])),   # 0.5 mg/L daarna
    ]
}
```

**`inject_fn` callback**

```python
def mijn_injectie(t: float, solver: InzingaFlowSolver):
    if t < 1800:
        solver.inject("R1", [1.0], volume=0.05)

results = runner.run(decay_k=..., inject_fn=mijn_injectie)
```

> `inject_fn` heeft voorrang op `inject_schedule` als beide zijn opgegeven.

#### `EPSRunner.time_axis()`

```python
t_uur = runner.time_axis(unit="h")   # "s", "min" of "h"
```

---

### HydraulicModel

Lage-niveau EPANET-koppeling via epynet. Normaliter intern aangemaakt door `InzingaFlowSolver`.

```python
HydraulicModel(inp_path: str, include_pumps: bool = False)
```

| Methode | Omschrijving |
|---|---|
| `solve(simtime=0)` | Los hydraulica op voor één EPS-tijdstip [s] |
| `get_topology()` | Retourneert `(pipe_start, pipe_end, pipe_length, pipe_area, node_count, pipe_ids, node_names)` — gecached |
| `get_topology_with_reversal()` | Topologie met pipe_start/end gecorrigeerd voor stroomrichting |
| `get_hydraulic_state()` | Retourneert `(flow [m³/s], velocity [m/s], reversed_mask)` — altijd SI |
| `summary()` | Diagnostisch overzicht als string |

---

### SegmentStore

Structure-of-Arrays opslag met vaste pre-allocatie.

```python
SegmentStore(capacity: int = 100_000, n_species: int = 1)
```

| Attribuut / Methode | Omschrijving |
|---|---|
| `pipe[:n]` | `ndarray (capacity,) int32` — pipe-index per segment |
| `x[:n]` | `ndarray (capacity,) float64` — positie langs leiding [m] |
| `C[:n]` | `ndarray (capacity, n_species) float64` — concentraties |
| `volume[:n]` | `ndarray (capacity,) float64` — segmentvolume [m³] |
| `n` | `int` — aantal actieve segmenten |
| `add(pipe, x, volume, C_vector)` | Voeg segment toe; auto-resize bij capaciteitsoverschrijding |
| `remove(indices)` | Verwijder via swap-with-last in O(1) |
| `pipe_a`, `x_a`, `C_a`, `volume_a` | Properties die views `[:n]` retourneren |

---

### LTA-kernfuncties

Alle functies in `lta.py` zijn volledig gevectoriseerd met NumPy.

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

| Functie | Handtekening | Omschrijving |
|---|---|---|
| `bulk_first_order_multi` | `(C, k_vec, dt) → None` | `C *= exp(-k_vec * dt)` in-place; broadcasting over segmenten |
| `wall_first_order_multi` | `(C, pipe, k_wall, dt) → None` | Wandverval per segment in-place |
| `combined_decay_factors` | `(k_bulk, k_wall_vol, dt, n_pipes) → (n_pipes, n_species)` | Gecombineerde vervalfactoren — eenmalig per hydraulica-update |
| `apply_combined_decay` | `(C, pipe, decay_pipe) → None` | `C *= decay_pipe[pipe]` in-place |
| `compute_wall_k` | `(diam, velocity, k_w, D_mol, nu) → (n_pipes, n_species) [1/s]` | Volumetrische wandreactiesnelheid via twee-film-model |
| `advect` | `(x, pipe, velocity, dt) → None` | `x += velocity[pipe] * dt` in-place |
| `node_mixing_multi` | `(exit_mask, pipe, C, flow, pipe_end, node_count) → (node_count, n_species)` | Debietgewogen menging via `bincount` |
| `tank_step_implicit` | `(C_tank, Q_in, C_in, Q_out, V_tank, k_b, dt) → None` | CSTR impliciet Euler in-place |

---

### GeochemSolver & SpeciesMap

PhreeqPython-integratie die eerste-orde bulkverval vervangt door volledige PHREEQC-geochemie.

> Vereist: `pip install phreeqpython`

#### SpeciesMap

```python
SpeciesMap(
    species_names: list[str],         # leesbare namen
    phreeqc_names: list[str | None],  # PHREEQC-elementnaam; None = geen koppeling
    units:         list[str],         # "mg/L", "mol/L", "meq/L", "mmol/L"
    is_pH:         list[bool] = [],   # True als stof de pH-waarde is
)
```

Voorbeeld:

```python
smap = SpeciesMap(
    species_names=["Cl2",  "pH",  "Alk"],
    phreeqc_names=["Cl",   None,  "Alk"],
    units        =["mg/L", "",    "meq/L"],
    is_pH        =[False,  True,  False],
)
```

Ondersteunde PHREEQC-elementen: `Cl`, `Ca`, `Mg`, `Na`, `K`, `Fe`, `Mn`, `N(5)` (NO₃), `N(3)` (NO₂), `N(-3)` (NH₄), `Alk`, `C(4)` (TIC/CO₂), `S(6)` (SO₄), `P` (fosfaat).

#### GeochemSolver

```python
GeochemSolver(
    species_map:         SpeciesMap,
    background_solution: dict | None = None,
    kinetics_script:     str  | None = None,   # PHREEQC KINETICS-blok
    equilibrium_phases:  dict | None = None,   # {fase: (SI_doel, hoeveelheid)}
    reaction_script:     str  | None = None,   # vrij PHREEQC-script (geavanceerd)
    db_path:             str  | None = None,   # pad naar .dat database
    batch_size:          int  = 500,
    min_C_threshold:     float = 1e-9,
)
```

| Methode | Omschrijving |
|---|---|
| `apply_geochemistry(store, dt, pipe_diam=None, pipe_vel=None)` | Vervangt `bulk_first_order_multi()`; PHREEQC-reacties per segment in batches |
| `apply_mixing(node_C, node_flow, dt)` | Geochemisch evenwicht na knoopmenging (pH-correctie) |
| `make_injection_solution(C_vector)` | Zet concentratieprofiel om naar PHREEQC-oplossingsdict |
| `langelier_index(C_vec)` | LSI voor CaCO₃ (> 0 = aankorsting, < 0 = corrosief) |
| `saturation_indices(C_vec)` | Verzadigingsindices voor Calcite, Dolomite, Goethite, e.a. |
| `reset()` | Reset PHREEQC-instantie bij herstart simulatie |

#### Voorgeconfigureerde recepten

```python
from inzingaflow.geochemistry import chlorine_decay_geochem, full_water_chemistry

# Chloor + pH (n_species=2)
geo = chlorine_decay_geochem(k_bulk_per_day=0.5)

# Chloor, pH, alkaliniteit, calcium, ijzer, mangaan (n_species=6)
geo = full_water_chemistry()
```

---

### Stabiliteitsfuncties

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

Berekent `min(dt_CFL, dt_rxn) × safety` waarbij:
- **CFL:** `dt_CFL = min(L / v)` over alle leidingen
- **Reactie:** `dt_rxn = 0.1 / k_max` (10× veiligheidsmarge)

#### `check_dt()` — retourneert rapport-dict

```python
report = solver.check_stability(dt=5.0, decay_k=np.array([0.001]))
# {
#   "cfl_ok":      bool,
#   "reaction_ok": bool,
#   "stable":      bool,
#   "dt_rec":      float,
#   "cfl_ratio":   float,   # dt / dt_CFL  (< 1 is veilig)
#   "violations":  list[str],
# }
```

#### `MassBalanceTracker`

Activeren met `track_mass=True`:

```python
solver = InzingaFlowSolver("netwerk.inp", n_species=1, track_mass=True)
# ... simulatie ...
mb = solver.mass_balance()
# {
#   "injected":      ndarray,   # massa ingespoten per stof
#   "in_system":     ndarray,   # massa nog in segmenten
#   "bulk_decay":    ndarray,   # gecumuleerd bulkverlies
#   "wall_decay":    ndarray,   # gecumuleerd wandverlies
#   "outflow":       ndarray,   # massa via eindknopen verlaten
#   "balance_error": ndarray,   # relatieve fout per stof
#   "ok":            bool,      # True als fout < 1%
# }
```

---

### `merge_segments()`

```python
from inzingaflow import merge_segments

n_verwijderd = merge_segments(
    store,               # SegmentStore
    tol=1e-6,            # concentratietolerantie [mg/L]
    min_volume=1e-12,    # segmenten kleiner dan dit worden altijd verwijderd [m³]
)
```

Algoritme: iteratieve parallel-reductie (O(n log n)); volledig vectorized. Wordt automatisch uitgevoerd door de solver elke `merge_interval` stappen.

---

## Geavanceerd gebruik

### Tijdvariabele injectie met callback

```python
def injectie(t: float, solver: InzingaFlowSolver):
    """Injecteer alleen als het debiet bij R1 boven 10 L/s ligt."""
    flow, _ = solver._get_hydraulics()
    out_pipes = solver._node_outpipes[solver.node_index["R1"]]
    q_tot = sum(flow[p] for p in out_pipes)
    if q_tot > 0.01:
        solver.inject("R1", C_vector=[1.0], volume=q_tot * 5.0)

results = runner.run(decay_k=np.array([0.0003]), inject_fn=injectie)
```

### Booster-injectiestations

```python
booster_schedule = {
    "B1": [(6*3600, 18*3600, np.array([0.8]))]  # 0.8 mg/L van uur 6–18
}

results = runner.run(
    decay_k=np.array([0.0003]),
    inject_schedule={"R1": [(0, 86400, np.array([1.0]))]},
    booster_schedule=booster_schedule,
)
```

### Aanbevolen tijdstap bepalen

```python
solver = InzingaFlowSolver("netwerk.inp", n_species=1)
dt_opt = solver.recommended_dt(decay_k=np.array([0.001]))
print(f"Aanbevolen tijdstap: {dt_opt:.1f} s")

runner = EPSRunner(solver, qual_dt=dt_opt, hyd_dt=300.0, duration=86400.0)
```

### Geochemisch rapport

```python
C_monster = np.array([0.5, 7.8, 3.0, 75.0, 0.05, 0.02])
rapport = solver.geochem_report(C_monster)

print(f"pH:  {rapport['pH']:.2f}")
print(f"LSI: {rapport['lsi']:.3f}  (>0 = aankorsting, <0 = corrosief)")
for mineraal, si in rapport['saturation_indices'].items():
    print(f"  {mineraal}: SI = {si:.3f}")
```

### Massabalanscontrole

```python
solver = InzingaFlowSolver("netwerk.inp", n_species=1, track_mass=True)
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=3600.0)
results = runner.run(
    decay_k=np.array([0.001]),
    inject_schedule={"R1": [(0, 3600, np.array([1.0]))]},
)

mb = solver.mass_balance()
print(f"Ingespoten: {mb['injected'][0]:.4f} kg")
print(f"In systeem: {mb['in_system'][0]:.4f} kg")
print(f"Bulkverval: {mb['bulk_decay'][0]:.4f} kg")
print(f"Uitstroom:  {mb['outflow'][0]:.4f} kg")
print(f"Fout:       {mb['balance_error'][0]*100:.3f}%  OK={mb['ok']}")
```

### Directe injectie in een leiding

```python
solver.inject_pipe(
    pipe_uid="P100",
    C_vector=np.array([2.0]),
    volume=0.01,
    x=solver.pipe_length[solver.pipe_index["P100"]] / 2,  # midden van leiding
)
```

---

## Wandreacties

InzingaFlow implementeert het twee-film-model van Rossman (1994):

```
Re    = v · D / ν
Sh    = 0.023 · Re^0.83 · Sc^(1/3)    (turbulent, Re > 4000)
Sh    = 3.65                            (laminair, Re ≤ 4000)
k_f   = Sh · D_mol / D    [m/s]        filmtransport
k_eff = k_f · k_w / (k_f + k_w)       serieschakeling
k_vol = k_eff · 4 / D     [1/s]        volumetrisch (cilinder A/V = 4/D)
```

### k_wall per leiding opgeven

```python
n_pipes = len(solver.pipe_ids)

# Uniform
k_wall = np.full((n_pipes, 1), 5e-6)   # [m/s]

# Per leiding variabel (bijv. op basis van materiaal)
k_wall = np.zeros((n_pipes, 1))
k_wall[0:20] = 1e-5   # gietijzer
k_wall[20:]  = 2e-6   # PVC

solver = InzingaFlowSolver("netwerk.inp", n_species=1, k_wall=k_wall)
```

> `k_wall = 0` geeft geen wandreactie, ongeacht de stroomsnelheid.

---

## Veelgestelde vragen

**Q: De simulatie geeft een CFL-waarschuwing. Wat moet ik doen?**

Gebruik `solver.recommended_dt()` om de optimale tijdstap te berekenen en geef die door als `qual_dt` aan `EPSRunner`.

**Q: De massafout is groter dan 1%. Wat zijn mogelijke oorzaken?**

- De tijdstap is te groot (CFL-schending)
- Wandreacties zijn actief — de massabalansschatting is een benadering voor wandverval
- Geochemie is actief: niet-lineaire reacties worden niet bijgehouden in de lineaire massabalans

**Q: Hoe voeg ik een nieuw PHREEQC-element toe?**

```python
smap = SpeciesMap(
    species_names=["Cl2","pH","Alk","Ca","Fe","Mn","SO4"],
    phreeqc_names=["Cl", None,"Alk","Ca","Fe","Mn","S(6)"],
    units        =["mg/L","","meq/L","mg/L","mg/L","mg/L","mg/L"],
    is_pH        =[False,True,False,False,False,False,False],
)
```

**Q: Worden pompen gesimuleerd?**

Standaard worden pompen overgeslagen (`include_pumps=False`). U kunt pompen includeren via `HydraulicModel("net.inp", include_pumps=True)`, maar dit heeft doorgaans alleen zin als de verblijftijd in de pomp relevant is.

**Q: Hoe interpreteer ik de Langelier Saturation Index (LSI)?**

| LSI | Betekenis | Risico |
|---|---|---|
| > +0.5 | Sterke CaCO₃-precipitatie | Aankorsting, verstopping |
| 0 tot +0.5 | Lichte verzadiging | Beschermende kalklaag |
| −0.5 tot 0 | Lichte onderverzadiging | Gering corrosierisico |
| < −0.5 | Sterke onderverzadiging | Corrosie van metaalleidingen |

---

## Woordenlijst

| Term | Omschrijving |
|---|---|
| **CFL** | Courant-Friedrichs-Lewy: stabiliteitseis `dt ≤ L/v` voor advectie |
| **CSTR** | Continuously Stirred Tank Reactor: perfect mengingstankmodel |
| **EPS** | Extended Period Simulation: tijdvariabele simulatie met periodiek bijgewerkte hydraulica |
| **LTA** | Lagrangian Transport Approach: transport vanuit het referentiekader van de vloeistof |
| **LSI** | Langelier Saturation Index: maat voor CaCO₃-verzadiging |
| **PHREEQC** | Geochemisch rekenprogramma van de USGS |
| **SoA** | Structure of Arrays: opslagopzet waarbij elke eigenschap een aparte array is |
| **`k_bulk`** | Bulkvervalconstante [1/s]: eerste-orde verval in de waterkolom |
| **`k_wall`** | Wandreactiesnelheid [m/s]: reactie aan de binnenzijde van de leiding |
| **`decay_pipe`** | Gecombineerde vervalfactor-tabel `(n_pipes × n_species)` = `exp(-(k_bulk + k_wall_vol) × dt)` |
| **segment** | Lagrangiaans vloeistofelement met vast volume, positie en concentratie |

---

## Versiehistorie

### 1.0.0

- Eerste release
- Vectorized LTA, multi-species, wandreacties, tanks, EPS
- GeochemSolver met PhreeqPython-integratie
- MassBalanceTracker, CFL-controle
- Parallel-reductie segmentmerging (O(n log n))
- Flow reversal detectie en segmentspiegeling
