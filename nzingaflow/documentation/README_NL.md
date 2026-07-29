🇳🇱 [Nederlands](README_NL.md) &nbsp;|&nbsp; 🇬🇧 [English](README.md)
# NzingaFlow

**Lagrangian Transport Kwaliteitssimulator voor EPANET-netwerken**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.24-orange)](https://numpy.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.2.1-informational)](CHANGELOG.md)

NzingaFlow simuleert waterchemische kwaliteit in drinkwaterdistributienetwerken via de **Lagrangian Transport Approach (LTA)**. Stoffen reizen als discrete segmenten mee met de waterstroming — zonder de numerieke diffusie van Euleriaanse methoden. Alle kernberekeningen zijn volledig gevectoriseerd met NumPy en optioneel versneld met Numba JIT-compilatie.

---

## Inhoudsopgave

- [Kenmerken](#kenmerken)
- [Installatie](#installatie)
- [Snelstart](#snelstart)
- [Architectuur](#architectuur)
- [API-referentie](#api-referentie)
  - [NzingaFlowSolver](#nzingaflowsolver)
  - [EPSRunner](#epsrunner)
  - [HydraulicModel](#hydraulicmodel)
  - [SegmentStore](#segmentstore)
  - [LTA-kernfuncties](#lta-kernfuncties)
  - [GeochemSolver & SpeciesMap](#geochemsolver--speciesmap)
  - [MsxReactionSystem](#msxreactionsystem)
  - [MsxSimulation & MsxNativeLib](#msxsimulation--msxnativelib)
  - [Stabiliteitsfuncties](#stabiliteitsfuncties)
  - [merge\_segments](#merge_segments)
- [Geavanceerd gebruik](#geavanceerd-gebruik)
- [Wandreacties](#wandreacties)
- [Veelgestelde vragen](#veelgestelde-vragen)
- [Woordenlijst](#woordenlijst)

---

## Kenmerken

- **Vectorized LTA-solver** — geen Python-lussen over segmenten; alle berekeningen via NumPy
- **Numba JIT-kernels** — optionele ~4–5× versnelling via `pip install numba`
- **Multi-species** — meerdere stoffen simultaan in één `C`-matrix `(n_segmenten × n_stoffen)`
- **Gecombineerd verval** — bulk- en wandverval samengevoegd in één geheugenpass (`combined_decay_multi`)
- **Wandreacties** — twee-film-model per leiding met drieregime Sherwood-correlatie (v1.1.0)
- **Temperatuurcorrectie** — Arrhenius/Hayduk-Laudie correctie op D_mol en k_wall (v1.1.0)
- **Lekkagemodellering** — proportioneel volumeverlies per segment zonder contaminantinstroom (v1.1.0)
- **MSX-reactiesysteem** — EPANET-MSX 2.0-compatibele multi-species reactielaag (v1.1.0)
- **MSX native library bridge** — directe ctypes-koppeling met `libepanetmsx` (EPANET-MSX 2.0); laadt `.msx`-bestanden ongewijzigd; native libs meegeleverd (v1.2.2)
- **Afsluiters in topologie** — PRV, PSV, TCV, FCV, GPV en PCV worden meegenomen in transport via `include_valves=True` (v1.2.1)
- **Tanks** — CSTR-model met impliciet Euler (onvoorwaardelijk stabiel)
- **Extended Period Simulation (EPS)** — automatische hydraulica-updates elke `hyd_dt` seconden
- **Volledige geochemie** — optionele PhreeqPython/PHREEQC-integratie
- **CFL-stabiliteitscontrole** — automatische waarschuwing en `recommended_dt()`
- **Massabalansregistratie** — via `MassBalanceTracker` (opt-in)
- **Flow reversal** — detectie en correcte segmentspiegeling
- **Pre-allocated buffers** — geen heap-allocaties in de hot-path
- **CSR pipe-index** — gecachede gesorteerde index in `SegmentStore`; lexsort overgeslagen indien geldig

### Prestatiecijfers

| Operatie | Grootte | Zonder Numba | Met Numba |
|---|---|---|---|
| `combined_decay_multi()` | 5.000 segs, 200 leidingen, 2 stoffen | ~63 µs | ~12 µs |
| `advect()` | 5.000 segs | ~15 µs | ~4 µs |
| `node_mixing_multi()` | 5.000 segs, 150 knopen | ~32 µs | ~8 µs |
| `merge_segments()` | 5.000 segs, 2 stoffen | ~2.200 µs | ~90 µs |
| **Volledige tijdstap** | 5.000 segs, 200 leidingen, 2 stoffen | **~280 µs** | **~60–90 µs** |

---

## Installatie

### Via PyPI

```bash
# Minimaal (alleen NumPy)
pip install nzingaflow

# Met Numba JIT (~4–5× sneller)
pip install "nzingaflow[numba]"

# Met volledige geochemie (PhreeqPython/PHREEQC)
pip install "nzingaflow[geochem]"

# Alles inclusief ontwikkeltools
pip install "nzingaflow[all]"
```

### Vanuit broncode

```bash
git clone https://github.com/nzingaflow/nzingaflow.git
cd nzingaflow
pip install -e ".[dev]"
```

> **Vereisten:** Python ≥ 3.10, NumPy ≥ 1.24, epynet ≥ 1.1

---

## Snelstart

### Eenvoudige simulatie (één stof)

```python
import numpy as np
from nzingaflow import NzingaFlowSolver, EPSRunner

solver = NzingaFlowSolver("netwerk.inp", n_species=1)

# Optioneel: Numba JIT-compilatie triggeren vóór de simulatie (~1 s eenmalig)
solver.warmup_numba()

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
solver = NzingaFlowSolver("netwerk.inp", n_species=2)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie
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

solver = NzingaFlowSolver("netwerk.inp", n_species=1, k_wall=k_wall)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie
```

### Met temperatuurcorrectie en lekkage (nieuw in v1.1.0)

```python
solver = NzingaFlowSolver(
    "netwerk.inp",
    n_species=1,
    k_wall=np.full((n_pipes, 1), 1e-5),
    temperature=12.0,          # [°C] — activeert Arrhenius/Hayduk-Laudie correctie
    leakage_fraction=0.12,     # 12% leidingverlies (typisch voor distributienetwerken)
)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie
```

### Met MSX multi-species reacties (nieuw in v1.1.0)

```python
from nzingaflow.msx import chloramine_decay_msx

rxn = chloramine_decay_msx(k_f=2.5e-4, k_ox=5e-5, solver='ros2')

solver = NzingaFlowSolver(
    "netwerk.inp",
    n_species=len(rxn.bulk_species),  # 3: HOCl, NH3, NH2Cl
    geochem=rxn,
)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(decay_k=np.zeros(3))
```

### Met de EPANET-MSX native bibliotheek (nieuw in v1.2.0)

Gebruik dit als u een bestaand `.msx`-bestand wilt uitvoeren via de officiële MSX C-solver,
zonder reactie-expressies in Python te herschrijven.

```python
from nzingaflow import MsxSimulation, run_msx

# Eenvoudigste gebruik: alles in één aanroep
result = run_msx("netwerk.inp", "netwerk.msx")
df = result.to_dataframe("CL2", element="node")   # DataFrame: tijd [h] × knoopnamen

# Meer controle via context-manager
with MsxSimulation("netwerk.inp", "netwerk.msx") as sim:
    state = sim.load()
    print([s.name for s in state.bulk_species()])

    sim.update_initial_quality(node_values={"R1": {"CL2": 1.0}})
    sim.configure_source("R1", "CL2", kind="CONCEN", level=1.0)

    result = sim.run()

print(result.time_hours())                   # tijdas [h]
print(result.node_concentrations("CL2"))     # array (T × N)
```

> Native bibliotheek (`libepanetmsx`/`epanet2`) wordt meegeleverd in `nzingaflow/lib/` — Linux x86-64 en
> Windows x86-64 bijgesloten. macOS is (nog) niet meegebouwd; bouw zelf via EPANETMSX's `CMakeLists.txt`.

### Met geochemie (PhreeqPython)

```python
from nzingaflow import NzingaFlowSolver, EPSRunner
from nzingaflow.geochemistry import full_water_chemistry

geo = full_water_chemistry()
solver = NzingaFlowSolver("netwerk.inp", n_species=6, geochem=geo)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie
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
1. Gecombineerd verval  C *= combined_exp[pipe]   waarbij combined_exp[p,s] = exp(-(k_bulk[s] + k_wall_vol[p,s]) * dt)
2. Advectie             x += v[pipe] * dt
3. Exit-detectie        x >= L[pipe]  →  segment verlaat leiding
4. Knoopmenging         debietgewogen  →  node_C (n_knopen × n_stoffen)
5. Tankmodel            CSTR impliciet Euler  →  node_C bijgewerkt
6. Pipe-routing         vectorized: doorgaande knopen in één pass; splitsingen in kleine lus
7. Merging              aangrenzende segs met |ΔC| < tol  →  samengevoegd (elke merge_interval stappen)
```

Stappen 1–4 worden uitgevoerd door Numba JIT-kernels als Numba is geïnstalleerd, zonder tussenliggende array-allocaties.

### Modulaire opbouw

| Module | Klasse / Functie | Verantwoordelijkheid |
|---|---|---|
| `solver.py` | `NzingaFlowSolver` | Volledige LTA-solver; koppeling van alle onderdelen |
| `eps.py` | `EPSRunner` | EPS-tijdlus, injectie, voortgang |
| `hydraulics.py` | `HydraulicModel` | EPANET-koppeling via epynet |
| `segments.py` | `SegmentStore` | Structure-of-Arrays opslag + CSR pipe-index |
| `lta.py` | `combined_decay_multi`, `advect`, … | Vectorized kernberekeningen met Numba JIT |
| `merging.py` | `merge_segments` | Parallel-reductie segmentmerging met Numba JIT |
| `stability.py` | `recommended_dt`, `MassBalanceTracker` | CFL-controle en massabalans |
| `geochemistry.py` | `GeochemSolver`, `SpeciesMap` | PhreeqPython-integratie |
| `msx.py` | `MsxReactionSystem` | Pure-Python MSX reactielaag: ODE-solvers + expressieparser (v1.1.0) |
| `msxlibrary.py` | `MsxSimulation`, `MsxNativeLib` | Directe ctypes-brug naar `libepanetmsx` (EPANET-MSX 2.0); laadt `.msx`-bestanden ongewijzigd; native libs meegeleverd (v1.2.2) |

### SegmentStore (Structure-of-Arrays)

```
pipe   [int32,   capacity]          pipe-index per segment
x      [float64, capacity]          positie langs leiding [m]
C      [float64, capacity × n_sp]   concentraties per stof
volume [float64, capacity]          segmentvolume [m³]
n                                   aantal actieve segmenten
```

Pre-allocatie vermijdt heap-allocaties tijdens de simulatie. Capaciteitsoverschrijding triggert automatische verdubbeling via een `on_resize`-callback die alle solver-buffers synchroon houdt. Een lazy CSR pipe-index (`build_csr()`) cachet de per-leiding gesorteerde volgorde voor `merge_segments`; de lexsort wordt overgeslagen zolang de index geldig is.

---

## API-referentie

### NzingaFlowSolver

```python
NzingaFlowSolver(
    inp_path:         str,
    n_species:        int   = 1,
    capacity:         int   = 200_000,
    k_wall:           ndarray | None = None,   # (n_pipes, n_species) of (n_species,) [m/s]
    D_mol:            float = 1.3e-9,          # moleculaire diffusiviteit [m²/s]
    nu:               float = 1e-6,            # kinematische viscositeit [m²/s]
    track_mass:       bool  = False,
    geochem                 = None,            # GeochemSolver of MsxReactionSystem instantie
    temperature:      float | None = None,     # watertemperatuur [°C] (v1.1.0)
    leakage_fraction: float = 0.0,             # fractioneel leidingverlies (v1.1.0)
    wall_mode:        str   = 'two_film',      # 'two_film' of 'direct' (v1.1.0)
    include_valves:   bool  = False,           # afsluiters in transporttopologie (v1.2.1)
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
| `geochem` | `GeochemSolver \| MsxReactionSystem \| None` | Vervangt eerste-orde bulkverval door volledige geochemie of MSX-reacties |
| `temperature` | `float \| None` | Watertemperatuur [°C]. Activeert Arrhenius-correctie op D_mol (`Ea≈17 kJ/mol`) en θ=1.047-correctie op k_wall (Rossman 2000). `None` = Rossman 1994-compatibel (geen correctie). *(v1.1.0)* |
| `leakage_fraction` | `float` | Fractie van leidingdebiet dat lekt (0.0–1.0). Elk segment verliest per tijdstap proportioneel volume; concentratie blijft constant. Typisch 0.05–0.20 voor distributienetwerken. *(v1.1.0)* |
| `wall_mode` | `str` | `'two_film'` (standaard): EPANET-compatibel serieschakeling `k_eff = k_f·k_w/(k_f+k_w)`. `'direct'`: k_wall is al k_eff — gebruik als k_wall uit directe kalibratie komt. *(v1.1.0)* |
| `include_valves` | `bool` | Als `True` worden afsluiters (PRV, PSV, PBV, FCV, TCV, GPV, PCV) opgenomen in de transporttopologie. EPANET lost de hydraulica voor afsluiters altijd correct op; deze optie zorgt dat NzingaFlow de verblijftijd en het stoftransport over afsluiters ook berekent. Standaard `False` voor achterwaartse compatibiliteit. *(v1.2.1)* |

**Methoden**

| Methode | Retourtype | Omschrijving |
|---|---|---|
| `step(dt, decay_k, merge_interval=10, merge_tol=1e-6, check_cfl=False)` | `ndarray (node_count, n_species)` | Voer één kwaliteitstijdstap uit |
| `update_hydraulics(simtime=0)` | `None` | Herbereken hydraulica; detecteert flow reversals en herbouwt routing-caches |
| `warmup_numba()` | `None` | Trigger Numba JIT-compilatie vóór de simulatie (eenmalig aanroepen na init) |
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
    solver:   NzingaFlowSolver,
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
def mijn_injectie(t: float, solver: NzingaFlowSolver):
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

Lage-niveau EPANET-koppeling via epynet. Normaliter intern aangemaakt door `NzingaFlowSolver`.

```python
HydraulicModel(inp_path: str, include_pumps: bool = False, include_valves: bool = False)
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
| `build_csr(n_pipes)` | Bouw of retourneer gecachede CSR pipe-index `(pipe_order, pipe_ptr)` |
| `pipe_a`, `x_a`, `C_a`, `volume_a` | Properties die views `[:n]` retourneren |
| `on_resize` | Callback `(new_capacity) → None`; aangeroepen bij capaciteitsverdubbeling |

---

### LTA-kernfuncties

Alle functies in `lta.py` zijn volledig gevectoriseerd met NumPy en JIT-gecompileerd met Numba indien beschikbaar.

```python
from nzingaflow.lta import (
    combined_decay_multi,   # bulk + wandverval gecombineerd in één pass  ← gebruik dit
    build_combined_exp,     # bereken gecombineerde vervalfactoren
    bulk_first_order_multi, # alleen bulkverval
    wall_first_order_multi, # alleen wandverval
    compute_wall_k,
    advect,
    exit_detect,
    node_mixing_multi,
    tank_step_implicit,
    warmup_numba,           # JIT-compilatie triggeren
    USE_NUMBA,              # True als Numba beschikbaar is
)
```

| Functie | Handtekening | Omschrijving |
|---|---|---|
| `build_combined_exp` | `(k_bulk, k_wall_vol, dt, n_pipes) → (n_pipes, n_species)` | Bereken `exp(-(k_bulk + k_wall_vol) * dt)`; eenmalig per hydraulica-update of dt-wijziging |
| `combined_decay_multi` | `(C, pipe, combined_exp, n=None) → None` | `C *= combined_exp[pipe]` in-place; één geheugenpass voor bulk- én wandverval |
| `bulk_first_order_multi` | `(C, k_vec, dt, n=None) → None` | `C *= exp(-k_vec * dt)` in-place; broadcasting over segmenten |
| `wall_first_order_multi` | `(C, pipe, k_wall, dt, n=None) → None` | Wandverval per segment in-place |
| `compute_wall_k` | `(diam, velocity, k_w, D_mol, nu) → (n_pipes, n_species) [1/s]` | Volumetrische wandreactiesnelheid via twee-film-model |
| `advect` | `(x, pipe, velocity, dt, n=None) → None` | `x += velocity[pipe] * dt` in-place |
| `exit_detect` | `(x, pipe, pipe_length, exit_mask, n=None) → int` | Vul `exit_mask` in-place; retourneert aantal exiterende segmenten |
| `node_mixing_multi` | `(exit_mask, pipe, C, flow, pipe_end, node_count, out=None, wC_buf=None, node_flow_buf=None, n=None) → ndarray` | Debietgewogen menging; geen allocaties als pre-allocated buffers worden meegegeven |
| `tank_step_implicit` | `(C_tank, Q_in, C_in, Q_out, V_tank, k_b, dt) → None` | CSTR impliciet Euler in-place |
| `warmup_numba` | `(n_species=1) → None` | Trigger JIT-compilatie; eenmalig aanroepen na aanmaken van de solver |

> **Prestatietip:** gebruik `combined_decay_multi` samen met `build_combined_exp` in plaats van aparte `bulk_first_order_multi` + `wall_first_order_multi` aanroepen. Dit halveert het aantal geheugenpassages over `C` en elimineert de `exp()`-berekening per tijdstap.

---

### GeochemSolver & SpeciesMap

PhreeqPython-integratie die eerste-orde bulkverval vervangt door volledige PHREEQC-geochemie.

> Vereist: `pip install nzingaflow[geochem]` of `pip install phreeqpython`

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
| `apply_geochemistry(store, dt, pipe_diam=None, pipe_vel=None)` | Vervangt `combined_decay_multi()`; PHREEQC-reacties per segment in batches |
| `apply_mixing(node_C, node_flow, dt)` | Geochemisch evenwicht na knoopmenging (pH-correctie) |
| `make_injection_solution(C_vector)` | Zet concentratieprofiel om naar PHREEQC-oplossingsdict |
| `langelier_index(C_vec)` | LSI voor CaCO₃ (> 0 = aankorsting, < 0 = corrosief) |
| `saturation_indices(C_vec)` | Verzadigingsindices voor Calcite, Dolomite, Goethite, e.a. |
| `reset()` | Reset PHREEQC-instantie bij herstart simulatie |

#### Voorgeconfigureerde recepten

```python
from nzingaflow.geochemistry import chlorine_decay_geochem, full_water_chemistry

# Chloor + pH (n_species=2)
geo = chlorine_decay_geochem(k_bulk_per_day=0.5)

# Chloor, pH, alkaliniteit, calcium, ijzer, mangaan (n_species=6)
geo = full_water_chemistry()
```

---

### MsxReactionSystem

MSX-compatibele multi-species reactielaag die de vier EPANET-MSX 2.0 concepten implementeert: `RATE`, `EQUIL`, `FORMULA` en oppervlaktesoorten (`WALL`). Geef een `MsxReactionSystem`-instantie mee als `geochem`-argument aan `NzingaFlowSolver`.

> Vereist: `pip install nzingaflow` (geen extra afhankelijkheid; scipy nodig voor `rk45`/`radau`-solvers)

```python
from nzingaflow.msx import MsxReactionSystem

rxn = MsxReactionSystem(
    bulk_species,          # lijst van bulk-soortsnamen
    wall_species=[],       # lijst van wandgebonden soortsnamen
    params={},             # {naam: waarde} reactieparameters
    pipe_rates={},         # {soort: expressie} in leidingen
    pipe_formulas={},      # {soort: expressie} afgeleide variabelen
    tank_rates={},         # {soort: expressie} in tanks
    solver='rk4',          # 'euler', 'rk4', 'ros2' (stijf, geen scipy), 'rk45', 'radau'
)
```

**Expressienotatie**

| Symbool | Omschrijving |
|---|---|
| Soortnaam | Bulkconcentratie, bijv. `Cl2`, `NH3` |
| Parameternaam | Scalaire constante, bijv. `k1`, `BFmax` |
| `Av` | Leidingoppervlak per volume [m²/m³] = 4/D — automatisch beschikbaar |
| `t` | Gesimuleerde tijd [s] |

**Numerieke solvers**

| Solver | Type | scipy nodig | Aanbevolen voor |
|---|---|---|---|
| `euler` | Expliciet, 1e orde | Nee | Snel; alleen niet-stijve systemen |
| `rk4` | Expliciet, 4e orde | Nee | Standaard voor de meeste systemen |
| `ros2` | Rosenbrock 2(1), adaptief | **Nee** | **Stijve systemen** (chloramine, biofilm) |
| `rk45` | Adaptief RK45 | Ja | Niet-stijf; variabele stapgrootte |
| `radau` | Impliciet Radau IIA | Ja | Stijf (legacy; gebruik `ros2`) |

**Expressie-parser:** string-expressies worden gecompileerd via een ingebouwde tokenizer/evaluator gebaseerd op EPANET-MSX `mathexpr.c` (Rossman/Shang/Uber — US EPA NRMRL). Geen sympy of eval() nodig. Compile-time validatie geeft `ValueError` bij onbekende variabelenamen.

**Wandsoorten**

Wandsoorten (bijv. biofilm `BF`) zijn gebonden aan de leidingwand en bewegen niet mee met het water. Ze koppelen aan bulksoorten via `RATE`-expressies met `Av`. Elk segment draagt zijn eigen wandconcentratie, geïnitialiseerd op nul.

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

**Methoden**

| Methode | Omschrijving |
|---|---|
| `apply_geochemistry(store, dt, pipe_diam, pipe_vel)` | Integreer reacties voor alle segmenten; vervangt `combined_decay_multi()` |
| `apply_mixing(node_C, node_flow, dt)` | Integreer tankreacties na knoopmenging |
| `get_wall_concentrations()` | Retourneer huidige wandconcentratie-array `(n_segmenten, n_wandsoorten)` |
| `set_wall_concentrations(C_wall)` | Overschrijf wandconcentraties (bijv. voor warm-start) |
| `set_pipe_param(pipe_idx, **kwargs)` | Overschrijf parameters voor één leiding |
| `reset()` | Reset wandconcentraties naar nul |

**Kant-en-klare MSX-modellen**

```python
from nzingaflow.msx import (
    chloramine_decay_msx,    # HOCl + NH3 → NH2Cl (Vikesland 2001); stoffen: [HOCl, NH3, NH2Cl]
    chlorine_nom_msx,        # Cl2 + NOM bulk/wand; stoffen: [Cl2, NOM]
    arsenic_oxidation_msx,   # AS3→AS5, wandadsorptie (Zhang 2004); stoffen: [AS3, AS5, NH2CL] + wand [AS5s]
)

rxn = chloramine_decay_msx(k_f=2.5e-4, k_ox=5e-5, solver='ros2')
rxn = chlorine_nom_msx(k_bulk=3e-4, k_wall=1e-5, solver='rk4')
rxn = arsenic_oxidation_msx(Ka=10.0, Kb=0.1, K1=5.0, K2=1.0, Smax=50.0, solver='radau')
```

---

### MsxSimulation & MsxNativeLib

Directe koppeling met de officiële EPANET-MSX C-bibliotheek via ctypes. Drie lagen boven de `MSX_*` C-API (EPANET-MSX 2.0).

> Native bibliotheek (`libepanetmsx`/`epanet2`) wordt meegeleverd in `nzingaflow/lib/` —
> Linux x86-64 en Windows x86-64 bijgesloten. Voor macOS: bouw zelf via de `CMakeLists.txt`
> uit [github.com/USEPA/EPANETMSX](https://github.com/USEPA/EPANETMSX) (ondersteunt Linux/macOS/Windows).

#### `MsxSimulation` — orchestrator (laag 3)

```python
MsxSimulation(
    inp_path:  str,           # pad naar EPANET .inp bestand
    msx_path:  str,           # pad naar EPANET-MSX .msx bestand
    lib_path:  str | None = None,  # expliciet pad naar libepanetmsx; None = auto-detectie
    strict:    bool = True,   # gooit MsxError bij niet-nul retourwaarden
)
```

**Methoden**

| Methode | Retourtype | Omschrijving |
|---|---|---|
| `load()` | `MsxNetworkState` | Open het MSX-bestand en bouw een volledig netwerk-snapshot op |
| `run(save_to_file=False, hyd_file=None)` | `MsxSimulationResult` | Voer volledige simulatie uit; optioneel via eerder opgeslagen hydraulicabestand |
| `update_initial_quality(node_values=None, link_values=None)` | `None` | Pas beginconcentraties aan vóór `run()`: `{naam: {stof: waarde}}` |
| `configure_source(node, species, kind, level, pattern_name=None)` | `None` | Stel bron in; `kind` = `'CONCEN'`, `'MASS'`, `'SETPOINT'`, `'FLOWPACED'` of `'NOSOURCE'` |
| `update_constant(name, value)` | `None` | Wijzig een named constant in het .msx-bestand |
| `add_time_pattern(name, multipliers)` | `None` | Voeg een nieuw tijdpatroon toe met de gegeven vermenigvuldigers |
| `close()` | `None` | Sluit de bibliotheek en geef geheugen vrij |

**Context-manager**

```python
with MsxSimulation("netwerk.inp", "netwerk.msx") as sim:
    state = sim.load()
    result = sim.run()
# sim.close() wordt automatisch aangeroepen
```

#### `MsxSimulationResult`

```python
result.time_s                        # ndarray — tijdstempels [s]
result.time_hours()                  # ndarray — tijdstempels [h]
result.node_quality                  # ndarray (T, N, S) — knoopconcentraties
result.link_quality                  # ndarray (T, L, S) — leidingconcentraties
result.node_names                    # list[str]
result.link_names                    # list[str]
result.species                       # list[MsxSpecies]

result.node_concentrations("CL2")   # ndarray (T, N) — één stof uit node_quality
result.link_concentrations("CL2")   # ndarray (T, L) — één stof uit link_quality
result.to_dataframe("CL2", element="node")  # pandas DataFrame: index=tijd_h, kolommen=knoopnamen
```

#### `MsxNetworkState`

Snapshot van het netwerk na `load()`. Bevat stoffen, constanten, bronnen en patronen.

```python
state.species               # list[MsxSpecies]
state.constants             # dict[str, float]
state.sources               # list[MsxSourceRecord]
state.node_initq            # ndarray (N, S) — initiële kwaliteit knopen
state.link_initq            # ndarray (L, S) — initiële kwaliteit leidingen
state.patterns              # dict[int, list[float]]

state.species_by_name("CL2")   # → MsxSpecies
state.bulk_species()            # → list[MsxSpecies]
state.wall_species()            # → list[MsxSpecies]
```

#### `MsxNativeLib` — ctypes-wrapper (laag 1)

Voor gevorderd gebruik: directe toegang tot alle `MSX_*` C-functies.

```python
from nzingaflow import MsxNativeLib

lib = MsxNativeLib(lib_path=None, strict=True)
lib.open("netwerk.msx")
lib.solve_hydraulics()
lib.solve_quality()

n_sp = lib.object_count(3)               # ObjectType.SPECIES = 3
name = lib.object_id(3, 1)              # naam van stof 1
conc = lib.concentration(0, 1, 1)       # NODE=0, knoop 1, stof 1

lib.set_constant(idx, waarde)
lib.set_initial_quality(obj_type, idx, sp_idx, waarde)
lib.set_source(knoop, stof, type, niveau, patroon)
lib.close()
```

#### Keuzehulp: `msx.py` of `msxlibrary.py`?

| Situatie | Aanbevolen module |
|---|---|
| Reactievergelijkingen zelf schrijven in Python | `msx.py` — `MsxReactionSystem` |
| Bestaand `.msx`-bestand hergebruiken | `msxlibrary.py` — `MsxSimulation` |
| Geen native bibliotheek beschikbaar | `msx.py` (geen afhankelijkheid) |
| MSX-C-solver moet tijdintegratie verzorgen | `msxlibrary.py` |
| Koppeling met `NzingaFlowSolver` via `geochem=` | `msx.py` — `MsxReactionSystem` |
| Standalone MSX-simulatie zonder LTA | `msxlibrary.py` — `run_msx()` |

---

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
solver = NzingaFlowSolver("netwerk.inp", n_species=1, track_mass=True)
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
from nzingaflow import merge_segments

n_verwijderd = merge_segments(
    store,               # SegmentStore
    tol=1e-6,            # concentratietolerantie [mg/L]
    min_volume=1e-12,    # segmenten kleiner dan dit worden altijd verwijderd [m³]
    n_pipes=0,           # leidingaantal voor CSR-index (0 = auto-detectie)
)
```

Algoritme: iteratieve parallel-reductie (O(n log n)); Numba JIT-kernels met early-exit kandidaatdetectie. Wordt automatisch uitgevoerd door de solver elke `merge_interval` stappen.

---

## Geavanceerd gebruik

### Numba warmup

```python
solver = NzingaFlowSolver("netwerk.inp", n_species=2)
solver.warmup_numba()   # ~1 s eenmalig; daarna zijn stappen snel

runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(decay_k=np.array([0.001, 0.0]))
```

Of gebruik de losse functies:

```python
from nzingaflow import warmup_numba, warmup_numba_merging, USE_NUMBA

print(f"Numba beschikbaar: {USE_NUMBA}")
warmup_numba(n_species=2)
warmup_numba_merging(n_species=2)
```

### Tijdvariabele injectie met callback

```python
def injectie(t: float, solver: NzingaFlowSolver):
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
solver = NzingaFlowSolver("netwerk.inp", n_species=1)
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
solver = NzingaFlowSolver("netwerk.inp", n_species=1, track_mass=True)
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

### Laag-niveau: gecombineerd verval direct gebruiken

```python
from nzingaflow.lta import build_combined_exp, combined_decay_multi

# Eenmalig bouwen per hydraulica-update (of bij dt-wijziging)
cexp = build_combined_exp(
    k_bulk=np.array([0.001, 0.0]),
    k_wall_vol=solver._k_wall_vol,   # (n_pipes, n_species) [1/s]
    dt=5.0,
    n_pipes=len(solver.pipe_ids),
)

# Elke tijdstap toepassen — één geheugenpass, geen exp()-berekening
combined_decay_multi(solver.segments.C, solver.segments.pipe, cexp,
                     n=solver.segments.n)
```

---

## Wandreacties

NzingaFlow implementeert het twee-film-model van Rossman (1994), uitgebreid in v1.1.0 met een drieregime Sherwood-correlatie, temperatuurafhankelijkheid en stagnatie-correctie.

### Filmtransportmodel

```
Re    = v · D / ν
Sh    = 0.023 · Re^0.83 · Sc^(1/3)                                (turbulent, Re > 4000)
Sh    = (3.66³ + max(1.615·(Re·Sc·D/L)^(1/3) − 0.7, 0)³)^(1/3)  (laminair, Re < 2300)
Sh    = lineaire interpolatie Sh_lam ↔ Sh_turb                    (transitie, 2300 ≤ Re ≤ 4000)
k_f   = max(Sh · D_mol / D, 4·D_mol/D)   [m/s]   (stagnatie-minimum)
k_eff = k_f · k_w / (k_f + k_w)                  (serieschakeling, wall_mode='two_film')
k_vol = k_eff · 4 / D                    [1/s]    (cilinder A/V = 4/D)
```

**Wijzigingen t.o.v. v1.0.0:** de vroegere harde drempel bij Re=4000 gaf bij nachtsituaties in DN100-leidingen een factor 1.5× fout. Het stagnatie-minimum voorkomt onderschatting bij nagenoeg nul-snelheid.

### Temperatuurafhankelijkheid (optioneel, v1.1.0)

Geef `temperature` [°C] mee aan `NzingaFlowSolver` om te activeren:

```
D_mol(T) = D_mol_20 · exp(17000/R · (1/T₀ − 1/T))   [Hayduk & Laudie 1974]
k_wall(T) = k_wall · 1.047^(T−20)                    [Rossman 2000]
```

Effect: factor 0.59× bij 5°C tot 1.38× bij 30°C op k_wall_vol.

### k_wall per leiding opgeven

```python
n_pipes = len(solver.pipe_ids)

# Uniform
k_wall = np.full((n_pipes, 1), 5e-6)   # [m/s]

# Per leiding variabel (bijv. op basis van materiaal)
k_wall = np.zeros((n_pipes, 1))
k_wall[0:20] = 1e-5   # gietijzer
k_wall[20:]  = 2e-6   # PVC

solver = NzingaFlowSolver("netwerk.inp", n_species=1, k_wall=k_wall)

# 'direct' modus: k_wall is al de effectieve snelheid (geen filmweerstand)
solver = NzingaFlowSolver("netwerk.inp", n_species=1, k_wall=k_wall, wall_mode='direct')
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

**Q: Hoe haal ik de beste prestaties?**

1. Installeer Numba: `pip install nzingaflow[numba]`
2. Roep `solver.warmup_numba()` eenmalig aan na het aanmaken van de solver, vóór de simulatielus
3. Gebruik `combined_decay_multi` + `build_combined_exp` in plaats van aparte bulk/wand-aanroepen (de solver doet dit automatisch)

**Q: Hoe voeg ik een nieuw PHREEQC-element toe?**

```python
smap = SpeciesMap(
    species_names=["Cl2","pH","Alk","Ca","Fe","Mn","SO4"],
    phreeqc_names=["Cl", None,"Alk","Ca","Fe","Mn","S(6)"],
    units        =["mg/L","","meq/L","mg/L","mg/L","mg/L","mg/L"],
    is_pH        =[False,True,False,False,False,False,False],
)
```

**Q: Wanneer gebruik ik MsxReactionSystem in plaats van GeochemSolver?**

Gebruik `MsxReactionSystem` als uw reacties uit te drukken zijn als gewone differentiaalvergelijkingen (eerste- of hogere-orde kinetiek, biofilmgroei, chloramineverval). Het is sneller dan PhreeqPython en vereist geen extra afhankelijkheden. Gebruik `GeochemSolver` als u volledige thermodynamisch evenwicht, mineraaloplossing/-precipitatie of pH-buffering via PHREEQC nodig heeft.

**Q: Wanneer gebruik ik MsxSimulation (msxlibrary) in plaats van MsxReactionSystem (msx.py)?**

Gebruik `MsxSimulation` als u een bestaand `.msx`-bestand ongewijzigd wilt uitvoeren via de officiële EPANET-MSX C-solver. Dit is handig als de reactievergelijkingen al in een `.msx`-bestand staan, of als u resultaten wilt vergelijken met de referentie-MSX-implementatie. `MsxSimulation` werkt als standalone simulator en koppelt niet met `NzingaFlowSolver`.

Gebruik `MsxReactionSystem` als u reacties in Python wilt definiëren en koppelen aan de NzingaFlow Lagrangian solver via `geochem=rxn`. Dit vereist geen native bibliotheek en is flexibeler voor parameterstudies.

**Q: Hoe modelleer ik leidinglekkage?**

Geef `leakage_fraction` mee aan `NzingaFlowSolver`. Een waarde van `0.12` betekent 12% volumeverlies per leiding. Elk segment verliest per tijdstap proportioneel volume; concentraties blijven ongewijzigd (conservatief mengmodel — geen contaminantinstroom vanuit grondwater). Typische waarden voor Nederlandse distributienetwerken liggen tussen 0.05 en 0.20.

**Q: Worden pompen gesimuleerd?**

Standaard worden pompen overgeslagen (`include_pumps=False`). U kunt pompen includeren via `HydraulicModel("net.inp", include_pumps=True)`, maar dit heeft doorgaans alleen zin als de verblijftijd in de pomp relevant is.

**Q: Worden afsluiters (kleppen) meegenomen in de transportberekening?**

Standaard niet (`include_valves=False`). EPANET lost de hydraulica voor afsluiters altijd correct op, maar NzingaFlow sloot ze traditioneel uit de Lagrangian topologie. Met `include_valves=True` worden alle EPANET-afsluitertypes (PRV, PSV, PBV, FCV, TCV, GPV, PCV) behandeld als korte leidingelementen, zodat verblijftijd en stoftransport ook over afsluiters worden berekend. Dit is met name relevant als een afsluiter de enige verbinding vormt tussen twee netwerksegmenten.

```python
solver = NzingaFlowSolver("netwerk.inp", include_valves=True)
```

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
| **CSR** | Compressed Sparse Row: indexformaat van `SegmentStore.build_csr()` voor gecachede per-leiding sortering |
| **EPS** | Extended Period Simulation: tijdvariabele simulatie met periodiek bijgewerkte hydraulica |
| **LTA** | Lagrangian Transport Approach: transport vanuit het referentiekader van de vloeistof |
| **LSI** | Langelier Saturation Index: maat voor CaCO₃-verzadiging |
| **MSX** | Multi-Species eXtension: EPANET-MSX-compatibele reactielaag voor willekeurige kinetische systemen |
| **MSX native bridge** | Directe ctypes-koppeling met `libepanetmsx`; implementatie in `msxlibrary.py` |
| **PHREEQC** | Geochemisch rekenprogramma van de USGS |
| **SoA** | Structure of Arrays: opslagopzet waarbij elke eigenschap een aparte array is |
| **`k_bulk`** | Bulkvervalconstante [1/s]: eerste-orde verval in de waterkolom |
| **`k_wall`** | Wandreactiesnelheid [m/s]: reactie aan de binnenzijde van de leiding |
| **`combined_exp`** | Gecombineerde vervalfactor-tabel `(n_pipes × n_species)` = `exp(-(k_bulk + k_wall_vol) × dt)`; voorberekend door `build_combined_exp()` |
| **`leakage_fraction`** | Fractie van leidingdebiet dat per tijdstap lekt; segmenten krimpen zonder concentratieverandering |
| **`temperature`** | Watertemperatuur [°C] voor Arrhenius/Hayduk-Laudie correcties op diffusiviteit en k_wall |
| **segment** | Lagrangiaans vloeistofelement met vast volume, positie en concentratieprofiel |

---

## Versiehistorie

### 1.2.2

- **MSX native library bridge — meegeleverde binaries + kritieke bugfixes.** `msxlibrary.py` (`MsxSimulation`/`MsxNativeLib`) was sinds de introductie in 1.2.0 in de praktijk niet functioneel: er was geen native bibliotheek op het systeem beschikbaar, en zelfs met de bibliotheek aanwezig faalde de brug alsnog door meerdere onderliggende bugs.
- Native binaries worden nu meegeleverd in `nzingaflow/lib/`: `libepanetmsx.so`/`libepanet2_msx.so` (Linux x86-64) en `epanetmsx.dll`/`epanet2_msx.dll` (Windows x86-64), gebouwd vanuit de officiële EPANET-MSX 2.0-broncode. macOS nog niet meegebouwd — zie `nzingaflow/lib/README.txt`.
- Fix: `MSXstep` gebruikte `c_long` in plaats van `c_double` voor de tijdsparameters (ABI-mismatch t.o.v. de MSX 2.0 C-API).
- Fix: `_epanet_open()` was een no-op — `ENopen()` werd nooit aangeroepen vóór `MSXopen()`, terwijl MSX daarvan afhankelijk is voor de gedeelde netwerk-state. Nu gekoppeld via nieuwe `en_open()`/`en_close()`-methoden.
- Fix: node-/link-tellingen en -namen liepen via `MSXgetcount`/`MSXgetID`, die dat objecttype niet ondersteunen — gaf overal `MSX fout 515`. Omgezet naar de juiste EPANET-laag (`ENgetcount`, `ENgetnodeid`, `ENgetlinkid`, `ENgetnodeindex`, `ENgetlinkindex`).
- Fix: de epanet2-companion-bibliotheek deelde haar naam (en op Linux: haar SONAME) met epynet's eigen bundled epanet2-bibliotheek, waardoor beide tegelijk in hetzelfde proces stilzwijgend naar het verkeerde, al-geladen exemplaar resolveerden. Hernoemd naar `libepanet2_msx.so`/`epanet2_msx.dll`.
- Gecorrigeerd: eerdere documentatie verwees naar EPANET-MSX 1.1 (de bron van de `MSXstep`-bug); deze brug is gebouwd tegen 2.0.
- Nieuwe `tests/test_msxlibrary.py`, tegen het officiële arseenoxidatie-voorbeeldnetwerk, inclusief regressietests voor bovenstaande bugs.
- Achterwaarts compatibel: de publieke API van `MsxSimulation`/`MsxNativeLib` is ongewijzigd.

### 1.2.1

- **Afsluiters in transporttopologie** (`include_valves`-parameter): alle EPANET-afsluitertypes (PRV, PSV, PBV, FCV, TCV, GPV, PCV) kunnen nu samen met leidingen worden opgenomen in de Lagrangian transporttopologie via `include_valves=True` op `HydraulicModel` en `NzingaFlowSolver`. Voorheen sloot NzingaFlow afsluiters altijd uit, wat geen verblijftijd/verval over afsluiterelementen berekende en topologische ontkoppeling kon veroorzaken wanneer een afsluiter de enige verbinding tussen twee netwerksegmenten was. Standaard `False`; achterwaarts compatibel.
- **Bugfix — `include_pumps=True` crashte op `diameter`**: `HydraulicModel.get_topology()` gebruikte `lnk.diameter` zonder fallback, terwijl epynet `Pump`-objecten geen `diameter` static property hebben. Elk netwerk met een echte pomp gaf hierdoor een `AttributeError` zodra `include_pumps=True` werd gebruikt. Opgelost met dezelfde `getattr(..., 0.0)`-fallback die al voor de lengte van afsluiters werd gebruikt; pompen worden nu behandeld als oppervlakteloze elementen.
- Regressietests voor beide punten toegevoegd in `tests/test_epynet_networks.py` (echte EPANET `.inp`-netwerken via epynet, gemarkeerd `requires_epynet`).

### 1.2.0

- **Nieuwe module `msxlibrary.py`** — directe ctypes-brug naar de officiële EPANET-MSX C-bibliotheek (`libepanetmsx.so` / `epanetmsx.dll`).
- **`MsxNativeLib`** — dunne wrapper rond alle `MSX_*` C-functies; automatische signatuurbinding en foutvertaling naar `MsxError`.
- **`MsxSimulation`** — hoog-niveau orchestrator met `load()`, `run()`, context-manager en schrijfhulpers (`update_initial_quality`, `configure_source`, `update_constant`, `add_time_pattern`).
- **`MsxSimulationResult`** — tijdreeksen `(T × N × S)` en `(T × L × S)` met `node_concentrations()`, `link_concentrations()`, `to_dataframe()`.
- **`MsxNetworkState`**, **`MsxSpecies`**, **`MsxSourceRecord`** — gestructureerde dataklassen voor netwerk-snapshots na `load()`.
- **`run_msx(inp, msx)`** — volledige simulatie in één aanroep.
- Automatische bibliotheekdetectie voor Windows, Linux en macOS.
- Alle nieuwe klassen geëxporteerd vanuit `nzingaflow.__init__`.
- Achterwaarts compatibel: bestaande `MsxReactionSystem`-code werkt ongewijzigd.

### 1.1.0

- **Drieregime Sherwood-correlatie** in `compute_wall_k()`: laminair (Graetz + entry-length), transitie (lineaire interpolatie), turbulent (Dittus-Boelter). Elimineert factor 1.5× fout bij nachtsituaties in DN100-leidingen.
- **Temperatuurcorrectie** (`temperature=`-parameter): Arrhenius-correctie op D_mol (Hayduk & Laudie 1974) en θ=1.047-correctie op k_wall (Rossman 2000). Effect: 0.59× bij 5°C tot 1.38× bij 30°C.
- **Stagnatie-minimum** in `compute_wall_k()`: minimaal filmtransport `k_f_min = 4·D_mol/D` bij nagenoeg nul-snelheid.
- **Lekkagemodellering** (`leakage_fraction=`-parameter): proportioneel volumeverlies per segment per tijdstap zonder contaminantinstroom.
- **`wall_mode`-parameter**: `'two_film'` (standaard, EPANET-compatibel) of `'direct'` (k_wall is al k_eff).
- **`MsxReactionSystem`** (`nzingaflow.msx`): EPANET-MSX 2.0-compatibele reactielaag met `RATE`, `EQUIL`, `FORMULA` en wandsoorten. Vijf numerieke solvers: `euler`, `rk4`, `ros2` (stijf, geen scipy), `rk45`, `radau`.
- **Ingebouwde expressie-parser** (gebaseerd op EPANET-MSX `mathexpr.c`): geen sympy of eval() nodig; compile-time validatie.
- **ROS2-solver** (gebaseerd op EPANET-MSX `ros2.c`, Verwer et al. 1999): adaptief Rosenbrock 2(1) zonder scipy-afhankelijkheid.
- **Kant-en-klare MSX-modellen**: `chloramine_decay_msx`, `chlorine_nom_msx`, `arsenic_oxidation_msx`.
- Alle nieuwe parameters zijn achterwaarts compatibel; defaults reproduceren v1.0.0-gedrag.

### 1.0.0

- Eerste release
- Vectorized LTA, multi-species, wandreacties, tanks, EPS
- GeochemSolver met PhreeqPython-integratie
- MassBalanceTracker, CFL-controle
- Parallel-reductie segmentmerging (O(n log n))
- Flow reversal detectie en segmentspiegeling
- Numba JIT-kernels voor alle hot-path operaties (~4–5× sneller)
- Gecombineerd bulk + wandverval in één geheugenpass (`combined_decay_multi` / `build_combined_exp`)
- Vectorized junction-routing via `_node_type` / `_node_out0`-cache (~734× t.o.v. Python-lus)
- CSR pipe-index in `SegmentStore` (gecachede lexsort voor `merge_segments`)
- Pre-allocated solver-buffers; geen heap-allocaties in hot-path
- `exit_detect()` als expliciete pre-allocated functie
- `warmup_numba()` / `warmup_numba_merging()` voor voorspelbare opstarttijd
- `on_resize`-callback op `SegmentStore` voor synchronisatie van externe buffers
