# NzingaFlow

**Lagrangian Transport kwaliteitssimulator voor EPANET-drinkwaternetwerken**

NzingaFlow implementeert de Lagrangian Transport Approach (LTA) voor waterkwaliteitssimulatie in drinkwaternetwerken. Het koppelt aan EPANET via [epynet](https://github.com/vitens/epynet) en ondersteunt multi-species, wandreacties, tanks (CSTR) en volledige geochemie via PhreeqPython.

---

## Installatie

### Minimaal (NumPy-only)
```bash
pip install nzingaflow
```

### Met Numba JIT (~4-5× sneller)
```bash
pip install "nzingaflow[numba]"
```

### Met volledige geochemie (PhreeqPython/PHREEQC)
```bash
pip install "nzingaflow[geochem]"
```

### Alles inclusief ontwikkeltools
```bash
pip install "nzingaflow[all]"
```

### Vanuit source (ontwikkeling)
```bash
git clone https://github.com/nzingaflow/nzingaflow.git
cd nzingaflow
pip install -e ".[dev]"
```

> **Setuptools < 68.2 (bijv. Ubuntu 24.04 systeempython)**  
> Als je `ModuleNotFoundError: No module named 'setuptools.backends'` krijgt, voeg dan `--no-build-isolation` toe:
> ```bash
> pip install --no-build-isolation -e ".[dev]"
> ```

---

## Snelstart

```python
import numpy as np
from nzingaflow import NzingaFlowSolver, EPSRunner

# Laad netwerk en maak solver aan
solver = NzingaFlowSolver("netwerk.inp", n_species=1)

# Optioneel: Numba warmup vóór simulatie (eenmalig ~1s compilatie)
solver.warmup_numba()

# Maak EPS-runner aan
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)

# Voer simulatie uit
results = runner.run(
    decay_k=np.array([0.001]),               # bulk vervalconstante [1/s]
    inject_schedule={"R1": [(0, 86400, np.array([1.0]))]},
    verbose=True,
)
# results.shape == (n_stappen, node_count, 1)
```

### Multi-species met wandreacties

```python
import numpy as np
from nzingaflow import NzingaFlowSolver, EPSRunner

n_pipes = 150  # aantal leidingen in het netwerk

solver = NzingaFlowSolver(
    "netwerk.inp",
    n_species=2,
    k_wall=np.array([[1e-5, 0.0]] * n_pipes),  # [m/s] per pipe per stof
)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie

runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(
    decay_k=np.array([0.001, 0.0]),
    inject_schedule={
        "R1": [(0, 86400, np.array([1.0, 0.5]))],
    },
)
```

### Volledige waterchemie (PhreeqPython)

```python
from nzingaflow import NzingaFlowSolver, EPSRunner
from nzingaflow.geochemistry import full_water_chemistry

geo = full_water_chemistry()  # chloor, pH, alkaliniteit, Ca, Fe, Mn
solver = NzingaFlowSolver("netwerk.inp", n_species=6, geochem=geo)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie
runner = EPSRunner(solver, qual_dt=10.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(decay_k=np.zeros(6))
```

### Multi-species waterchemie met MSX-reactielagen

```python
import numpy as np
from nzingaflow import NzingaFlowSolver, EPSRunner
from nzingaflow.msx import chloramine_decay_msx

# Chloramine-verval: HOCl + NH3 → NH2Cl  (stijf systeem → ROS2)
rxn = chloramine_decay_msx(k_f=2.5e-4, k_ox=5.0e-5, solver='ros2')

solver = NzingaFlowSolver(
    "netwerk.inp",
    n_species = len(rxn.bulk_species),  # 3: HOCl, NH3, NH2Cl
    geochem   = rxn,
)
solver.warmup_numba()   # optioneel: JIT-compilatie triggeren vóór simulatie
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(
    decay_k         = np.zeros(3),
    inject_schedule = {"R1": [(0, 86400, np.array([1.0, 0.5, 0.0]))]},
    verbose         = True,
)
# results.shape == (n_stappen, node_count, 3)
```

Zie [MSX-reactielagen](#msx-reactielagen) voor alle beschikbare modellen en het gebruik van eigen reactievergelijkingen.

---

## Performance

| Configuratie | µs/tijdstap | Noot |
|---|---|---|
| NumPy (standaard) | ~280 | n=5000 segs, 2 stoffen |
| Numba JIT | ~60–90 | Na warmup (~1s eenmalig) |

Numba-compilatie vindt plaats bij de eerste aanroep. Gebruik `solver.warmup_numba()` om dit vóór de simulatie te doen.

---

## Architectuur

```
nzingaflow/
├── solver.py        NzingaFlowSolver — hoofd-API, EPS-koppeling
├── lta.py           Kernberekeningen (advect, decay, mixing) met Numba JIT
├── merging.py       Segment-merging met Numba JIT
├── segments.py      SegmentStore — Structure-of-Arrays opslag + CSR-index
├── hydraulics.py    HydraulicModel — epynet koppeling
├── eps.py           EPSRunner — Extended Period Simulation
├── stability.py     CFL-check, massabalans
├── geochemistry.py  GeochemSolver — PhreeqPython integratie
├── msx.py           MsxReactionSystem — pure-Python MSX reactielaag
│                      (ingebouwde parser + ROS2-solver, geen scipy/sympy)
├── msxlibrary.py   MsxSimulation — directe brug naar libepanetmsx.so/.dll
│                      (ctypes wrapper; laadt .msx-bestanden ongewijzigd)
└── parse_inp.py     EPANET .inp parser (fallback zonder epynet)
```

**Twee complementaire MSX-lagen:**

| Module | Wanneer gebruiken | Native lib nodig |
|---|---|---|
| `msx.py` — `MsxReactionSystem` | Eigen reactievergelijkingen in Python; plugt in via `geochem=` | Nee |
| `msxlibrary.py` — `MsxSimulation` | Bestaand `.msx`-bestand hergebruiken; MSX-C-solver zelf laten integreren | Nee — meegeleverd in `nzingaflow/lib/` (Linux x86-64, Windows x86-64; macOS nog niet) |

### Kernprincipes

- **Lagrangiaanse advectie**: segmenten bewegen met de stroming; geen numerieke diffusie voor puls-transport
- **Vectorized NumPy**: alle operaties zonder Python-loops over segmenten
- **Numba JIT**: optionele native-code compilatie voor 4-5× speedup
- **Pre-allocated buffers**: geen heap-allocaties in de hot-path
- **CSR pipe-index**: gecachede gesorteerde index voor efficiënte merging
- **Gecombineerde decay**: bulk + wand in één geheugenpass

---

## API-overzicht

### NzingaFlowSolver

```python
solver = NzingaFlowSolver(
    inp_path,           # pad naar EPANET .inp bestand
    n_species=1,        # aantal te simuleren stoffen
    capacity=200_000,   # initiële segmentcapaciteit
    k_wall=None,        # (n_pipes, n_species) wandreactiesnelheid [m/s]
    track_mass=False,   # massabalans bijhouden
    geochem=None,       # GeochemSolver instantie
    include_valves=False,  # afsluiters meenemen in transporttopologie (v1.2.1)
)

solver.warmup_numba()                      # JIT-compilatie triggeren
solver.inject("R1", C_vector, volume)      # injecteer in knoop
solver.update_hydraulics(simtime=3600)     # hydraulica bijwerken
node_C = solver.step(dt=5.0, decay_k=k)   # één kwaliteitstijdstap
report = solver.mass_balance()             # massabalansrapport
```

### EPSRunner

```python
runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
results = runner.run(
    decay_k,
    inject_schedule=None,    # {node: [(t_start, t_end, C_vec), ...]}
    inject_fn=None,          # callable(t, solver)
    booster_schedule=None,   # {node: [(t_start, t_end, C_set), ...]}
    merge_interval=10,
    merge_tol=1e-6,
    verbose=False,
)
t = runner.time_axis(unit='h')
```

### MSX native library bridge (ctypes → libepanetmsx)

Directe koppeling met de officiële EPANET-MSX C-bibliotheek. `libepanetmsx`/`epanet2` worden meegeleverd in `nzingaflow/lib/` (Linux x86-64 en Windows x86-64); voor macOS moet je de bibliotheek zelf bouwen (zie EPANETMSX's eigen `CMakeLists.txt`, die Linux/macOS/Windows ondersteunt).

```python
from nzingaflow import MsxSimulation, run_msx

# Hoog-niveau: alles in één aanroep
result = run_msx("net2-cl2.inp", "net2-cl2.msx")
df = result.to_dataframe("CL2", element="node")   # pandas DataFrame (tijd × knopen)

# Context-manager voor fijnere controle
with MsxSimulation("net2-cl2.inp", "net2-cl2.msx") as sim:
    state = sim.load()
    print([s.name for s in state.bulk_species()])   # ['CL2']

    # Initiële kwaliteit aanpassen voor simulatie
    sim.update_initial_quality(
        node_values={"J1": {"CL2": 0.8}, "J2": {"CL2": 0.5}}
    )

    # Bronsterkte aanpassen
    sim.configure_source("R1", "CL2", kind="CONCEN", level=1.0)

    result = sim.run()

# Tijdreeksen opvragen
print(result.time_hours())                         # array van tijdstippen [h]
print(result.node_concentrations("CL2"))           # array (T × N)
print(result.link_concentrations("CL2"))           # array (T × L)
```

**Beschikbare klassen en functies:**

| Naam | Type | Omschrijving |
|---|---|---|
| `MsxSimulation` | klasse | Hoog-niveau orchestrator: `load()` → `run()` → resultaat |
| `MsxSimulationResult` | dataclass | Tijdreeksen + hulpmethoden (`node_concentrations`, `to_dataframe`) |
| `MsxNetworkState` | dataclass | Snapshot na laden: stoffen, constanten, bronnen, patronen |
| `MsxSpecies` | dataclass | Metadata per stof (naam, bulk/wand, eenheden, toleranties) |
| `MsxSourceRecord` | dataclass | Brondefinitie per (knoop, stof)-paar |
| `MsxNativeLib` | klasse | Dunne ctypes-wrapper rond alle `MSX_*` C-functies |
| `MsxError` | uitzondering | Niet-nul foutcode vanuit de C-bibliotheek |
| `run_msx(inp, msx)` | functie | Volledige simulatie in één aanroep |

### MSX-reactielagen

```python
from nzingaflow.msx import (
    MsxReactionSystem,
    chloramine_decay_msx,    # HOCl + NH3 → NH2Cl  (Vikesland 2001)
    chlorine_nom_msx,        # Cl2 + NOM bulk- en wandreactie
    arsenic_oxidation_msx,   # AS3→AS5 + wandadsorptie (Zhang 2004)
)

# Voorgeconfigureerd model
rxn = chloramine_decay_msx(solver='ros2')   # stijf → ROS2 (geen scipy)

# Eigen reactiesysteem
rxn = MsxReactionSystem(
    bulk_species = ['Cl2', 'NOM'],
    params       = {'k_b': 3e-4, 'k_w': 1e-5},
    pipe_rates   = {
        'Cl2': '-k_b * Cl2 * NOM - k_w * Cl2 * Av',
        'NOM': '-k_b * Cl2 * NOM',
    },
    tank_rates   = {'Cl2': '-k_b * Cl2 * NOM', 'NOM': '-k_b * Cl2 * NOM'},
    solver = 'rk4',
)

# Per-leiding parameteroverride (MSX [PARAMETERS])
rxn.set_pipe_param(5,  k_w=2e-6)   # gietijzer
rxn.set_pipe_param(12, k_w=5e-7)   # PVC
```

**ODE-solvers:**

| Solver | Methode | Scipy nodig | Aanbevolen voor |
|---|---|---|---|
| `euler` | Voorwaarts Euler | Nee | Eenvoudige, niet-stijve systemen |
| `rk4` | Runge-Kutta 4e orde | Nee | Standaard niet-stijf (standaard) |
| `ros2` | Rosenbrock 2(1) adaptief | **Nee** | **Stijve systemen** (chloramine, biofilm) |
| `rk45` | Dormand-Prince adaptief | Ja | Niet-stijf met foutcontrole |
| `radau` | Radau IIA | Ja | Stijf (achterwaarts compatibel) |

**Expressie-parser:** string-expressies worden gecompileerd via een ingebouwde parser gebaseerd op EPANET-MSX `mathexpr.c` (Rossman/Shang/Uber — US EPA). Geen sympy nodig. Ondersteunt `+ - * / ^ ()` en wiskundige functies (`abs`, `sqrt`, `exp`, `log`, `sin`, `cos`, `step`, etc.).

### LTA-kernfuncties (laag-niveau)

```python
from nzingaflow import (
    combined_decay_multi,    # bulk + wand verval in één pass
    build_combined_exp,      # bereken gecombineerde vervalfactoren
    advect,                  # x += v[pipe] * dt
    exit_detect,             # detecteer segmenten die leiding verlaten
    node_mixing_multi,       # debietgewogen knoopmenging
    warmup_numba,            # JIT-compilatie triggeren
    USE_NUMBA,               # True als Numba beschikbaar is
)
```

---

## Testen

```bash
# Alle validatietests (wetenschappelijke verificatie)
pytest tests/

# Snelle tests
pytest tests/ -m "not slow"

# Met coverage
pytest tests/ --cov=nzingaflow --cov-report=html
```

De testsuite bevat 10 validatieklassen (V1–V10) met analytische referentieoplossingen:

- **V1**: Enkelvoudige leiding — eerste-orde verval
- **V2**: Massabehoud — conservatieve tracer
- **V3**: Knoopmenging — debietgewogen verdunning
- **V4**: Tankmodel — CSTR steady-state en transiënt
- **V5**: Wandverval — twee-film model
- **V6**: CFL-stabiliteit en convergentie
- **V7**: Serienetwerk — stapelverval
- **V8**: Splitsingsknoop — massabehoud
- **V9**: Tijdstap-gevoeligheidsanalyse
- **V10**: Segmentmerging — massabehoud

---

## Vereisten

- Python ≥ 3.10
- NumPy ≥ 1.24
- epynet ≥ 1.1

Optioneel:
- numba ≥ 0.57 — voor ~4-5× snellere kernels
- phreeqpython ≥ 1.4 — voor volledige geochemie (PHREEQC)
- scipy ≥ 1.10 — voor validatietests en MSX-solvers `rk45`/`radau`
- `libepanetmsx`/`epanet2` — meegeleverd (`nzingaflow/lib/`) voor `MsxSimulation`; Linux x86-64 en Windows x86-64 bijgesloten, macOS (nog) niet

> **MSX zonder scipy:** de ingebouwde `ros2`- en `rk4`-solvers in `MsxReactionSystem` vereisen alleen NumPy.
> String-expressies vereisen geen sympy.
>
> **MSX zonder native bibliotheek:** `MsxReactionSystem` (`msx.py`) werkt zonder `libepanetmsx`.
> `MsxSimulation` (`msxlibrary.py`) heeft de meegeleverde native bibliotheek nodig (Linux/Windows bijgesloten; macOS zelf bouwen).

---

## Licentie

MIT License — zie [LICENSE](LICENSE) voor details.
