# Changelog

## [1.1.0] — 2026

### Verbeterd wandreactiemodel — drie correcties met meetbaar effect

**`lta.py` — `compute_wall_k()` herschreven**

1. **Sherwood-correlatie: drie regimes** (was: harde drempel bij Re=4000)
   - Laminair (Re < 2300): Graetz-oplossing met entry-length correctie
     `Sh = (3.66³ + max(1.615·(Re·Sc·D/L)^(1/3) − 0.7, 0)³)^(1/3)`
     Basiswaarde Sh=3.66 (uniforme wandconcentratie) i.p.v. 3.65 (uniforme flux).
     Entry-length correctie actief als `pipe_length` wordt meegegeven.
   - Transitie (2300 ≤ Re ≤ 4000): lineaire interpolatie Sh_lam ↔ Sh_turb.
     De vroegere harde drempel bij Re=4000 gaf bij nachtsituaties (v~0.03 m/s
     in DN100) een factor 1.5× fout op k_wall_vol.
   - Turbulent (Re > 4000): Dittus-Boelter `0.023·Re^0.83·Sc^(1/3)` — ongewijzigd.

2. **Temperatuurafhankelijkheid** (nieuw, optioneel via `temperature=[°C]`)
   - `D_mol(T) = D_mol_20 · exp(17000/R · (1/T₀ − 1/T))`  [Hayduk & Laudie 1974]
   - `k_wall(T) = k_wall · 1.047^(T−20)`  [Rossman 2000]
   - Effect: factor 0.59× (5°C) tot 1.38× (30°C) op k_wall_vol.
   - Relevant bij seizoensvaratie of geotemperatuur-gegevens.

3. **Stagnatie-zone correctie** (nieuw, automatisch)
   - Bij stilstaand water (Re → 0) domineert moleculaire diffusie over convectie.
   - Minimum filmtransport: `k_f_min = 4·D_mol/D` (puur diffusief, Bessel-reeks).
   - Implementatie: `k_f = max(k_f_convectief, k_f_diffusief)`.
   - Achterwaarts compatibel: bij turbulente snelheden is k_f_convectief altijd groter.

**Nieuwe API-parameters voor `NzingaFlowSolver.__init__()`:**
- `temperature: float | None = None` — watertemperatuur [°C]; None = Rossman 1994-compatibel
- `leakage_fraction: float = 0.0` — fractie debiet dat lekt als massaverlies per leiding

**Lekkage-modellering in `solver.py` — `step()`:**
- Elk segment verliest per tijdstap proportioneel volume: `V(t+dt) = V(t)·exp(−λ·dt)`
- `λ = leakage_fraction / verblijftijd_leiding` (tijdschaalonafhankelijk)
- Concentratie blijft constant (conservatief mengmodel, geen contaminant-instroming)
- Typische waarde voor NL-netwerken: `leakage_fraction=0.10–0.15`

**Achterwaartse compatibiliteit:** volledig behouden. Alle nieuwe parameters hebben
defaults die het oude gedrag reproduceren.


## [1.0.0] — 2025

### Eerste stabiele release

**Kernfunctionaliteit**
- Vectorized LTA-solver via `NzingaFlowSolver` + `EPSRunner`
- Multi-species ondersteuning (n_species > 1)
- Wandreacties via twee-film-model (Rossman 1994)
- CSTR-tankmodel met impliciet Euler
- Volledige geochemie via PhreeqPython (`GeochemSolver`)
- Flow reversal detectie en correctie

**Performance**
- Numba JIT kernels voor advect, decay, exit-detect, node-mixing, merging (~4-5×)
- Gecombineerde bulk+wand decay in één geheugenpass (`combined_decay_multi`)
- Pre-allocated werkbuffers: geen heap-allocaties in hot-path
- CSR pipe-index in `SegmentStore`: lexsort overgeslagen indien gecached
- Vectorized `_handle_exits` via `_node_type`/`_node_out0` routing cache (~734×)
- Numba merging kernels met early-exit (~29×)

**Correctheidsfixes**
- `MassBalanceTracker`: wandverlies-schatting gebruikt nu pre-stap massa
- `_apply_flow_reversal`: Python-loop vervangen door vectorized `np.isin`
- `booster_inject`: volume-eenheid gecorrigeerd (flow × dt → m³)
- `parse_inp`: conversiefactoren MGD/IMGD/AFD/CFS gecorrigeerd
- `SpeciesMap.__post_init__`: `assert` → `ValueError`
- `HydraulicModel._to_cms`: gemarkeerd als deprecated

**Package**
- `py.typed` marker (PEP 561)
- Volledig gedocumenteerde publieke API
- 10 wetenschappelijke validatietests (V1–V10)
- CI via GitHub Actions
