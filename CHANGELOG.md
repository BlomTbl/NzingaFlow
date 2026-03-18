# Changelog

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
