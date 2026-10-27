# Changelog

## [Unreleased]

### Review: silent failures, stale test, dead module, docs, lint and types

- **Changed: alkalinity fallback no longer returns 0.** `_sol_value(sol, 'Alk')` raises a `RuntimeError` when the speciation of a solution is not available, instead of silently returning 0. A failing species lookup inside the sum is no longer swallowed either. This path is only used when the exact PHREEQC `ALK` (`_alk_batch`) fails; the normal path is unchanged.
- **Changed: failing saturation indices are visible.** `saturation_indices()` returns `NaN` for a mineral that cannot be computed and logs a warning, instead of leaving the key out of the result.
- **Changed: validation test no longer skips silently.** In `test_validation.py` the steady-state test used `pytest.skip` when no segments were in the steady-state window, so it checked nothing and only showed as "skipped". It now fails with a message.
- **Removed: `tests/test_inzingaflow.py`.** It was an outdated copy of `tests/test_validation.py` (about 60 tests) with the pre-v1.1.0 Sherwood formula (`Sh = 3.65`) and a local copy of that formula instead of the code under test. The current tests live in `test_validation.py`.
- **Removed: `nzingaflow/inp_sources.py`.** The entry above already said it was removed; the file was still in the package. It was not imported anywhere.
- **Added: lint, format and type checks in CI.** New workflow `.github/workflows/ci.yml` runs `ruff check`, `ruff format --check` and `mypy` on every push and pull request. It needs no private epynet and no secrets. The release build runs the same checks before the tests.
- **Changed: code formatted and lint-clean.** `ruff format` applied to the whole package and test suite (formatting only; the 151 tests give the same result). Unused imports, unused variables and outdated type syntax fixed. `E402`/`F811` are allowed in `tests/` (pytest fixture imports and `importorskip`).
- **Changed: type annotations.** Attributes that start as `None` are now annotated (`_pipe_list`, `_units`, `_k_wall_vol`, `SegmentStore.on_resize`); `nodes` in `parse_inp` is typed as `dict[str, dict[str, Any]]`; the CSR cache in `SegmentStore` also checks `_csr_ptr`; `MsxSimulation.run` raises a clear error when no state is loaded. `mypy nzingaflow` passes.
- **Known limit (mypy):** `geochemistry` and `solver` (arrays and PHREEQC objects that only exist after the first solve) and `lta`/`merging` (numba) have documented, per-module error-code exceptions in `pyproject.toml`. All other modules are fully checked.
- **Fixed: dead code and a weak test.** Removed the unused `v_max` in `stability.check_dt`. `test_mass_conservation_in_mixing` now checks the output of `node_mixing_multi` (flow-weighted mass balance) instead of recomputing the result in the test. `test_series_decay_bolus` now asserts the arrival time (never early, at most one `dt` of delay per pipe in the test loop).
- **Added: optional extras `[msx]` and `[pandas]`.** `scipy` (only for the `rk45`/`radau` solvers of `MsxReactionSystem`) and `pandas` (only for `MsxSimulationResult.to_dataframe()`) were imported lazily but not listed anywhere, so a plain install failed with a bare `ModuleNotFoundError`. Both now have an extra, and `[all]` includes them.
- **Changed: clear error when `scipy` or `pandas` is missing.** The `ImportError` now names the extra to install (`pip install nzingaflow[msx]` / `[pandas]`). The `euler`, `rk4` and `ros2` solvers still work without `scipy`.
- **Added: module dependency diagram.** `nzingaflow/documentation/module-dependencies.svg` (EN) and `module-afhankelijkheden.svg` (NL), shown in the README under *Module Overview*. Drawn from the real imports: the solver imports its seven helper modules lazily, `eps` and `geochemistry` are linked through objects, and `msx`, `msxlibrary`, `parse_inp` and `locale_guard` stand alone. `parse_inp.py` is not imported by any module and is only used by tests.
- **Tests:** new `tests/test_geochem_foutafhandeling.py` (3 tests) for the two geochemistry changes, and `tests/test_optionele_afhankelijkheden.py` (5 tests) for the missing-`scipy`/`pandas` errors. The suite runs 156 tests (it was 208 at the start of this review, before the stale duplicate `test_inzingaflow.py` was removed).
- **Docs:** the module table lists `network_sources.py`, `parse_inp.py` and `locale_guard.py`; the `NzingaFlowSolver` parameter table lists `auto_reservoir_sources`; broken anchors fixed. The copies in `nzingaflow/documentation/` match the root READMEs again.

### Sources from the loaded epynet network (replaces `inp_sources.py`)

- **Changed:** reservoir and junction sources, and their patterns and pattern timing, are read from the loaded epynet network (`nzingaflow/network_sources.py`) instead of parsing the `.inp` text. epynet reads the file with the EPANET kernel, so pattern start and timestep follow EPANET's own interpretation.
- **Removed:** `nzingaflow/inp_sources.py` and `tests/test_inp_sources.py`. Replaced by `tests/test_network_sources.py` (4 tests).
- Checked on FRL-10: 5 active reservoirs and 2 junction sources, identical to the text parser (strengths, patterns, pattern step 300 s, start 0).
- Note: the reservoir head is read from its elevation (the fixed head in EPANET). `EN_HEAD` is 0 before a solve.

### pH from alkalinity and carbon (PHREEQC)

- **New:** `geochemistry.ph_from_alk_ct(alk, ct, background=None, temp=15.0)` computes pH per (ALK meq/kgw, CT mmol/kgw) pair with PHREEQC, with optional fixed background ions. Unique pairs are computed once, one PHREEQC run per chunk.
- Tests: `tests/test_ph_alk_ct.py` (matches direct PHREEQC, NaN for invalid input, duplicate pairs).

### PHREEQC-route: five fixes (verified against direct PHREEQC)

- **Fixed: `C(4)`, `N(5)`, `S(6)` read back as 0.** `total_element()` only accepts the bare element name, so these mappings wiped carbonate/nitrate/sulfate after every step. Read-back now uses the element name (`_readback_name`). Caveat: with several oxidation states of one element in a mapping, only the element total is read back.
- **Fixed: default background added ions.** `DEFAULT_BACKGROUND` no longer contains Ca, Mg, Na, Cl, SO₄ or Alkalinity. With a carbonate mixing node the old default moved pH by ~0.3 units (9.06 instead of 9.36, the direct PHREEQC value). Ions must be given explicitly in `background_solution`.
- **Fixed: supersaturated equilibrium phases never precipitated.** `saturate()` only dissolves. `_run_eq_phases` now uses `desaturate()` above the SI target; the `amount` value sets the `in_phase` limit for dissolution. Checked: calcite at SI 0.74 goes to Ca 1.81 mmol/L, pH 7.50, as in direct PHREEQC.
- **Fixed: kinetics crashed.** PhreeqPython 1.6.2 has no `add_kinetics()` / `add_reaction_raw()`, and `total(…, 'mol/L')` returns `None`. Kinetics now run through `ip.run_string()` (`_run_phreeqc_block`). The `kinetics_script` placeholders are `{dt_s}`, `{dt_d}` and one per element in mol/L. The RATES block must end with a `SAVE rate * TIME` line; without it PHREEQC hangs, so the code raises a `ValueError` first. `chlorine_decay_geochem()` now gives first-order decay that matches exp(−kt).
- **Fixed: meq/L on non-alkalinity elements.** It was read back with factor 1, which is wrong for divalent ions. `SpeciesMap` now rejects it; use mmol/L or mg/L. `phreeqc_names='Alkalinity'` is also rejected (use `'Alk'`).
- **Fixed: first solution number was 0.** PhreeqPython starts its counter at −1, and `USE solution 0` hangs PHREEQC. The counter now starts at 0, so the first solution is number 1.
- **Behaviour change:** `DEFAULT_BACKGROUND` no longer contains ions (see above), and chlorine recipe is now first-order on total Cl (pH dependence is not modelled).
- Tests: `tests/test_geochem_regressie.py` (10 tests, real PHREEQC).

### EPSRunner: `record_every` and `merge_span`

- **New:** `EPSRunner.run(record_every=n)` stores only every n-th quality step. The result matrix costs `steps × nodes × n_species × 8` bytes; at a small `qual_dt` that was most of the memory (24 h at 10 s on 26,893 nodes: peak 2081 MB, now 334 MB with hourly recording; stored values and run time identical). `time_axis()` follows the last `run()`. Default 1 = unchanged.
- **New:** `EPSRunner.run(merge_span=...)` is passed to `solver.step` (default 1.0 = unchanged). Before, `merge_span=None` could only be reached by replacing `solver.step`.
- `merge_rel_tol` and the two new parameters are now in the `run()` signature in the README, with measurements on one network for `qual_dt` against merge settings.

### Verification against EPANET 2.3.3: fixes

- **Fixed: Pattern Start.** Source patterns used `(t - start)`; EPANET uses `((t + start) // step) mod length`. Wrong for any network with a non-zero Pattern Start. Verified against EPANET.
- **Fixed: reservoir without source.** It kept the held node concentration, which changes when the reservoir itself receives water (flow reversal). EPANET keeps the initial quality. The initial quality is now stored at `set_initial_quality` and also reported for the reservoir node.
- **Fixed: quality clock.** The clock for automatic source patterns now follows `update_hydraulics(simtime)` instead of counting steps from 0.
- **Fixed: water drifted on every step through PHREEQC.** Concentrations were written per litre (PHREEQC converts with a density of 1 kg/L minus the dissolved mass) and read back as per kg of water, and alkalinity was summed from a fixed list of free species that missed ion pairs and, with iron and manganese, hydroxo complexes. Measured on six species (Cl2, pH, Alk, Ca, Fe, Mn), average relative change per species after 60 rounds: Alk 73 %, Ca 1.7 %, Fe 2.0 %, Mn 1.7 %, Cl2 1.4 %. Now: 0.00000 % for all of them after 576 rounds (pH 0.0003 %, rounding to four decimals).
  - All PHREEQC input is per kg of water (`mg/L` via our own molar mass table, the same in both directions; `meq/L` → `meq/kgw`; background `units` normalised to `mol/kgw`). **Behaviour change:** "mg/L" in the species vector now means mg per kg of water (0.03 to 0.3 % different from per litre).
  - Alkalinity is read from PHREEQC's own `ALK` (`_alk_batch`, one extra run per batch) in the batch, mixing and `get_conc` paths. `_ALK_SPECIES` (coefficients as in phreeqc.dat, including Fe(III) = -2) remains as a fallback.
  - Also faster: 0.068 s per round of 100 segments against 0.100 s before (one machine, 100 segments).
  - Checked with `phreeqc.dat` and `vitens.dat` (0.00000 % drift). `wateq4f_PWN.dat` cannot be loaded by phreeqpython at all.
- Tests: `test_geochem_alk_roundtrip.py` (real PHREEQC), regression tests for reversal, Pattern Start and the clock.

### Junctions with negative demand: automatic CONCEN sources

- **New:** a `CONCEN` source on a junction with negative demand sets the concentration of the inflowing water (strength × pattern), as in EPANET. Applied only in steps with negative demand. Parsing: `read_junction_sources` in `inp_sources.py`.
- Other source types on junctions give a `UserWarning` and are not applied automatically.

### Reservoirs: automatic sources

- **New:** `auto_reservoir_sources=True` (default). Reservoirs with head > 0 are fed automatically from `[SOURCES]` (CONCEN with pattern) or keep their initial quality. Reservoirs with head 0 are skipped. Parsing lives in `nzingaflow/inp_sources.py` and needs no epynet.
- **Behaviour change:** before, reservoirs delivered no water to the network. Test networks that fed reservoirs with `inject_schedule` now set `auto_reservoir_sources=False`.
- `NotImplementedError` for MASS/SETPOINT/FLOWPACED sources on active reservoirs; `UserWarning` when `inject()` targets an automatically fed reservoir.

### Distribution: private packages, per-platform wheels, release workflow

- **Installation** now describes the private setup: install EPYnetDTD first from its private repository (`git+ssh`), then NzingaFlow from a GitHub release (wheel per platform) or from source. The PyPI instructions are removed.
- **Wheel tags:** wheels are tagged `py3-none-<platform>` instead of `py3-none-any`, because they contain the MSX binaries (`setup.py`, `PlatformBdistWheel`). `build-system` requires `setuptools>=70.1` for this.
- **Release workflow** `.github/workflows/release.yml`: runs the tests and builds a wheel on Linux and Windows, plus an sdist once, on each `v*` tag, and creates a draft GitHub release. Needs the secret `EPYNET_DEPLOY_KEY`.
- `ImportError` message in `hydraulics.py` points to the private install.

### Error handling: no silent fallbacks, installation docs corrected

- **`solver._get_tank_volumes()`** raises `RuntimeError` (naming the tank) instead of silently
  using 1000 m³.
- **`hydraulics.get_closed_mask()`** raises `RuntimeError` when a valve status cannot be read,
  instead of treating the valve as open.
- **`parse_inp.load_from_epynet()`** emits a `RuntimeWarning` (with link/reservoir/pattern id)
  for every emergency value (source quality 1.0, pattern 1.0, diameter/length, flow, failed solve).
- **Import guard:** a missing EPYnetDTD now gives an ImportError explaining that `epynet>=2.0`
  is not on PyPI and how to install it.
- **Docs:** README/README_NL installation section no longer claims a NumPy-only install.
- **Packaging:** removed the stray duplicate `nzingaflow/pyproject.toml`.
- `geochemistry._read_back()`: a PHREEQC read-back failure is now counted/warned via `_record_failure` instead of silently ignored.
- New `tests/test_validatie_numeriek.py`: arrival time, mass balance (single pipe, series), steady-state mixing (analytical), closed/open valve, flow reversal and comparison with an EPANET quality run. Finding: with empty pipes (no `set_initial_quality()`), mixing junctions with ≥2 inflows dilute with water that has no segments, losing a fixed amount of mass (~0.07 m³·C in the test network, independent of dt). With pre-filled pipes mass is conserved (<0.1 %). Documented in `EPSRunner.run` and tested.
- Tests: `tests/test_foutafhandeling.py`; stale expectation in `test_mass_balance_tank` aligned
  with the (correct) `Q_in` formula.

### PHREEQC-route: keep chemistry of flowing-through segments

- **`solver._record_pipe_out_chem()` (new).** For each pipe, the pH, alkalinity and carbon are kept from the last segment that fully passed through the pipe, or that came to rest at its end. This is the PHREEQC counterpart of `_pipe_out_C`. Called in `_cascade_thru` (pass-through and end placement) and in the splitting branch.
- Reason: at a pass-through node (for example a target node with an outgoing pipe in a subnetwork), segments move through the node in the cascade and never end up on the incoming pipe. The pH could then not be read from the pipe outlets.
- Checked: FRL-10, target node 164593422, 48 h: subnetwork and full network give exactly the same pH per hour (difference 0.0).

### Transport: a segment may cross several short through-pipes in one time step

- **`solver._handle_exits()` / new `_cascade_thru()`.** Remaining travel time after
  leaving a pipe is now carried across consecutive through-nodes within the same
  time step, instead of stopping after one pipe. Previously a segment could cross
  only one pipe per step, which added ~one time step of delay per short pipe. On
  FRL-10 (many pipes of a few metres, flow ~0.1 m³/s) this produced a 2 h lag on
  the peak EC at node 164593422 against EPANET; after the change the lag is
  ~0.6 h (passage times 15–20 min behind EPANET). Splits, end nodes and pipes not
  fully traversed still stop as before.
- The uitlaatconcentratie (`_pipe_out_C`) of each pipe fully traversed in the
  cascade is set to the segment's concentration, so intermediate nodes see the
  arrival.
- Tests changed to the physical travel-time behaviour:
  `test_short_pipe_is_crossed_within_one_step` (replaces the one-pipe-per-step
  assertion) and `test_zero_length_valve_costs_no_extra_step` (replaces
  `test_without_skip_valve_costs_a_step`). No segment exits twice in one step; the
  double-exit concern is unchanged.


### Water-quality fixes from the pulse4.inp comparison against EPANET

- **External inflow at junctions with negative demand.** Such junctions now receive
  water with concentration 0 (or `set_source_quality(node, C)`), mixed with the
  arriving water and released into the outflow pipes. Reservoirs are excluded
  (their negative demand is supply, not inflow).
- **Node mixing is continuous over pipe flow.** Each inflowing pipe contributes Q·dt
  with the concentration at its outlet (last arrival). This replaces mixing of the
  segments that arrive in a single step, which produced spurious zeros in steps
  without arrivals. Outflow of nodes with ≥2 inflows or external inflow gets one mixed
  concentration.
- **Persistent node concentration.** `step()` returns the last known node value
  instead of 0; `node_arrived` flags arrivals in the current step.
- **Initial condition.** Pipes are initialised with the initial quality of their
  downstream node (EPANET behaviour) and split into segments of length ≈ v·dt
  (`set_initial_quality(..., dt=...)`).
- **Merging with a position constraint.** `merge_segments(..., max_span=...)`;
  `step()` uses `merge_span·v·dt` (default 1.0).
- **Skipped open valves update their outlet concentration.** Nodes downstream of a
  skipped valve (J9, J10 in pulse4.inp) were stuck on a stale value.
- **Tank mixing models.** A warning is issued for tanks whose [MIXING] model is not
  MIXED (FIFO, LIFO, 2COMP are not implemented; tanks are CSTR).
- **Hydraulic source.** The node demand is read optionally; a hydraulic source without
  `get_node_demand()` still works.

Regression: `tests/test_pulse4_regression.py` (RMSE ≤ 1.5 mg/L per node vs EPANET
2.3.3, scenario: 6 h, hydraulic step 300 s, quality step 1 s). Measured: mean RMSE
0.23 mg/L, maximum 0.6 mg/L (tank T1, MIXED vs FIFO).


### Valves: skip open short valves, block closed ones

- **`skip_short_valves=True` (new, only with `include_valves=True`).** An open valve with
  `L < v·dt` is skipped: a segment passes it within the same step instead of spending a
  whole time step in a zero-length element (which delayed transport by up to one step per
  valve). Chains of valves are resolved, the time spent in skipped valves (`L/v`) is
  subtracted from the carried overshoot. Skipping only happens over through-nodes;
  valves before a split, end node or tank stay normal elements. `solver.skip_info()` reports
  valves / closed / skipped. On FRL-10 (dt = 300 s): 3433 valves, 37 closed, 3278 skipped;
  tracer arrives a median of 25 min earlier within 6 h.
- **`cache_links=True` (new option, default off).** `HydraulicModel`/`NzingaFlowSolver` no longer
  rebuild the link-object list after every `solve()`: identical results (tested bit-for-bit),
  ~0.08 s less per hydraulic step on 27.6k links (FRL-10, 6 h EC run: 1.17× faster).
- **Closed valves never pass water (bug).** A zero-length element satisfied `x >= L` at
  `x = 0` also at zero velocity, so segments went straight through *closed* valves. Exit
  detection now requires `v > 0`; closed valves (EPANET status CLOSED / flow ≈ 0) are never
  skipped and block.
- **Routing counts only outlets with flow.** A closed valve or dead branch no longer turns a
  through-node into a split (no more zero-volume segments in closed branches). If a node has
  only stagnant outlets, behaviour is unchanged.
- **`_on_store_resize()` lost exit buffers (bug).** When the segment store doubled during a
  split loop, `_buf_exit_*` were replaced by empty arrays, so exited mass of that step
  disappeared from the mass balance (0.5 % on FRL-10). Contents are now preserved.

### Bugfix (batch 2) — overshoot at pipe transitions, tank inflow, CFL, mass balance, EPS step count

- **`solver._handle_exits()` — overshoot dropped.** A segment that left a pipe
  was placed at `x = 0` in the next pipe, discarding the distance already
  travelled beyond the pipe end (`x − L`). Every pipe transit was therefore
  rounded up to a whole number of time steps (≈ dt/2 too long on average per
  pipe). The excess is now carried over as *time* (`(x − L)/v_old`) and
  converted with the new pipe's velocity, for through-nodes and splits. Capped
  just below the new pipe's length so a segment never exits twice in one step.
- **`solver._update_tanks()` — dilution from nothing.** `Q_in` came from the
  hydraulic flow, but `node_C[tank]` was 0 on every step without an arriving
  segment, so the tank was diluted with C = 0 while no mass entered. Inflow is
  now taken from the segments that actually reach the tank this step
  (`Q_in = ΣV/dt`, volume-weighted `C_in`).
- **`stability.recommended_dt()` / `check_dt()` — zero-length elements.**
  Valves (`include_valves=True`) have length 0 and gave `dt_CFL = 0`, so every
  dt counted as a CFL violation. Zero-length elements are now ignored.
- **`MassBalanceTracker`** — (a) leakage mass was never booked (new
  `leakage` counter and `record_leakage()`); (b) `booster_inject()` overwriting
  existing segments changed the mass without booking it (new
  `record_injection_mass()`); (c) decay was estimated from the previous step's
  mass, missing decay of mass injected in between, and summed bulk and wall
  losses over the full mass. The solver now measures decay exactly around the
  decay pass and splits it per segment (`k_bulk/(k_bulk+k_wall)`);
  `record_step(..., decay_loss=...)` accepts it, the old estimate remains the
  default for direct callers.
- **`set_initial_quality()`** did not book the initial mass in the tracker: `injected` stayed 0 and `balance_error` reported 0 (`ok=True`) regardless of the real error. Now booked as injection (verified on EPANET `net1.inp`: error 1e-16).
- **`EPSRunner`** — `int(duration/qual_dt)` truncated float noise
  (`0.3/0.1 → 2` steps). New `_n_steps()`, also used by `time_axis()`.
- **Tank source (new).** Tank nodes now absorb arriving segments (routing type
  0, also with outflow pipes) and release water to their outflow pipes as new
  segments at `x = 0` (`volume = Q·dt`, `C = C_tank`), so `_tank_C` finally
  drives the transport. The CSTR is mass-conserving with tank mass as state:
  `V_after = V + V_in − Q_out·dt`, `M_new = (M_old + V_in·C_in)/(1 + Q_out·dt/V_after + k·dt)`.
  `set_initial_quality()` initialises tanks from their node concentration;
  `mass_balance()` reports `in_tanks`, tank decay is booked as bulk decay, and
  the EPANET volume reset at `update_hydraulics()` is booked as an injection.
  `step()` no longer returns early with 0 segments when tanks exist.
  `PhreeqSolutionMode`: released tank water gets the background solution.
- **`solver._get_tank_volumes()` — units.** EPANET returns tank volume in ft³
  for US flow units; it was used as m³ (6980 m³ read as 246 500 on `net1.inp`).
  New `units.volume_to_m3()`.
- Tests: `tests/test_regressions_batch2.py` (most fail on the batch-1 code).

### Bugfix — decay cache, pipe length (US units), tank CSTR, time_axis

- **`solver.step()` — stale decay cache.** The rebuild check used
  `np.allclose(..., atol=1e-8)`; in 1/s that is ≈ 0.001/day, so a change such as
  `decay_k` `[0.0]` → `[5e-9]` was silently ignored. Now an exact comparison
  (`np.array_equal` + shape check) via `_combined_exp_needs_rebuild()`.
- **`hydraulics.get_topology()` / `parse_inp()` — pipe length.** EPANET reports
  length in ft for US flow units (GPM, CFS, MGD, IMGD, AFD); it was not
  converted (travel times 3.28× too long). New: `units.length_to_m()`.
- **`lta.tank_step_implicit()` — CSTR formula.** The denominator used `Q_out`
  instead of `Q_in`; wrong whenever Q_in ≠ Q_out (a draining tank diluted, a
  filling tank did not, C could exceed C_in). Correct form:
  `V·dC/dt = Q_in·(C_in − C) − k·V·C`. Unchanged for Q_in = Q_out. An empty
  tank (V = 0) produced `NaN`; V is now clamped.
- **`EPSRunner.time_axis()`** started at 0; `results[i]` belongs to `(i+1)·qual_dt`.
- Tests: `tests/test_regressions_batch1.py`; `test_mass_balance_tank` (in
  `test_inzingaflow.py` and `test_validation.py`) encoded the old formula and was updated.

### Docs

- README's (EN and NL, in the root and in `nzingaflow/documentation/`) updated: step 6 of the calculation steps describes the travel time over consecutive pass-through nodes; signatures of `NzingaFlowSolver` and `HydraulicModel` complete (`skip_short_valves`, `cache_links`, `reversal_deadband`); new section on the flow-reversal dead band, `merge_span`, `set_source_quality()` and the experimental `set_mixed_pipes()`.
- `README.md`, `README_NL.md` and `nzingaflow/documentation/README*.md` brought in line with the code: version badge (1.3.1); root READMEs now contain the `PhreeqSolutionMode` section (previously only in `documentation/`); tank model, LTA phase list (incl. tank release and overshoot carry-over), `set_initial_quality()`, `EPSRunner.time_axis()` semantics, `recommended_dt()` zero-length handling, `MassBalanceTracker` report keys (`in_tanks`, `leakage`, `accounted`) and the FAQ on mass-balance errors; the in-README changelog now lists 1.3.1, 1.3.0 and the batch-1/2 fixes (the 1.3.0 items were still listed as "Unreleased"); MSX bridge version wording made consistent (introduced 1.2.0, bundled binaries 1.2.2); missing `### Stability Functions` heading restored in the Dutch README (broken ToC anchor); relative links in `documentation/` fixed (`../LICENSE`, `../CHANGELOG*.md`); `PKG-INFO` description regenerated from the root README.
- Docstrings updated: `MassBalanceTracker` (usage example had a stale `record_step` signature; `report()` keys), `NzingaFlowSolver` class and `step()` (tank source, overshoot, phase 7b), `units.py` (`length_to_m`, `volume_to_m3`).

## [1.3.1] — 2026-09-17

### Bugfix — `PhreeqSolutionMode` was never actually wired into the simulation loop

`nzingaflow/geochemistry.py::PhreeqSolutionMode` (the solution-number,
`pp.mix_solutions()`-based geochemistry mode) has existed since it was
introduced, but was never connected to `NzingaFlowSolver`'s automatic
step loop beyond the shared `apply_geochemistry()`/`apply_mixing()`
interface — and that connection was itself broken. The class docstring's
claim that "`NzingaFlowSolver` doesn't need any changes" was false: nothing
in `solver.py` ever referenced `PhreeqSolutionMode`-specific state at all.
Concretely:

- `NzingaFlowSolver.step()` called `self._geochem.apply_mixing(node_C,
  node_flow, dt)` without the `node_inflows` argument.
  `PhreeqSolutionMode.apply_mixing()` returns immediately (a no-op) when
  `node_inflows is None` — so node mixing silently never ran. The
  simulation appeared to execute normally; no exception was raised.
- Even with `node_inflows` supplied, `sol_ids` (the array mapping segment
  store index → PHREEQC solution number) was never kept in sync with
  `SegmentStore` mutations: segment removal at end nodes uses
  swap-with-last, and segment splitting adds new rows — neither was
  mirrored on `sol_ids`, so indices would silently point at the wrong
  segment's solution after the first removal or split.

**Fixes (`nzingaflow/solver.py`)**
- `step()` now detects a solution-number-based geochemistry mode via duck
  typing (`hasattr(self._geochem, 'sol_ids')`) and builds `node_inflows`
  itself with the new `_build_node_inflows()` helper — same weighting as
  `_node_mixing_kernel` in `lta.py`, but carrying PHREEQC solution numbers
  instead of concentration vectors.
- `_handle_exits()` now calls `geochem.mirror_removal()` before removing
  end-node segments, and `geochem.copy_solution()` + `on_segment_added()`
  when a segment splits at a junction, keeping `sol_ids` aligned with the
  store.
- `inject()`, `inject_pipe()`, and `booster_inject()` now call the new
  `_sync_geochem_new_segment()` so newly added segments get at least a
  safe (background-solution) `sol_ids` entry instead of a stale/orphaned
  one; these C-vector-based injection methods don't attempt to translate
  the injected concentration into a PHREEQC solution.
- Periodic segment merging (`merge_segments`, every `merge_interval`
  steps) is now skipped when a solution-number geochem mode is active:
  `merge_segments` fully repacks `SegmentStore` rows (sort + reduce) with
  no `sol_ids` remap hook, which would otherwise silently desynchronise
  the array after the first merge.

**Fixes (`nzingaflow/geochemistry.py`)**
- New `PhreeqSolutionMode` methods: `on_segment_added()`, `mirror_removal()`,
  `copy_solution()` — the synchronisation primitives `solver.py` needs
  (see above).
- Corrected the module-level and class-level usage examples for
  `PhreeqSolutionMode`, which referenced nonexistent methods
  (`apply_reactions`, `apply_node_mixing`) and/or never actually called
  `NzingaFlowSolver.step()` in their sample simulation loop — meaning
  following the documented usage literally would have run neither
  advection nor node mixing.

**Docs:** added a `PhreeqSolutionMode` reference section to
`nzingaflow/documentation/README.md` / `README_NL.md` (previously
undocumented there entirely) covering the new methods and the
merge-skipping limitation above.

**Not yet fixed:** `merge_segments()` has no `sol_ids`-remap hook at all;
segment merging simply doesn't run in this mode rather than running
correctly. A full fix would extend `merge_segments()` to optionally carry
a parallel array through its reordering/reduction passes. No regression
test exercises `PhreeqSolutionMode` against a real network yet (needs
`phreeqpython` + a PHREEQC database in the test environment).

## [1.3.0] — 2026-08-17

### Hydraulics refactor + `units.py`

`HydraulicModel` internals reworked around `Topology`/`HydraulicState` dataclasses, with a cached topology/pipe list and a read-once hydraulic snapshot after `solve()`. Public API unchanged except two additions: `close()` and context-manager support (`with HydraulicModel(...) as hm:`) for explicit EPANET session cleanup (`EN_closeH`).

- Replaced the internal hydraulic solver with a session-reusing implementation (`EN_openH()` stays open across `solve()` calls; `EN_INITFLOW` used for correct cold-start behaviour), improving EPS performance.
- New module `nzingaflow/units.py`: centralises EPANET unit-system conversions and enums (fixes AFD/MLD and diameter/velocity US↔SI conversion bugs), now shared by `hydraulics.py` and `parse_inp.py`.
- `parse_inp.py` updated to use typed node/link classes and the corrected unit conversions.
- **Bugfix** — `solver.py::_get_tank_volumes()`: used `n.volume`, which does not exist on EPYnetDTD's `Tank` (renamed `tank_volume`, `EN_TANKVOLUME`). The resulting `AttributeError` was silently swallowed by a broad `except Exception`, so every tank's current volume always returned the hardcoded 1000.0 m³ fallback — never the real, EPANET-computed volume. No existing test exercised a tank network against real EPANET, so this went undetected; a regression test (`TestTankVolume`) has been added.
- New tests: `tests/gen_grid_network.py`; extended `tests/test_epynet_networks.py` with session-reuse/regression, tank-volume, and memoization checks.

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
