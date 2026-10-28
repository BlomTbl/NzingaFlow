# Changelog

## [Unreleased]

### Review: stille fouten, verouderde test, dode module, documentatie, lint en types, prestaties

- **Gewijzigd: alkaliniteits-fallback geeft niet langer 0.** `_sol_value(sol, 'Alk')` gooit een `RuntimeError` als de speciatie van een oplossing niet beschikbaar is, in plaats van stilzwijgend 0 terug te geven. Ook een fout bij het opzoeken van een soort in de som wordt niet langer weggeslikt. Dit pad wordt alleen gebruikt als de exacte PHREEQC-`ALK` (`_alk_batch`) faalt; het normale pad is ongewijzigd.
- **Gewijzigd: mislukte verzadigingsindices zijn zichtbaar.** `saturation_indices()` geeft `NaN` terug voor een mineraal dat niet berekend kan worden, en logt een waarschuwing, in plaats van de sleutel weg te laten.
- **Gewijzigd: validatietest slaat niet meer stil over.** In `test_validation.py` werd de steady-state-test met `pytest.skip` overgeslagen als er geen segmenten in het steady-state-venster zaten. Die test controleerde dan niets en verscheen alleen als "skipped". Hij faalt nu met een melding.
- **Verwijderd: `tests/test_inzingaflow.py`.** Het was een verouderde kopie van `tests/test_validation.py` (ongeveer 60 tests) met de Sherwood-formule van vóór v1.1.0 (`Sh = 3.65`) en een lokale kopie van die formule in plaats van de code die getest moest worden. De actuele tests staan in `test_validation.py`.
- **Verwijderd: `nzingaflow/inp_sources.py`.** Het vorige item meldde al dat het verwijderd was; het bestand stond nog in het pakket. Het werd nergens geïmporteerd.
- **Toegevoegd: lint-, opmaak- en typecontrole in CI.** Nieuwe workflow `.github/workflows/ci.yml` draait `ruff check`, `ruff format --check` en `mypy` bij elke push en pull request. Daarvoor is geen private epynet en geen secret nodig. De release-build draait dezelfde controles vóór de tests.
- **Gewijzigd: code opgemaakt en lint-schoon.** `ruff format` toegepast op het hele pakket en de testsuite (alleen opmaak; de 151 tests geven hetzelfde resultaat). Ongebruikte imports, ongebruikte variabelen en verouderde typesyntaxis opgelost. `E402`/`F811` zijn toegestaan in `tests/` (pytest-fixtureimports en `importorskip`).
- **Gewijzigd: typeannotaties.** Attributen die als `None` beginnen zijn nu geannoteerd (`_pipe_list`, `_units`, `_k_wall_vol`, `SegmentStore.on_resize`); `nodes` in `parse_inp` is getypt als `dict[str, dict[str, Any]]`; de CSR-cache in `SegmentStore` controleert ook `_csr_ptr`; `MsxSimulation.run` geeft een duidelijke fout als er geen toestand is geladen. `mypy nzingaflow` slaagt.
- **Bekende beperking (mypy):** `geochemistry` en `solver` (arrays en PHREEQC-objecten die pas na de eerste solve bestaan) en `lta`/`merging` (numba) hebben gedocumenteerde uitzonderingen per module en foutcode in `pyproject.toml`. Alle andere modules worden volledig gecontroleerd.
- **Opgelost: dode code en een zwakke test.** Ongebruikte `v_max` in `stability.check_dt` verwijderd. `test_mass_conservation_in_mixing` controleert nu de uitvoer van `node_mixing_multi` (debietgewogen massabalans) in plaats van het resultaat in de test opnieuw te berekenen. `test_series_decay_bolus` controleert nu de aankomsttijd (nooit te vroeg, hooguit één `dt` vertraging per leiding in de testlus).
- **Toegevoegd: optionele extra's `[msx]` en `[pandas]`.** `scipy` (alleen voor de solvers `rk45`/`radau` van `MsxReactionSystem`) en `pandas` (alleen voor `MsxSimulationResult.to_dataframe()`) werden lazy geïmporteerd maar stonden nergens vermeld, waardoor een gewone installatie faalde met een kale `ModuleNotFoundError`. Beide hebben nu een extra, en `[all]` bevat ze.
- **Gewijzigd: duidelijke foutmelding als `scipy` of `pandas` ontbreekt.** De `ImportError` noemt nu de extra die je moet installeren (`pip install nzingaflow[msx]` / `[pandas]`). De solvers `euler`, `rk4` en `ros2` werken nog steeds zonder `scipy`.
- **Toegevoegd: afhankelijkheidsschema van de modules.** `nzingaflow/documentation/module-dependencies.svg` (EN) en `module-afhankelijkheden.svg` (NL), getoond in de README onder *Modulaire opbouw*. Getekend uit de echte imports: de solver importeert zijn zeven hulpmodules lazy, `eps` en `geochemistry` zijn via objecten gekoppeld, en `msx`, `msxlibrary`, `parse_inp` en `locale_guard` staan los. `parse_inp.py` wordt door geen enkele module geïmporteerd en alleen in tests gebruikt.
- **Gewijzigd: prestatiecijfers zijn nu gemeten in plaats van geschat.** De README beweerde `merge_segments()` ~2.200 → ~90 µs en een volledige tijdstap ~280 → ~60–90 µs met Numba; geen van beide was te reproduceren. Gemeten (na de sorteerwijziging hierboven): `merge_segments()` ~285 → ~185 µs (1,5–1,6×) en een volledige tijdstap ~350 → ~235 µs op een keten van 200 leidingen (1,4–1,9×). De drie kleine kernels klopten ongeveer (4–9×). Tabel en kenmerkenregels in de README's (EN en NL), de docstring van `merging.py` (~2600 → ~90 µs) en die van `lta.py` (~28 µs winst van `combined_decay_multi`, nu als factor) gecorrigeerd. De bewering "alle hot-path operaties ~4–5× sneller" is beperkt tot de kernels. De claim over de kernelketen in `lta.py` (~160 → ~30–45 µs) is bevestigd.
- **Toegevoegd: `benchmarks/bench_kernels.py`.** Reproduceerbare benchmark van de kernels en de volledige tijdstap, met en zonder Numba (`--compare`, `--no-numba`, `--json`). Opgenomen in de ruff-controles van CI en de release-build.
- **Gewijzigd: `merge_segments()` is sneller.** De leidingindex (`SegmentStore.build_csr`) wordt nu gesorteerd met een stabiele sortering op één samengestelde sleutel (`segments._sort_by_pipe_x`) in plaats van `np.lexsort`, met een O(n)-exactheidscontrole en een terugval op `np.lexsort` (NaN/inf, zeer grote netwerken). De volgorde is identiek aan `np.lexsort`, dus de resultaten veranderen niet: 192 van 192 merges in echte solver-runs waren bit-identiek, en 30 nieuwe tests dekken willekeurige data, gelijke posities, bijna gesorteerde stores, de terugval en het merge-resultaat (mutatiegecontroleerd: zonder de controle of de stabiele sortering falen ze). Met Numba, op een gesorteerde store: `merge_segments()` 694 → 185 µs (3,8×), alles samenvoegbaar 1.235 → 750 µs, willekeurige volgorde 735 → 490 µs; volledige tijdstap 301 → 233 µs (keten van 200 leidingen) en 447 → 371 µs (10×10-grid). De index wordt na elke merge nog steeds met opzet als ongeldig gemarkeerd: de solver past `pipe` en `x` op vier plekken rechtstreeks aan (stroomomkering, doorstroming, splitsing) zonder `add()`/`remove()`, dus de vlag is niet te vertrouwen. Ik kon geen geval construeren waarin vertrouwen op de vlag een fout resultaat geeft, maar de correctheid zou dan van toevallige invalidatie afhangen. De misleidende docstring van `SegmentStore` ("sort overgeslagen indien geldig") zegt dit nu.
- **Tests:** nieuw `tests/test_geochem_foutafhandeling.py` (3 tests) voor de twee geochemie-wijzigingen, `tests/test_optionele_afhankelijkheden.py` (5 tests) voor de foutmeldingen bij ontbrekende `scipy`/`pandas`, en `tests/test_csr_sortering.py` (30 tests) voor de snellere sortering. De suite draait 186 tests (aan het begin van deze review waren het er 208, vóór het verwijderen van de verouderde dubbele `test_inzingaflow.py`).
- **Documentatie:** de modultabel bevat `network_sources.py`, `parse_inp.py` en `locale_guard.py`; de parametertabel van `NzingaFlowSolver` bevat `auto_reservoir_sources`; gebroken ankers hersteld. De kopieën in `nzingaflow/documentation/` komen weer overeen met de README's in de root.

### Bronnen uit het geladen epynet-netwerk (vervangt `inp_sources.py`)

- **Gewijzigd:** reservoir- en junctionbronnen, hun patronen en de patroontiming worden uit het geladen epynet-netwerk gelezen (`nzingaflow/network_sources.py`), niet meer uit de `.inp`-tekst. epynet leest het bestand met de EPANET-kern, dus patroonstart en -stap volgen de betekenis van EPANET.
- **Verwijderd:** `nzingaflow/inp_sources.py` en `tests/test_inp_sources.py`. Vervangen door `tests/test_network_sources.py` (4 tests).
- Gecontroleerd op FRL-10: 5 actieve reservoirs en 2 junctionbronnen, gelijk aan de tekstparser (sterkten, patronen, patroonstap 300 s, start 0).
- Let op: de opvoerhoogte van een reservoir wordt uit de elevation gelezen (de vaste head in EPANET). `EN_HEAD` is vóór een solve 0.

### pH uit alkaliniteit en koolstof (PHREEQC)

- **Nieuw:** `geochemistry.ph_from_alk_ct(alk, ct, background=None, temp=15.0)` berekent per paar (ALK meq/kgw, CT mmol/kgw) de pH met PHREEQC, met optionele vaste achtergrondionen. Unieke paren worden één keer berekend, één PHREEQC-run per blok.
- Tests: `tests/test_ph_alk_ct.py` (gelijk aan directe PHREEQC, NaN bij ongeldige invoer, dubbele paren).

### PHREEQC-route: vijf fouten hersteld (gecontroleerd tegen directe PHREEQC)

- **Gerepareerd: `C(4)`, `N(5)`, `S(6)` gaven 0 terug.** `total_element()` kent alleen de elementnaam; hierdoor verdween carbonaat/nitraat/sulfaat na elke stap. Terugschrijven gebruikt nu de elementnaam (`_readback_name`). Let op: bij meerdere oxidatietoestanden van één element in één mapping is alleen het elementtotaal terug te lezen.
- **Gerepareerd: standaardachtergrond voegde ionen toe.** `DEFAULT_BACKGROUND` bevat geen Ca, Mg, Na, Cl, SO₄ of Alkalinity meer. Bij een carbonaatmenging verschoof de pH daardoor ~0,3 eenheden (9,06 i.p.v. 9,36, de directe PHREEQC-waarde). Ionen moet je expliciet in `background_solution` zetten.
- **Gerepareerd: overzadigde evenwichtsfasen sloegen nooit neer.** `saturate()` lost alleen op. `_run_eq_phases` gebruikt nu `desaturate()` boven het SI-doel; `amount` bepaalt de `in_phase`-limiet bij oplossen. Gecontroleerd: calciet bij SI 0,74 gaat naar Ca 1,81 mmol/L en pH 7,50, zoals directe PHREEQC.
- **Gerepareerd: kinetiek crashte.** PhreeqPython 1.6.2 heeft geen `add_kinetics()` / `add_reaction_raw()`, en `total(…, 'mol/L')` geeft `None`. Kinetiek loopt nu via `ip.run_string()` (`_run_phreeqc_block`). Placeholders in `kinetics_script`: `{dt_s}`, `{dt_d}` en één per element in mol/L. Het RATES-blok moet eindigen met `SAVE rate * TIME`; zonder die regel hangt PHREEQC, daarom krijg je eerst een `ValueError`. `chlorine_decay_geochem()` geeft nu eerste-orde verval dat exp(−kt) volgt.
- **Gerepareerd: meq/L bij andere elementen dan alkaliniteit.** Dat werd met factor 1 teruggelezen, wat voor tweewaardige ionen fout is. `SpeciesMap` weigert dit nu; gebruik mmol/L of mg/L. `phreeqc_names='Alkalinity'` wordt ook geweigerd (gebruik `'Alk'`).
- **Gerepareerd: eerste oplossingsnummer was 0.** PhreeqPython begint met teller −1, en `USE solution 0` laat PHREEQC hangen. De teller start nu bij 0, zodat de eerste oplossing nummer 1 krijgt.
- **Gedrag gewijzigd:** `DEFAULT_BACKGROUND` bevat geen ionen meer, en het chloorrecept is nu eerste-orde in totaal chloor (pH-afhankelijkheid wordt niet gemodelleerd).
- Tests: `tests/test_geochem_regressie.py` (10 tests, echte PHREEQC).

### EPSRunner: `record_every` en `merge_span`

- **Nieuw:** `EPSRunner.run(record_every=n)` bewaart alleen elke n-de kwaliteitsstap. De resultaatmatrix kost `stappen × knopen × n_species × 8` byte; bij een kleine `qual_dt` was dat het grootste deel van het geheugen (24 uur bij 10 s op 26.893 knopen: piek 2081 MB, nu 334 MB met uurlijkse opslag; bewaarde waarden en rekentijd identiek). `time_axis()` volgt de laatste `run()`. Standaard 1 = ongewijzigd.
- **Nieuw:** `EPSRunner.run(merge_span=...)` wordt doorgegeven aan `solver.step` (standaard 1,0 = ongewijzigd). Eerder was `merge_span=None` alleen te bereiken door `solver.step` te vervangen.
- `merge_rel_tol` en de twee nieuwe parameters staan nu in de `run()`-signatuur in de README, met metingen op één netwerk van `qual_dt` tegenover merge-instellingen.

### Verificatie tegen EPANET 2.3.3: reparaties

- **Hersteld: Pattern Start.** Bronpatronen gebruikten `(t - start)`; EPANET gebruikt `((t + start) // stap) mod lengte`. Fout voor elk netwerk met een Pattern Start ≠ 0. Geverifieerd tegen EPANET.
- **Hersteld: reservoir zonder bron.** Het hield de vastgehouden knoopconcentratie, die verandert als het reservoir zelf water ontvangt (stroomomkering). EPANET houdt de beginkwaliteit. De beginkwaliteit wordt nu bij `set_initial_quality` bewaard en ook voor de reservoirknoop gerapporteerd.
- **Hersteld: kwaliteitsklok.** De klok voor automatische bronpatronen volgt nu `update_hydraulics(simtime)` in plaats van stappen vanaf 0 te tellen.
- **Hersteld: water dreef bij elke stap door PHREEQC weg.** Concentraties gingen per liter naar PHREEQC (dat rekent om met een dichtheid van 1 kg/L min de opgeloste massa) en werden als per kg water teruggelezen, en de alkaliniteit werd opgeteld uit een vaste lijst vrije soorten die ionparen en, bij ijzer en mangaan, hydroxocomplexen miste. Gemeten op zes soorten (Cl2, pH, Alk, Ca, Fe, Mn), gemiddelde relatieve verandering per soort na 60 rondes: Alk 73 %, Ca 1,7 %, Fe 2,0 %, Mn 1,7 %, Cl2 1,4 %. Nu: 0,00000 % voor al deze soorten na 576 rondes (pH 0,0003 %, afronding op vier decimalen).
  - Alle PHREEQC-invoer is per kg water (`mg/L` via onze eigen molmassatabel, in beide richtingen dezelfde; `meq/L` → `meq/kgw`; achtergrond-`units` genormaliseerd naar `mol/kgw`). **Gedragswijziging:** "mg/L" in de soortenvector betekent nu mg per kg water (0,03 tot 0,3 % anders dan per liter).
  - Alkaliniteit komt uit PHREEQC's eigen `ALK` (`_alk_batch`, één extra run per batch) in het batch-, meng- en `get_conc`-pad. `_ALK_SPECIES` (coëfficiënten als in phreeqc.dat, inclusief Fe(III) = -2) blijft als terugvaloptie.
  - Ook sneller: 0,068 s per ronde van 100 segmenten tegen 0,100 s eerder (één machine, 100 segmenten).
  - Gecontroleerd met `phreeqc.dat` en `vitens.dat` (0,00000 % drift). `wateq4f_PWN.dat` kan phreeqpython helemaal niet laden.
- Tests: `test_geochem_alk_roundtrip.py` (echte PHREEQC), regressietests voor omkering, Pattern Start en de klok.

### Junctions met negatieve vraag: automatische CONCEN-bronnen

- **Nieuw:** een `CONCEN`-bron op een junction met negatieve vraag bepaalt de concentratie van het instromende water (sterkte × patroon), zoals in EPANET. Alleen toegepast in stappen met negatieve vraag. Parsing: `read_junction_sources` in `inp_sources.py`.
- Andere bronsoorten op junctions geven een `UserWarning` en worden niet automatisch toegepast.

### Reservoirs: automatische bronnen

- **Nieuw:** `auto_reservoir_sources=True` (standaard). Reservoirs met head > 0 worden automatisch gevoed vanuit `[SOURCES]` (CONCEN met patroon) of houden hun beginkwaliteit. Reservoirs met head 0 worden overgeslagen. De parsing staat in `nzingaflow/inp_sources.py` en vereist geen epynet.
- **Gedragswijziging:** voorheen leverden reservoirs geen water aan het netwerk. Testnetwerken die reservoirs met `inject_schedule` voedden, zetten nu `auto_reservoir_sources=False`.
- `NotImplementedError` voor MASS/SETPOINT/FLOWPACED-bronnen op actieve reservoirs; `UserWarning` als `inject()` een automatisch gevoed reservoir als doel heeft.

### Distributie: privé-pakketten, wheels per platform, release-workflow

- **Installatie** beschrijft nu de privé-opzet: EPYnetDTD eerst uit de privé-repository (`git+ssh`), daarna NzingaFlow uit een GitHub-release (wheel per platform) of uit de broncode. De PyPI-instructies zijn verwijderd.
- **Wheel-tags:** wheels krijgen de tag `py3-none-<platform>` in plaats van `py3-none-any`, omdat ze de MSX-binaries bevatten (`setup.py`, `PlatformBdistWheel`). Daarvoor vereist `build-system` nu `setuptools>=70.1`.
- **Release-workflow** `.github/workflows/release.yml`: draait de tests en bouwt bij elke `v*`-tag een wheel op Linux en Windows, plus één keer een sdist, en maakt een concept-GitHub-release. Vereist het secret `EPYNET_DEPLOY_KEY`.
- De `ImportError`-melding in `hydraulics.py` verwijst naar de privé-installatie.

### Foutafhandeling: geen stille terugvalwaarden, installatiedocumentatie gecorrigeerd

- **`solver._get_tank_volumes()`** geeft een `RuntimeError` (met tanknaam) i.p.v. stil 1000 m³.
- **`hydraulics.get_closed_mask()`** geeft een `RuntimeError` als de afsluiterstatus niet te
  lezen is, i.p.v. de afsluiter als open te behandelen.
- **`parse_inp.load_from_epynet()`** geeft een `RuntimeWarning` (met link-/reservoir-/patroon-id)
  bij elke noodwaarde (bronkwaliteit 1.0, patroon 1.0, diameter/lengte, debiet, mislukte solve).
- **Importcontrole:** ontbrekende EPYnetDTD geeft nu een ImportError met uitleg dat
  `epynet>=2.0` niet op PyPI staat en hoe je het installeert.
- **Docs:** installatiesectie van README/README_NL belooft geen NumPy-only installatie meer.
- **Packaging:** de dubbele `nzingaflow/pyproject.toml` is verwijderd.
- `geochemistry._read_back()`: een mislukte PHREEQC-terugleesactie wordt nu geteld en gemeld via `_record_failure` i.p.v. stil genegeerd.
- Nieuw `tests/test_validatie_numeriek.py`: aankomsttijd, massabalans (enkele leiding, serie), stationaire menging (analytisch), gesloten/open afsluiter, stroomomkering en vergelijking met een EPANET-kwaliteitsberekening. Bevinding: met lege leidingen (zonder `set_initial_quality()`) verdunnen mengknopen met ≥2 inkomende leidingen met water zonder segmenten, waardoor een vaste hoeveelheid massa verloren gaat (~0,07 m³·C in het testnetwerk, onafhankelijk van dt). Met gevulde leidingen blijft massa behouden (<0,1 %). Vastgelegd in `EPSRunner.run` en getest.
- Tests: `tests/test_foutafhandeling.py`; verouderde verwachting in `test_mass_balance_tank`
  gelijkgetrokken met de (correcte) `Q_in`-formule.

### Transport: een segment kan in één tijdstap over meerdere korte doorgaande leidingen gaan

- **`solver._handle_exits()` / nieuw `_cascade_thru()`.** De reistijd die na het verlaten van een leiding over is, wordt nu binnen dezelfde tijdstap doorgegeven over opeenvolgende doorgaande knopen. Eerder stopte een segment na één leiding per stap, wat ongeveer één tijdstap vertraging per korte leiding gaf. Op FRL-10 gaf dat een vertraging van 2 u bij de piek van EC op knoop 164593422; na de wijziging ongeveer 0,6 u (passagetijden 15–20 min achter EPANET). Splitsingen, eindknopen en leidingen die niet volledig worden doorlopen, stoppen zoals voorheen.
- De uitlaatconcentratie (`_pipe_out_C`) van elke volledig doorlopen leiding wordt gelijk gesteld aan de concentratie van het segment, zodat tussenliggende knopen de aankomst zien.
- Testen aangepast aan het fysische gedrag: `test_short_pipe_is_crossed_within_one_step` (vervangt de regel "max één leiding per stap") en `test_zero_length_valve_costs_no_extra_step` (vervangt `test_without_skip_valve_costs_a_step`). Geen segment verlaat een leiding twee keer in dezelfde stap.

### PHREEQC-route: chemie van doorstromende segmenten bijhouden

- **`solver._record_pipe_out_chem()` (nieuw).** Per leiding worden pH, alkaliniteit en koolstof bewaard van het laatste segment dat de leiding volledig heeft doorlopen of aan het eind tot stilstand is gekomen. Dit is de PHREEQC-tegenhanger van `_pipe_out_C`. Aangeroepen in `_cascade_thru` (doorgang en eindplaatsing) en in de splitsingstak.
- Reden: bij een doorgaande knoop (bijvoorbeeld een doelknoop met een uitgaande leiding in een deelnetwerk) gaan segmenten door de knoop heen, en komen ze niet op de inkomende leiding te staan. De pH was dan uit de uitgangen van de leiding niet te lezen.
- Gecontroleerd: FRL-10, doelknoop 164593422, 48 u: deelnetwerk en volledig netwerk geven per uur exact dezelfde pH (verschil 0,0).

### Waterkwaliteit-fixes uit de vergelijking met pulse4.inp (EPANET)

- **Externe instroom bij junctions met negatieve vraag.** Zulke knopen krijgen water met concentratie 0 (of `set_source_quality(node, C)`), gemengd met het aankomende water en afgegeven aan de uitgaande leidingen. Reservoirs zijn uitgesloten.
- **Knoopmenging is continu over de leidingstroom.** Elke inkomende leiding draagt Q·dt bij met de uitlaatconcentratie (laatste aankomst). Dat vervangt het mengen van segmenten die in één stap aankomen, wat in stappen zonder aankomst onechte nullen gaf.
- **Persistente knoopconcentratie.** `step()` geeft de laatste bekende knoopwaarde terug in plaats van 0; `node_arrived` markeert aankomsten in de huidige stap.
- **Beginconditie.** Leidingen worden geïnitialiseerd met segmenten van ongeveer v·dt (maximaal 50 per leiding), met de stroomafwaartse knoop als beginconcentratie.

### Kleppen: korte open afsluiters overslaan, gesloten blokkeren

- **`skip_short_valves=True` (nieuw, alleen met `include_valves=True`).** Een open klep met `L < v·dt` wordt overgeslagen: een segment passeert hem in dezelfde stap, in plaats van een hele tijdstap in een element van lengte nul door te brengen. Ketens van kleppen worden opgelost; de tijd in overgeslagen kleppen (`L/v`) wordt van de meegenomen overshoot afgetrokken. Overslaan gebeurt alleen over doorgaande knopen; kleppen vóór een splitsing, eindknoop of tank blijven normale elementen. `solver.skip_info()` geeft het aantal kleppen, gesloten kleppen en overgeslagen kleppen.
- **`cache_links=True` (nieuwe optie, standaard uit).** `HydraulicModel`/`NzingaFlowSolver` bouwen de lijst met linkobjecten niet meer na elke `solve()`, met identieke resultaten.

### Documentatie

- README's (NL en EN, in de root en in `nzingaflow/documentation/`) bijgewerkt: stap 6 van de rekenstappen beschrijft de reistijd over opeenvolgende doorgaande knopen; signaturen van `NzingaFlowSolver` en `HydraulicModel` compleet (`skip_short_valves`, `cache_links`, `reversal_deadband`); nieuwe sectie over de stroomrichting-deadband, `merge_span`, `set_source_quality()` en het experimentele `set_mixed_pipes()`.


### Bugfix (batch 2) — overshoot bij leidingovergang, tankinstroom, CFL, massabalans, EPS-stappen

- **`solver._handle_exits()` — overshoot weggegooid.** Een segment dat een
  leiding verliet kwam in de volgende leiding op `x = 0`, waardoor de al
  afgelegde afstand voorbij het leidingeinde (`x − L`) verloren ging. Elke
  leidingpassage werd zo naar boven afgerond op een geheel aantal tijdstappen
  (gemiddeld ≈ dt/2 te lang per leiding). De overshoot wordt nu als *tijd*
  meegenomen (`(x − L)/v_oud`) en met de snelheid van de nieuwe leiding
  omgerekend, voor doorgaande knopen en splitsingen. Begrensd net onder de
  lengte van de nieuwe leiding, zodat een segment nooit twee keer in één stap exit.
- **`solver._update_tanks()` — verdunning uit het niets.** `Q_in` kwam uit het
  hydraulische debiet, maar `node_C[tank]` was 0 op elke stap zonder aankomend
  segment: de tank werd verdund met C = 0 terwijl er geen massa binnenkwam. De
  instroom komt nu uit de segmenten die de tank deze stap werkelijk bereiken
  (`Q_in = ΣV/dt`, volumegewogen `C_in`).
- **`stability.recommended_dt()` / `check_dt()` — lengte-0 elementen.**
  Afsluiters (`include_valves=True`) hebben lengte 0 en gaven `dt_CFL = 0`,
  waardoor elke dt als CFL-schending gold. Lengte-0 elementen worden genegeerd.
- **`MassBalanceTracker`** — (a) lekmassa werd nooit geboekt (nieuwe
  teller `leakage` en `record_leakage()`); (b) `booster_inject()` die bestaande
  segmenten overschrijft veranderde de massa zonder boeking (nieuw
  `record_injection_mass()`); (c) verval werd geschat op de massa van de vorige
  stap, miste verval van tussentijds geïnjecteerde massa en telde bulk- en
  wandverlies over de volle massa. De solver meet verval nu exact rond de
  vervalstap en splitst per segment (`k_bulk/(k_bulk+k_wall)`);
  `record_step(..., decay_loss=...)` accepteert dat, de oude schatting blijft
  standaard voor directe aanroepers.
- **`set_initial_quality()`** boekte de beginmassa niet in de tracker: `injected` bleef 0 en `balance_error` meldde 0 (`ok=True`) ongeacht de werkelijke fout. Wordt nu als injectie geboekt (gecontroleerd op EPANET `net1.inp`: fout 1e-16).
- **`EPSRunner`** — `int(duration/qual_dt)` kapte af op float-ruis
  (`0.3/0.1 → 2` stappen). Nieuw `_n_steps()`, ook gebruikt door `time_axis()`.
- **Tankbron (nieuw).** Tankknopen absorberen nu aankomende segmenten
  (routingtype 0, ook met uitgaande leidingen) en geven water af aan hun
  uitgaande leidingen als nieuwe segmenten op `x = 0` (`volume = Q·dt`,
  `C = C_tank`), waardoor `_tank_C` eindelijk het transport aanstuurt. De CSTR
  is massabehoudend met tankmassa als toestand: `V_na = V + V_in − Q_uit·dt`,
  `M_nieuw = (M_oud + V_in·C_in)/(1 + Q_uit·dt/V_na + k·dt)`.
  `set_initial_quality()` initialiseert tanks met de knoopconcentratie;
  `mass_balance()` rapporteert `in_tanks`, tankverval wordt als bulkverval
  geboekt en de EPANET-volumereset bij `update_hydraulics()` als injectie.
  `step()` keert niet meer vroeg terug bij 0 segmenten als er tanks zijn.
  `PhreeqSolutionMode`: afgegeven tankwater krijgt de achtergrondoplossing.
- **`solver._get_tank_volumes()` — eenheden.** EPANET geeft tankvolume in ft³
  bij US-flow-eenheden; dat werd als m³ gebruikt (6980 m³ werd 246 500 op
  `net1.inp`). Nieuw: `units.volume_to_m3()`.
- Tests: `tests/test_regressions_batch2.py` (de meeste falen op de batch-1-code).

### Bugfix — decay cache, leidinglengte (US-eenheden), tank-CSTR, time_axis

- **`solver.step()` — stale decay-cache.** Herbouw-controle gebruikte
  `np.allclose(..., atol=1e-8)`; in 1/s is dat ≈ 0,001/dag, dus een wijziging
  van bijv. `decay_k` `[0.0]` → `[5e-9]` werd stilzwijgend genegeerd. Nu exacte
  vergelijking (`np.array_equal` + shape-check) via `_combined_exp_needs_rebuild()`.
- **`hydraulics.get_topology()` / `parse_inp()` — leidinglengte.** EPANET geeft
  lengte in ft bij US-flow-eenheden (GPM, CFS, MGD, IMGD, AFD); die werd niet
  omgerekend (reistijden 3,28× te lang). Nieuw: `units.length_to_m()`.
- **`lta.tank_step_implicit()` — CSTR-formule.** Noemer bevatte `Q_out` i.p.v.
  `Q_in`; fout zodra Q_in ≠ Q_out (leegstromende tank verdunde, vullende tank
  verdunde niet, C kon boven C_in uitkomen). Correct: `V·dC/dt = Q_in·(C_in−C) − k·V·C`.
  Voor Q_in = Q_out ongewijzigd. Lege tank (V = 0) gaf `NaN`; V wordt nu begrensd.
- **`EPSRunner.time_axis()`** begon bij 0; `results[i]` hoort bij `(i+1)·qual_dt`.
- Tests: `tests/test_regressions_batch1.py`; `test_mass_balance_tank` (in
  `test_inzingaflow.py` en `test_validation.py`) legde de oude formule vast en is aangepast.

### Documentatie

- `README.md`, `README_NL.md` en `nzingaflow/documentation/README*.md` in lijn gebracht met de code: versiebadge (1.3.1); de root-README's bevatten nu de sectie `PhreeqSolutionMode` (stond alleen in `documentation/`); tankmodel, fasenlijst van de LTA (incl. tankafgifte en overshoot-doorgifte), `set_initial_quality()`, de betekenis van `EPSRunner.time_axis()`, `recommended_dt()` met lengte-0-elementen, de rapportsleutels van `MassBalanceTracker` (`in_tanks`, `leakage`, `accounted`) en de FAQ over massabalansfouten; de versiehistorie in de README noemt nu 1.3.1, 1.3.0 en de fixes van batch 1/2 (de 1.3.0-punten stonden nog als "Unreleased"); versieverwijzingen voor de MSX-brug consistent gemaakt (geïntroduceerd in 1.2.0, meegeleverde binaries sinds 1.2.2); ontbrekende kop `### Stabiliteitsfuncties` in de NL-README hersteld (kapotte ToC-anker); relatieve links in `documentation/` gerepareerd (`../LICENSE`, `../CHANGELOG*.md`); beschrijving in `PKG-INFO` opnieuw gegenereerd uit de root-README.
- Docstrings bijgewerkt: `MassBalanceTracker` (gebruiksvoorbeeld had een verouderde `record_step`-signatuur; `report()`-sleutels), `NzingaFlowSolver`-klasse en `step()` (tankbron, overshoot, fase 7b), `units.py` (`length_to_m`, `volume_to_m3`).

## [1.3.1] — 2026-09-17

### Bugfix — `PhreeqSolutionMode` was nooit daadwerkelijk aangesloten op de simulatielus

`nzingaflow/geochemistry.py::PhreeqSolutionMode` (de oplossingnummer-modus
op basis van `pp.mix_solutions()`) bestaat al sinds de introductie, maar
was nooit gekoppeld aan `NzingaFlowSolver`'s automatische steplus buiten
de gedeelde `apply_geochemistry()`/`apply_mixing()`-interface — en die
koppeling was zelf ook stuk. De bewering in de klasse-docstring dat
"`NzingaFlowSolver` niet aangepast hoeft te worden" klopte niet: niets in
`solver.py` verwees ooit naar `PhreeqSolutionMode`-specifieke state.
Concreet:

- `NzingaFlowSolver.step()` riep `self._geochem.apply_mixing(node_C,
  node_flow, dt)` aan zonder het `node_inflows`-argument.
  `PhreeqSolutionMode.apply_mixing()` doet niets (`return`) zodra
  `node_inflows is None` — knoopmenging liep dus stilzwijgend nooit. De
  simulatie leek normaal te draaien; er werd geen fout gegooid.
- Ook mét `node_inflows` bleef `sol_ids` (de array die store-index koppelt
  aan PHREEQC-oplossingnummer) nooit synchroon met mutaties op de
  `SegmentStore`: segmentverwijdering op eindknopen gebeurt via
  swap-with-last, en segmentsplitsing voegt nieuwe rijen toe — geen van
  beide werd op `sol_ids` gespiegeld, waardoor indices na de eerste
  verwijdering of splitsing stilzwijgend naar de oplossing van het
  verkeerde segment zouden wijzen.

**Fixes (`nzingaflow/solver.py`)**
- `step()` herkent een oplossingnummer-gebaseerde geochemie-modus nu via
  duck typing (`hasattr(self._geochem, 'sol_ids')`) en bouwt `node_inflows`
  zelf op met de nieuwe helper `_build_node_inflows()` — dezelfde weging
  als `_node_mixing_kernel` in `lta.py`, maar met PHREEQC-oplossingnummers
  in plaats van concentratievectoren.
- `_handle_exits()` roept nu `geochem.mirror_removal()` aan vóór het
  verwijderen van eindknoop-segmenten, en `geochem.copy_solution()` +
  `on_segment_added()` wanneer een segment op een splitsingsknoop
  opsplitst, zodat `sol_ids` uitgelijnd blijft met de store.
- `inject()`, `inject_pipe()` en `booster_inject()` roepen nu de nieuwe
  `_sync_geochem_new_segment()` aan zodat nieuw toegevoegde segmenten
  tenminste een veilige (achtergrondoplossing) `sol_ids`-waarde krijgen
  in plaats van een verweesde/oude; deze C-vector-gebaseerde
  injectiemethoden proberen niet de geïnjecteerde concentratie om te
  zetten naar een PHREEQC-oplossing.
- Periodieke segment-merging (`merge_segments`, elke `merge_interval`
  stappen) wordt nu overgeslagen zodra een oplossingnummer-gebaseerde
  geochemie-modus actief is: `merge_segments` herschikt `SegmentStore`-
  rijen volledig (sort + reductie) zonder enige `sol_ids`-remap-hook, wat
  de array anders na de eerste merge stilzwijgend zou desynchroniseren.

**Fixes (`nzingaflow/geochemistry.py`)**
- Nieuwe `PhreeqSolutionMode`-methoden: `on_segment_added()`,
  `mirror_removal()`, `copy_solution()` — de synchronisatieprimitieven die
  `solver.py` hierboven nodig heeft.
- Gebruiksvoorbeelden op module- en klasseniveau voor `PhreeqSolutionMode`
  gecorrigeerd: die verwezen naar niet-bestaande methoden
  (`apply_reactions`, `apply_node_mixing`) en/of riepen in hun
  voorbeeld-simulatielus nooit daadwerkelijk `NzingaFlowSolver.step()`
  aan — het voorbeeld letterlijk volgen had dus noch advectie noch
  knoopmenging laten draaien.

**Documentatie:** een `PhreeqSolutionMode`-referentiesectie toegevoegd aan
`nzingaflow/documentation/README.md` / `README_NL.md` (daar voorheen
volledig ongedocumenteerd), met de nieuwe methoden en de bovengenoemde
merge-beperking.

**Nog niet opgelost:** `merge_segments()` heeft helemaal geen
`sol_ids`-remap-hook; segment-merging draait in deze modus daarom
gewoon niet, in plaats van correct te draaien. Een volledige fix zou
`merge_segments()` moeten uitbreiden zodat die optioneel een parallelle
array meeneemt door zijn herschik-/reductiepassen. Er is nog geen
regressietest die `PhreeqSolutionMode` tegen een echt netwerk draait
(vereist `phreeqpython` + een PHREEQC-database in de testomgeving).

## [1.3.0] — 2026-08-17

### Hydraulica-refactor + `units.py`

Interne opbouw van `HydraulicModel` herzien rond `Topology`/`HydraulicState`-dataclasses, met een gecachede topologie/leidinglijst en een read-once hydraulica-snapshot na `solve()`. Publieke API ongewijzigd, op twee toevoegingen na: `close()` en context-manager-ondersteuning (`with HydraulicModel(...) as hm:`) voor expliciete opruiming van de EPANET-sessie (`EN_closeH`).

- De interne hydraulische solver vervangen door een sessie-hergebruikende implementatie (`EN_openH()` blijft open tussen `solve()`-aanroepen; `EN_INITFLOW` voor correct cold-start-gedrag), wat de EPS-performance verbetert.
- Nieuwe module `nzingaflow/units.py`: centraliseert EPANET-eenhedenconversies en -enums (lost AFD/MLD- en diameter/snelheid-US↔SI-conversiebugs op), nu gedeeld door `hydraulics.py` en `parse_inp.py`.
- `parse_inp.py` bijgewerkt om typed node/link-klassen en de gecorrigeerde eenhedenconversies te gebruiken.
- **Bugfix** — `solver.py::_get_tank_volumes()`: gebruikte `n.volume`, dat niet bestaat op EPYnetDTD's `Tank` (heet `tank_volume`, `EN_TANKVOLUME`). De resulterende `AttributeError` werd stil opgevangen door een brede `except Exception`, waardoor elk tankvolume altijd de hardgecodeerde fallback-waarde 1000,0 m³ teruggaf — nooit het echte, door EPANET berekende volume. Geen bestaande test gebruikte een tanknetwerk tegen echte EPANET, dus dit bleef onopgemerkt; een regressietest (`TestTankVolume`) is toegevoegd.
- Nieuwe tests: `tests/gen_grid_network.py`; `tests/test_epynet_networks.py` uitgebreid met sessie-hergebruik/regressie-, tankvolume- en memoisatiecontroles.

## [1.2.2] — 2026-07-28

### MSX native library bridge — meegeleverde binaries + kritieke bugfixes

`msxlibrary.py` (`MsxSimulation`/`MsxNativeLib`) was sinds de introductie in
v1.2.0 in de praktijk niet functioneel: er was geen `libepanetmsx`/`epanetmsx.dll`
op het systeem beschikbaar, en zelfs met de bibliotheek aanwezig faalde de
brug alsnog door meerdere onderliggende bugs. Beide zijn nu opgelost.

**Meegeleverde binaries (`nzingaflow/lib/`)**
- `libepanetmsx.so` + `libepanet2_msx.so` (Linux x86-64) en `epanetmsx.dll` +
  `epanet2_msx.dll` (Windows x86-64), gebouwd vanuit de officiële EPANET-MSX
  2.0-broncode (github.com/USEPA/EPANETMSX). macOS is nog niet meegebouwd
  (zie `nzingaflow/lib/README.txt` voor bouwinstructies).
- `_default_lib_path()` zoekt nu ook in `lib/`, platform-bewust.

**Bugfixes in `msxlibrary.py`**
- `MSXstep` gebruikte `c_long` i.p.v. `c_double` voor de tijdsparameters —
  een ABI-mismatch t.o.v. de MSX 2.0 C-API (op Windows potentieel
  geheugencorrumperend, elders stille foutieve tijdwaarden).
- `_epanet_open()` was een no-op; `ENopen()` werd nooit aangeroepen vóór
  `MSXopen()`, terwijl MSX daarvan afhankelijk is voor de gedeelde
  netwerk-state (zie de officiële CLI-referentie `msxmain.c`). Nu wordt
  het EPANET-netwerk daadwerkelijk geopend via een nieuwe `en_open()`/
  `en_close()`-koppeling.
- Node-/link-tellingen en -namen liepen via `MSXgetcount`/`MSXgetID`, die
  dat objecttype niet ondersteunen (alleen SPECIES/CONSTANT/PARAMETER/
  PATTERN) — gaf overal `MSX fout 515`. Omgezet naar de juiste EPANET-laag
  (`ENgetcount`, `ENgetnodeid`, `ENgetlinkid`, `ENgetnodeindex`,
  `ENgetlinkindex`), met een aparte foutvertaling (`ENgeterror`) omdat
  EN- en MSX-foutcodes een andere nummering hebben.
- **Bibliotheeknaam-botsing met epynet:** de epanet2-bibliotheek waar
  `libepanetmsx` tegen linkt deelde dezelfde naam (en op Linux: dezelfde
  SONAME) als epynet's eigen bundled `epanet2`-bibliotheek. Draaiden beide
  in hetzelfde proces (zoals in NzingaFlow, dat epynet gebruikt), dan
  resolvde de dynamic linker/Windows-loader stilzwijgend naar het verkeerde,
  al-geladen exemplaar. Opgelost met een unieke naam (`libepanet2_msx.so` /
  `epanet2_msx.dll`) voor de companion-bibliotheek.

**Documentatie:** de eerdere verwijzing naar EPANET-MSX **1.1** (waar de
`MSXstep`-bug vandaan kwam) is gecorrigeerd naar **2.0** in zowel de
module-docstring als de README's.

**Tests:** nieuwe `tests/test_msxlibrary.py`, tegen het officiële
arseenoxidatie-voorbeeldnetwerk (USEPA EPANETMSX Examples/), inclusief
regressietests voor de hierboven genoemde bugs.

**Achterwaartse compatibiliteit:** de publieke API van `MsxSimulation`/
`MsxNativeLib` is ongewijzigd; dit is uitsluitend een bugfix- en
bundeling-release.

## [1.2.1] — 2026-06-22

### Afsluiters (valves) in transporttopologie — parameter `include_valves`

**`hydraulics.py` — `HydraulicModel`**
- Nieuwe parameter `include_valves: bool = False` in `HydraulicModel.__init__()`.
  Bij `True` worden alle EPANET-afsluitertypes (PRV, PSV, PBV, FCV, TCV, GPV, PCV)
  samen met de leidingen aan de transporttopologie toegevoegd.
- `_get_links()` bijgewerkt: voegt `list(self.net.valves)` toe als `include_valves=True`.
- `summary()` bijgewerkt: het label voor actieve links toont nu `+valves` als de vlag is ingeschakeld.

**`solver.py` — `NzingaFlowSolver`**
- Nieuwe parameter `include_valves: bool = False` in `NzingaFlowSolver.__init__()`,
  doorgegeven aan `HydraulicModel`.

**Achtergrond**

EPANET lost de hydraulica voor afsluiters altijd correct op; NzingaFlow sloot
afsluiters echter altijd uit de Lagrangian transporttopologie. Dit betekende:
- Geen verblijftijd of verval berekend over afsluiterelementen.
- Afsluiters die de *enige verbinding* vormden tussen twee netwerksegmenten
  zorgden voor een topologische ontkoppeling in NzingaFlow.

Met `include_valves=True` worden afsluiters behandeld als korte leidingelementen;
debiet en stroomsnelheid worden rechtstreeks uit de EPANET-oplossing gehaald.

**Achterwaartse compatibiliteit:** standaard is `False`; bestaand gedrag verandert niet.

```python
# Afsluiter-transport inschakelen
solver = NzingaFlowSolver("netwerk.inp", include_valves=True)

# Of direct via HydraulicModel
hyd = HydraulicModel("netwerk.inp", include_valves=True)
```

### Bugfix — `include_pumps=True` crashte op `diameter`

**`hydraulics.py` — `HydraulicModel.get_topology()`**
- `diam_raw` gebruikt nu `getattr(lnk, 'diameter', 0.0)` in plaats van
  `lnk.diameter`, analoog aan de bestaande length-fallback voor afsluiters.

**Achtergrond**

epynet `Pump`-objecten hebben geen `diameter` static property (EPANET kent
geen diameter voor pompen). Elk netwerk met een echte pomp gaf
`AttributeError: ('Nonexistant Attribute', 'diameter')` zodra
`include_pumps=True` werd gebruikt. Pompen worden nu behandeld als
elementen met oppervlak 0 (analoog aan hoe afsluiters al als lengteloze
elementen worden behandeld), zodat `get_topology()` niet meer crasht.

Regressietests voor zowel het afsluiter-transportpad als deze pump-fix zijn
toegevoegd in `tests/test_epynet_networks.py` (echte EPANET `.inp`-netwerken
geladen via epynet, gemarkeerd met `requires_epynet`).

---

## [1.2.0] — 2026-04-24

### Nieuwe module `nzingaflow/msxlibrary.py` — directe koppeling met de EPANET-MSX native bibliotheek

Biedt drie lagen boven de `epanetmsx.dll` / `libepanetmsx.so` C-API:

**`MsxNativeLib` — ctypes-wrapper (laag 1)**
- Dunne wrapper rond alle `MSX_*` C-functies (EPANET-MSX 1.1, revisie 11/01/10).
- Automatische signatuurbinding via `_bind_signatures()`; foutcodes → `MsxError` bij `strict=True`.
- Automatische bibliotheekdetectie (`_default_lib_path()`) voor Windows, Linux en macOS.
- Volledige dekking: open/close, hydraulica, kwaliteit, stapsgewijze simulatie, bronnen,
  patronen, constanten, parameters, initiële kwaliteiten.

**`MsxNetworkState` / dataklassen (laag 2)**
- `MsxNetworkState` — snapshot na `load()`: stoffen, constanten, bronnen, initiële kwaliteiten, patronen.
- `MsxSpecies` — metadata per stof (index, naam, bulk/wand, eenheden, toleranties).
- `MsxSourceRecord` — brondefinitie per (knoop, stof)-paar.
- `MsxSimulationResult` — tijdreeksen `(T × N × S)` node_quality en `(T × L × S)` link_quality;
  helper-methoden `node_concentrations()`, `link_concentrations()`, `to_dataframe()`.

**`MsxSimulation` — orchestrator (laag 3)**
- Hoog-niveau interface: `load()` → `run()` → `MsxSimulationResult`.
- Context-manager (`with MsxSimulation(...) as sim:`).
- Schrijfhulpers: `update_initial_quality()`, `configure_source()`, `update_constant()`,
  `add_time_pattern()`.
- `hyd_file`-parameter voor hergebruik van eerder opgeslagen hydraulicabestand.

**Gemaksfunctie `run_msx(inp, msx)`**
- Volledige simulatie in één aanroep; geeft `MsxSimulationResult` terug.

**Relatie tot bestaand `msx.py`:**
- `msx.py` (`MsxReactionSystem`) — pure-Python ODE-solver; plugt in via `geochem=` in `NzingaFlowSolver`.
  Geen native bibliotheek nodig; geschikt voor Lagrangian transport met eigen reactie-uitdrukkingen.
- `msxlibrary.py` (`MsxSimulation`) — directe brug naar de officiële EPANET-MSX C-solver;
  leest `.msx`-bestanden ongewijzigd; vereist `libepanetmsx.so` / `epanetmsx.dll`.
  Geschikt wanneer een bestaand `.msx`-bestand gebruikt moet worden of wanneer de MSX-solver
  zelf de tijdintegratie verzorgt.

**Nieuwe exports in `nzingaflow/__init__.py`:**
`MsxNativeLib`, `MsxNetworkState`, `MsxSimulation`, `MsxSimulationResult`,
`MsxSpecies`, `MsxSourceRecord`, `MsxError`, `run_msx`.

---

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
- `wall_mode: str = 'two_film'` — interpretatie van k_wall:
  - `'two_film'` (standaard): EPANET-compatibel serieschakeling `k_eff = k_f·k_w/(k_f+k_w)`
  - `'direct'`: k_wall is al k_eff; geen filmweerstand toegepast

**Lekkage-modellering in `solver.py` — `step()`:**
- Elk segment verliest per tijdstap proportioneel volume: `V(t+dt) = V(t)·exp(−λ·dt)`
- `λ = leakage_fraction / verblijftijd_leiding` (tijdschaalonafhankelijk)
- Concentratie blijft constant (conservatief mengmodel, geen contaminant-instroming)
- Typische waarde voor NL-netwerken: `leakage_fraction=0.10–0.15`

---

### MSX multi-species reactielaag — nieuw module `nzingaflow/msx.py`

Implementeert de vier kernconcepten van EPANET-MSX 2.0:

- **`RATE`** — kinetische ODE: `dC/dt = f(C_bulk, C_wall, params)`
- **`EQUIL`** — evenwichtsconditie: `0 = g(C_bulk, C_wall, params)`
- **`FORMULA`** — afgeleide variabele: `C = h(C_bulk, C_wall, params)`
- **`WALL`-soorten** — wandgebonden stoffen (bijv. biofilm); bewegen niet mee met het water

**`MsxReactionSystem` — nieuwe klasse:**
- Expressies als Python-string of directe callable; strings gecompileerd via ingebouwde parser (geen sympy/eval nodig)
- Vijf numerieke ODE-solvers: `euler`, `rk4`, `ros2` (stijf, geen scipy), `rk45` (scipy), `radau` (scipy, legacy)
- `Av = 4/D [m²/m³]` automatisch beschikbaar in alle pipe-expressies
- Per-leiding parameter-overschrijving via `set_pipe_param(pipe_idx, **kwargs)`
- Volledig compatibel met de `geochem`-interface van `NzingaFlowSolver`

**Ingebouwde expressie-parser** (gebaseerd op EPANET-MSX `mathexpr.c`, Rossman/Shang/Uber — US EPA):
- Tokenizer + postfix-evaluator; geen externe afhankelijkheden voor string-expressies
- Ondersteunt: `+ - * / ^ ()` en functies `abs sgn sqrt exp log log10 sin cos tan cot asin acos atan acot sinh cosh tanh coth step`
- Compile-time validatie: onbekende variabelenamen geven een `ValueError` met duidelijke melding

**ROS2 stijve solver** (gebaseerd op EPANET-MSX `ros2.c`, Verwer et al. 1999 / Rossman — US EPA):
- Rosenbrock 2(1) met adaptieve stapgrootte-aanpassing
- Jacobian via eindige differenties; LU-decompositie via `numpy.linalg.solve`
- Aanbevolen voor stijve systemen (chloramine-kinetiek, arsenaat-adsorptie) — geen scipy vereist
- Vervangt `radau` als standaard voor stijve voorgeconfigureerde modellen

**Kant-en-klare modellen:**
- `chloramine_decay_msx(k_f, k_ox, solver='ros2')` — HOCl + NH3 → NH2Cl (Vikesland 2001); 3 stoffen
- `chlorine_nom_msx(k_bulk, k_wall, solver='rk4')` — Cl2 + NOM bulk/wand; 2 stoffen
- `arsenic_oxidation_msx(Ka, Kb, K1, K2, Smax, solver='ros2')` — AS3→AS5 + wandadsorptie (Zhang 2004); 3 bulk + 1 wandsoort

**Achterwaartse compatibiliteit:** volledig behouden. Alle nieuwe parameters hebben
defaults die het v1.0.0-gedrag reproduceren. `solver='radau'` blijft werken voor bestaande code.


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
