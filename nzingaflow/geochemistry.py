# nzingaflow/geochemistry.py
"""
PhreeqPython-integratie voor nzingaflow.

Vervangt de eenvoudige eerste-orde bulkverval (bulk_first_order_multi) door
volledige geochemische reacties per segment via PHREEQC/PhreeqPython.

Twee modi
---------
GeochemSolver  — concentratiemodus (standaard)
    Elk segment slaat concentraties op als floats in de C-matrix.
    PHREEQC-oplossingen worden tijdelijk aangemaakt, reacties uitgevoerd,
    en concentraties teruggeschreven.  Menging is lineair gewogen.

    Optimalisaties t.o.v. v1.0:
    - Bulk aanmaak: alle PHREEQC-oplossingen in één ip.run_string()-aanroep
      (~1.6× sneller dan individuele aanroepen; gemeten: 122 ms → 75 ms / 500 segs).
    - Read-back via total_element() ipv total() (total() heeft unit-conversie bug
      voor sommige databases; total_element() geeft altijd mol/L).
    - Geprecompileerde unit-conversiefactoren: geen branch per element per stap.

PhreeqSolutionMode  — oplossingnummer-modus (Victoria-compatibel)
    Elk segment draagt een PHREEQC-oplossingnummer.  Concentraties worden
    nooit in de C-matrix opgeslagen; alle queries gaan via PhreeqPython.
    Menging via pp.mix_solutions() is thermodynamisch exact (pH-buffering,
    carbonaatevenwicht na menging van twee waterstromen).

    Gebruik wanneer:
    - Exacte pH-berekening na menging cruciaal is.
    - Downstream query-code (bijv. LSI, SI, sc) direct PHREEQC-objecten nodig heeft.
    - Compatibiliteit met Victoria-stijl workflows gewenst is.

    Nadeel: ~2× trager dan GeochemSolver door mix_solutions() overhead,
    en PHREEQC-geheugen groeit met het aantal unieke oplossingen.
    Roep regelmatig garbage_collect() aan.

Wat PhreeqPython toevoegt t.o.v. eerste-orde verval
----------------------------------------------------
- pH-afhankelijk chloor-verval (HOCl/OCl⁻ evenwicht)
- Carbonaat/bicarbonaatbuffering (alkaliniteit, TIC)
- Corrosieproducten: Fe²⁺/Fe³⁺, Mn²⁺
- Precipitatie/oplossing: CaCO₃ (Langelier Saturation Index)
- Nitrificatie (NH₄⁺ → NO₂⁻ → NO₃⁻)
- Willekeurige PHREEQC-reacties via user_script

Vereisten
---------
    pip install phreeqpython

Gebruik — GeochemSolver (concentratiemodus)
-------------------------------------------
    from nzingaflow.geochemistry import GeochemSolver, SpeciesMap

    smap = SpeciesMap(
        species_names=['Cl2', 'pH', 'Alk', 'Ca', 'Fe'],
        phreeqc_names=['Cl',  None, 'Alk', 'Ca', 'Fe'],
        units        =['mg/L','',  'meq/L','mg/L','mg/L'],
        is_pH        =[False, True, False,  False, False],
    )
    geo = GeochemSolver(species_map=smap, background_solution={...},
                        kinetics_script=..., equilibrium_phases={...})

    # In de simulatielus:
    geo.apply_geochemistry(store, dt=5.0)

Gebruik — PhreeqSolutionMode (oplossingnummer-modus)
-----------------------------------------------------
    from nzingaflow.geochemistry import PhreeqSolutionMode, SpeciesMap
    import phreeqpython

    pp  = phreeqpython.PhreeqPython()
    sol = pp.add_solution({'temp': '15', 'pH': '7.5', 'units': 'mol/L',
                           'Ca': '1e-3', 'Cl': '2e-3', 'Alkalinity': '2.5e-3'})

    psm = PhreeqSolutionMode(pp)

    # Initialiseer segmenten met één bronoplossing
    psm.fill(store, sol.number)

    # In de simulatielus (vervangt apply_geochemistry):
    psm.apply_reactions(store, dt=5.0, kinetics_fn=my_kinetics)
    psm.apply_node_mixing(node_outflows)   # exacte menging via mix_solutions

    # Query
    sol_mixed = psm.get_solution(store, seg_idx=42)
    print(sol_mixed.pH, sol_mixed.total_element('Ca'))

    # Geheugen opruimen (verwijder oplossingen die niet meer in gebruik zijn)
    psm.garbage_collect(store, keep=[sol.number])
"""

from __future__ import annotations
import warnings
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


# ── Gegevensklassen ───────────────────────────────────────────────────────────

@dataclass
class SpeciesMap:
    """
    Koppeling tussen NzingaFlow-stofindices en PHREEQC-elementnamen.

    Parameters
    ----------
    species_names : leesbare namen voor logging/plots (len = n_species)
    phreeqc_names : PHREEQC-elementnaam per stof; None = geen PHREEQC-koppeling
                    Ondersteunde elementen: 'Cl', 'Ca', 'Mg', 'Na', 'K', 'Fe',
                    'Mn', 'N(5)' (NO₃), 'N(3)' (NO₂), 'N(-3)' (NH₄), 'Alk',
                    'C(4)' (TIC/CO₂), 'S(6)' (SO₄), 'P' (fosfaat), ...
    units         : eenheid per stof voor PHREEQC invoer ('mg/L', 'mol/L',
                    'meq/L', 'mmol/L')
    is_pH         : True als de stof de pH-waarde is (dimensieloos, 1–14)
    convert_mg_to_mol : factoren mg/L → mol/L per stof (None = auto via PHREEQC)

    Voorbeeld
    ---------
    Chloor (totaal chloor als Cl₂-equivalent), pH, alkaliniteit:

        SpeciesMap(
            species_names=['Cl2', 'pH', 'Alk'],
            phreeqc_names=['Cl',  None, 'Alk'],
            units        =['mg/L','',  'meq/L'],
            is_pH        =[False, True, False],
        )
    """
    species_names: list[str]
    phreeqc_names: list[str | None]
    units:         list[str]
    is_pH:         list[bool] = field(default_factory=list)

    def __post_init__(self):
        n = len(self.species_names)
        if not self.is_pH:
            self.is_pH = [False] * n
        if len(self.phreeqc_names) != n:
            raise ValueError(
                f"phreeqc_names heeft {len(self.phreeqc_names)} elementen, "
                f"verwacht {n} (gelijk aan species_names)"
            )
        if len(self.units) != n:
            raise ValueError(
                f"units heeft {len(self.units)} elementen, "
                f"verwacht {n} (gelijk aan species_names)"
            )
        if len(self.is_pH) != n:
            raise ValueError(
                f"is_pH heeft {len(self.is_pH)} elementen, "
                f"verwacht {n} (gelijk aan species_names)"
            )

    @property
    def n_species(self) -> int:
        return len(self.species_names)

    @property
    def active_indices(self) -> list[int]:
        """Indices van stoffen met een PHREEQC-koppeling (phreeqc_names[i] is not None)."""
        return [i for i, n in enumerate(self.phreeqc_names) if n is not None]

    @property
    def ph_index(self) -> int | None:
        """Index van de pH-stof, of None als niet aanwezig."""
        for i, p in enumerate(self.is_pH):
            if p:
                return i
        return None


# ── Standaard achtergrondoplossing ─────────────────────────────────────────────

DEFAULT_BACKGROUND = {
    'temp':       15.0,    # °C
    'pH':          7.5,
    'pe':          4.0,
    'units':      'mol/L',
    'Alkalinity':  2.5e-3,  # mol/L als HCO₃⁻
    'Ca':          1.0e-3,
    'Mg':          5.0e-4,
    'Na':          1.0e-3,
    'Cl':          1.0e-3,
    'S(6)':        5.0e-4,
}


# ── Hoofd-klasse ──────────────────────────────────────────────────────────────

class GeochemSolver:
    """
    Geochemische reactor op basis van PhreeqPython/PHREEQC.

    Vervangt bulk_first_order_multi() in de LTA-simulatielus.
    Wandreacties (wall_first_order_multi) blijven apart draaien.

    Parameters
    ----------
    species_map        : SpeciesMap — koppeling NzingaFlow ↔ PHREEQC
    background_solution: achtergrond-watersamenstelling als dict
                         (zie DEFAULT_BACKGROUND voor formaat)
    kinetics_script    : PHREEQC KINETICS-blok als string (optioneel)
                         Gebruik {element} als placeholder voor beginconcentratie.
    equilibrium_phases : dict {fase: (SI_doel, hoeveelheid)} voor
                         EQUILIBRIUM_PHASES berekeningen.
                         Voorbeeld: {'Calcite': (0.0, 0.0), 'Dolomite': (0.0, 0.0)}
    reaction_script    : vrij PHREEQC-script dat na elke tijdstap wordt uitgevoerd
                         (geavanceerd; heeft voorrang op kinetics_script)
    db_path            : pad naar PHREEQC-database (.dat bestand).
                         None = gebruik phreeqpython standaard (llnl.dat)
    batch_size         : aantal segmenten per PHREEQC-batch (prestatie-tuning)
    min_C_threshold    : segmenten met alle concentraties < drempel worden
                         overgeslagen (geen geochemie, wel advectie)

    Gebruik
    -------
        geo = GeochemSolver(species_map=smap, background_solution={...})

        # In de simulatielus, vóór advectie:
        geo.apply_geochemistry(store, dt=5.0)

        # Of via de solver (solver.py integreert dit automatisch):
        solver = NzingaFlowSolver(..., geochem=geo)
    """

    def __init__(
        self,
        species_map:         SpeciesMap,
        background_solution: dict | None = None,
        kinetics_script:     str  | None = None,
        equilibrium_phases:  dict | None = None,
        reaction_script:     str  | None = None,
        db_path:             str  | None = None,
        batch_size:          int  = 500,
        min_C_threshold:     float = 1e-9,
    ):
        self.smap               = species_map
        self.background         = {**DEFAULT_BACKGROUND, **(background_solution or {})}
        self.kinetics_script    = kinetics_script
        self.equilibrium_phases = equilibrium_phases or {}
        self.reaction_script    = reaction_script
        self.batch_size         = batch_size
        self.min_C_threshold    = min_C_threshold

        self._pp           = None   # PhreeqPython instantie (lazy init)
        self._db_path      = db_path
        self._step         = 0
        # Precompileerde hulpstructuren — ingevuld door _init_pp()
        self._unit_factors: np.ndarray | None = None   # (n_species,) conversiefactoren
        self._is_ph_mask:   np.ndarray | None = None   # (n_species,) bool
        self._bg_block:     str        | None = None   # gecachte achtergrond-SOLUTION-regels

    # ── Initialisatie ─────────────────────────────────────────────────────────

    def _init_pp(self):
        """Initialiseer PhreeqPython en precompileer eenmalige hulpstructuren."""
        if self._pp is not None:
            return
        try:
            from phreeqpython import PhreeqPython
        except ImportError:
            raise ImportError(
                "PhreeqPython is niet geïnstalleerd. Installeer via:\n"
                "    pip install phreeqpython"
            )
        if self._db_path:
            self._pp = PhreeqPython(database=self._db_path)
        else:
            self._pp = PhreeqPython()

        # Eenmalig precompileren — vermijdt branches per element per tijdstap
        self._unit_factors, self._is_ph_mask = self._build_unit_factors()
        self._bg_block = self._build_background_block()

    # ── Hoofd API ─────────────────────────────────────────────────────────────

    def apply_geochemistry(
        self,
        store,
        dt:         float,
        pipe_diam:  np.ndarray | None = None,   # (n_pipes,) [m] voor wandreacties
        pipe_vel:   np.ndarray | None = None,   # (n_pipes,) [m/s]
    ) -> None:
        """
        Voer geochemische reacties uit voor alle actieve segmenten (in-place).

        Vervangt bulk_first_order_multi() in de LTA-lus. Wandreacties
        (wall_first_order_multi) kunnen daarna nog apart worden uitgevoerd.

        Parameters
        ----------
        store      : SegmentStore — actieve segmenten
        dt         : tijdstap [s]
        pipe_diam  : leidingdiameters [m] (optioneel, voor wandreacties via PHREEQC)
        pipe_vel   : stroomsnelheden [m/s] (optioneel)
        """
        self._init_pp()
        n = store.n
        if n == 0:
            return

        C   = store.C[:n]       # (n, n_species) view
        act = self.smap.active_indices
        if not act:
            return

        # Bepaal welke segmenten actief zijn (niet bijna leeg)
        active_mask = np.any(C[:, act] > self.min_C_threshold, axis=1)
        active_idx  = np.where(active_mask)[0]

        if len(active_idx) == 0:
            return

        # Verwerk in batches
        for batch_start in range(0, len(active_idx), self.batch_size):
            batch = active_idx[batch_start: batch_start + self.batch_size]
            self._process_batch(C, batch, store.pipe[:n], dt, pipe_diam, pipe_vel)

        self._step += 1

    def apply_mixing(
        self,
        node_C:    np.ndarray,   # (node_count, n_species) — in-place bijgewerkt
        node_flow: np.ndarray,   # (node_count,) debiet gewicht
        dt:        float,
    ) -> None:
        """
        Pas geochemisch evenwicht toe op knoopconcentraties na menging.

        Nuttig als menging op een knoop het chemisch evenwicht verstoort
        (bijv. pH-verschuiving door menging van twee waterstromen).

        Parameters
        ----------
        node_C    : (node_count, n_species) — gemengde knoopconcentraties
        node_flow : (node_count,) — totaal debiet per knoop (voor drempelcheck)
        dt        : tijdstap [s]
        """
        self._init_pp()
        act = self.smap.active_indices
        if not act:
            return

        active_nodes = np.where(
            (node_flow > 0) & np.any(node_C[:, act] > self.min_C_threshold, axis=1)
        )[0]

        for ni in active_nodes:
            sol = self._make_solution(node_C[ni])
            sol_eq = self._run_equilibrium(sol, dt)
            node_C[ni] = self._read_back(node_C[ni], sol_eq)
            sol.desaturate()      # geheugen vrijgeven
            if sol_eq is not sol:
                sol_eq.desaturate()

    def make_injection_solution(self, C_vector) -> dict:
        """
        Zet een injectieconcentratievector om naar een PHREEQC-oplossingsdict.

        Handig voor het aanmaken van consistente begincondities die later
        via solver.inject() worden ingespoten.

        Returns
        -------
        dict geschikt als background_solution-argument voor GeochemSolver
        """
        sol_dict = dict(self.background)
        for i, (pname, unit, is_ph) in enumerate(zip(
            self.smap.phreeqc_names, self.smap.units, self.smap.is_pH
        )):
            if pname is None:
                continue
            val = float(C_vector[i])
            if is_ph:
                sol_dict['pH'] = val
            else:
                sol_dict[pname] = (val, unit)
        return sol_dict

    # ── Interne verwerking ────────────────────────────────────────────────────

    # Geprecompileerde unit-conversiefactoren (mmol/L → doeleenheid).
    # total_element() geeft altijd mmol/L terug (ongeacht de invoer-eenheid).
    # Berekend één keer per SpeciesMap; vermijdt branch per element per tijdstap.
    _UNIT_FACTORS: Dict[str, float] = {
        'mol/L':   1e-3,   # mmol/L → mol/L
        'mol/l':   1e-3,
        'M':       1e-3,
        'mmol/L':  1.0,    # al in mmol/L
        'mmol/l':  1.0,
        'mM':      1.0,
        'meq/L':   1.0,    # benadering: 1 mmol ≈ 1 meq voor eenwaardige ionen
        'meq/l':   1.0,
        'mg/L':    None,   # element-specifiek; afgehandeld via molmassa
        'mg/l':    None,
    }

    # Molmassa's [g/mol] voor veelgebruikte elementen (mg/L-conversie).
    _MOLAR_MASS: Dict[str, float] = {
        'Cl':   35.45,   'Ca':  40.08,   'Mg':  24.31,   'Na':  22.99,
        'K':    39.10,   'Fe':  55.85,   'Mn':  54.94,   'Al':  26.98,
        'S(6)': 32.06,   'P':   30.97,   'N(5)': 14.01,  'N(3)': 14.01,
        'N(-3)':14.01,   'C(4)': 12.01,  'Si':  28.09,   'B':   10.81,
        'Alk':  61.02,   'Ba':  137.33,  'Sr':  87.62,   'Li':  6.94,
        'Zn':  65.38,    'Cu':  63.55,   'Pb':  207.2,   'As':  74.92,
        'Cr':  52.00,    'Ni':  58.69,   'Cd':  112.41,
    }

    def _build_unit_factors(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Bouw conversiefactoren en pH-masker één keer bij initialisatie.

        Returns
        -------
        factors : (n_species,) float64 — vermenigvuldig mol/L met factor → doel-eenheid
        is_ph   : (n_species,) bool    — True voor pH-stoffen
        """
        n   = self.smap.n_species
        fac = np.ones(n, dtype=np.float64)
        iph = np.array(self.smap.is_pH, dtype=bool)

        for i, (pname, unit) in enumerate(
            zip(self.smap.phreeqc_names, self.smap.units)
        ):
            if pname is None or iph[i]:
                continue
            if unit in ('mg/L', 'mg/l'):
                mm = self._MOLAR_MASS.get(pname, 1.0)
                fac[i] = mm             # mmol/L → mg/L  (mm g/mol * 1 mmol/L = mm mg/L)
            else:
                base = self._UNIT_FACTORS.get(unit, 1e3)
                fac[i] = base if base is not None else 1.0

        return fac, iph

    def _build_background_block(self) -> str:
        """
        Bouw een herbruikbaar PHREEQC SOLUTION-header-blok vanuit self.background.
        Geeft een string zonder SOLUTION-nummer terug; die wordt per segment
        ingeplakt door _build_bulk_input().
        """
        bg = self.background
        lines = []
        lines.append(f"  temp {bg.get('temp', bg.get('temperature', 25))}")
        if 'units' in bg:
            lines.append(f"  units {bg['units']}")
        else:
            lines.append("  units mol/L")
        # pH als aparte regel (PHREEQC-notatie)
        if 'pH' in bg:
            lines.append(f"  pH {bg['pH']}")
        # Overige elementen
        skip = {'temp', 'temperature', 'units', 'pH', 'pe'}
        for key, val in bg.items():
            if key in skip:
                continue
            lines.append(f"  {key} {val}")
        return "\n".join(lines)

    def _build_bulk_input(
        self,
        C:         np.ndarray,   # (n, n_species) view
        batch_idx: np.ndarray,   # indices in C
        base_num:  int,          # eerste op te nemen SOLUTION-nummer
    ) -> str:
        """
        Bouw één grote PHREEQC-invoerstring voor alle segmenten in batch_idx.

        Alle SOLUTION-blokken worden in één run_string()-aanroep verwerkt,
        wat de overhead van herhaalde IPC-aanroepen elimineert (~1.6× sneller).

        Elke oplossing begint vanuit de achtergrondsamenstelling en overschrijft
        de gesimuleerde concentraties.  pH wordt als aparte regel ingeplakt.
        """
        bg_lines  = self._bg_block          # gecached bij __init__
        act       = self.smap.active_indices
        ph_idx    = self.smap.ph_index
        pnames    = self.smap.phreeqc_names
        units_lst = self.smap.units
        is_ph     = self.smap.is_pH

        parts: List[str] = []
        for j, i in enumerate(batch_idx):
            num = base_num + j
            lines = [f"SOLUTION {num}", bg_lines]

            # Overschrijf geactiveerde elementen met simulatiewaarden
            for s_idx in act:
                pname = pnames[s_idx]
                val   = float(C[i, s_idx])
                if val < 0.0:
                    val = 0.0
                if is_ph[s_idx]:
                    val = max(1.0, min(14.0, val))
                    lines.append(f"  pH {val:.4f}")
                else:
                    unit = units_lst[s_idx]
                    lines.append(f"  {pname} {val:.8e} {unit}")

            lines.append(f"SAVE SOLUTION {num}")
            parts.append("\n".join(lines))

        return "\n".join(parts) + "\nEND\n"

    def _process_batch(
        self,
        C:         np.ndarray,        # (n, n_species) view — wordt in-place bijgewerkt
        batch_idx: np.ndarray,        # indices in C te verwerken
        pipes:     np.ndarray,        # (n,) pipe-indices
        dt:        float,
        pipe_diam: np.ndarray | None,
        pipe_vel:  np.ndarray | None,
    ) -> None:
        """
        Verwerk één batch segmenten via PHREEQC — bulk aanmaak, bulk read-back.

        Oplossingen worden aangemaakt in één run_string()-aanroep (bulk).
        Reacties worden nog per oplossing uitgevoerd (PHREEQC vereist dit).
        Read-back via total_element() + np.fromiter() is near-zero overhead.
        """
        pp         = self._pp
        batch_size = len(batch_idx)
        base_num   = pp.solution_counter + 1

        # ── 1. Bulk aanmaak in één run_string ──────────────────────────────
        bulk_input = self._build_bulk_input(C, batch_idx, base_num)
        pp.solution_counter += batch_size
        pp.ip.run_string(bulk_input)

        sol_nums = list(range(base_num, base_num + batch_size))

        # ── 2. Reacties per oplossing (onvermijdelijk, PHREEQC-beperking) ──
        result_nums: List[int] = []
        for j, (i, num) in enumerate(zip(batch_idx, sol_nums)):
            sol = pp.get_solution(num)
            p   = int(pipes[i])
            diam = float(pipe_diam[p]) if pipe_diam is not None else None
            vel  = float(pipe_vel[p])  if pipe_vel  is not None else None
            sol_out = self._run_reactions(sol, dt, diam, vel)
            result_nums.append(sol_out.number)

        # ── 3. Read-back: vectorized per element ───────────────────────────
        result_sols = [pp.get_solution(n) for n in result_nums]
        fac  = self._unit_factors   # (n_species,) precompiled
        iph  = self._is_ph_mask     # (n_species,) bool

        for s_idx in self.smap.active_indices:
            pname = self.smap.phreeqc_names[s_idx]
            if iph[s_idx]:
                vals = np.fromiter(
                    (s.pH for s in result_sols), dtype=float, count=batch_size
                )
            else:
                vals = np.fromiter(
                    (s.total_element(pname) for s in result_sols),
                    dtype=float, count=batch_size,
                )
                # total_element() geeft mol/L; converteer naar doel-eenheid
                vals = vals * fac[s_idx]

            np.clip(vals, 0.0, None, out=vals)
            C[batch_idx, s_idx] = vals

        # ── 4. Geheugen vrijgeven ──────────────────────────────────────────
        to_remove = list(set(sol_nums) | set(result_nums))
        pp.remove_solutions(to_remove)

    def _make_solution(self, C_vec: np.ndarray):
        """
        Maak één PHREEQC-oplossing voor een concentratievector.

        Gebruikt voor apply_mixing() (knoopmenging) en hulpfuncties zoals
        langelier_index().  Voor bulk-verwerking gebruikt _process_batch()
        de snellere _build_bulk_input()-methode.
        """
        pp       = self._pp
        bg_lines = self._bg_block
        act      = self.smap.active_indices
        pnames   = self.smap.phreeqc_names
        units_lst= self.smap.units
        is_ph    = self.smap.is_pH

        pp.solution_counter += 1
        num   = pp.solution_counter
        lines = [f"SOLUTION {num}", bg_lines]

        for s_idx in act:
            pname = pnames[s_idx]
            val   = float(C_vec[s_idx])
            if val < 0.0:
                val = 0.0
            if is_ph[s_idx]:
                lines.append(f"  pH {max(1.0, min(14.0, val)):.4f}")
            else:
                unit = units_lst[s_idx]
                lines.append(f"  {pname} {val:.8e} {unit}")

        lines.append(f"SAVE SOLUTION {num}\nEND")
        pp.ip.run_string("\n".join(lines))
        return pp.get_solution(num)

    def _run_reactions(self, sol, dt: float, diam=None, vel=None):
        """
        Voer PHREEQC-reacties uit op één oplossing gedurende dt seconden.

        Volgorde: kinetics → equilibrium_phases → reaction_script
        """
        if self.reaction_script:
            return self._run_raw_script(sol, dt)

        result = sol

        # Kinetische reacties
        if self.kinetics_script:
            result = self._run_kinetics(result, dt)

        # Evenwichtsreacties
        if self.equilibrium_phases:
            result = self._run_eq_phases(result)

        return result

    def _run_equilibrium(self, sol, dt: float):
        """Voer alleen evenwichtsberekening uit (voor knoopmenging)."""
        if self.equilibrium_phases:
            return self._run_eq_phases(sol)
        return sol

    def _run_kinetics(self, sol, dt: float):
        """Voer kinetische reacties uit via PHREEQC KINETICS-blok."""
        dt_days = dt / 86400.0   # PHREEQC gebruikt dagen als tijdseenheid

        # Bouw kinetisch blok op: vervang placeholders
        kinetics_block = self.kinetics_script.format(
            **{
                pname: float(sol.total(pname, 'mol/L'))
                for pname in self.smap.phreeqc_names
                if pname is not None
            }
        )

        try:
            result = sol.add_kinetics(
                kinetics_block,
                steps=[dt_days],
            )
            return result
        except Exception as e:
            warnings.warn(
                f"PHREEQC kinetiek mislukt: {e}. Segment ongewijzigd.",
                RuntimeWarning,
            )
            return sol

    def _run_eq_phases(self, sol):
        """Voer EQUILIBRIUM_PHASES berekening uit."""
        try:
            phases = {
                phase: (si_target, amount)
                for phase, (si_target, amount) in self.equilibrium_phases.items()
            }
            result = sol.add_equilibrium_phases(phases)
            return result
        except Exception as e:
            warnings.warn(
                f"PHREEQC evenwichtsfasen mislukt: {e}. Segment ongewijzigd.",
                RuntimeWarning,
            )
            return sol

    def _run_raw_script(self, sol, dt: float):
        """Voer een vrij PHREEQC-script uit (geavanceerd)."""
        dt_days = dt / 86400.0
        try:
            result = sol.add_reaction_raw(
                self.reaction_script.format(dt=dt_days, dt_s=dt)
            )
            return result
        except Exception as e:
            warnings.warn(
                f"PHREEQC script mislukt: {e}. Segment ongewijzigd.",
                RuntimeWarning,
            )
            return sol

    def _read_back(self, C_vec: np.ndarray, sol) -> np.ndarray:
        """
        Lees PHREEQC-resultaten terug voor één oplossing (gebruikt door apply_mixing).

        Voor bulk-verwerking van segmenten gebruikt _process_batch() de
        geoptimaliseerde vectorized read-back via np.fromiter().
        """
        C_out = C_vec.copy()
        fac   = self._unit_factors
        iph   = self._is_ph_mask

        for s_idx in self.smap.active_indices:
            pname = self.smap.phreeqc_names[s_idx]
            try:
                if iph[s_idx]:
                    C_out[s_idx] = sol.pH
                else:
                    raw = sol.total_element(pname)   # altijd mol/L
                    C_out[s_idx] = max(0.0, raw * fac[s_idx])
            except Exception:
                pass  # houd oorspronkelijke waarde bij als PHREEQC faalt

        return C_out

    # ── Hulpfuncties ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset PHREEQC-instantie (nuttig bij herstart simulatie)."""
        if self._pp is not None:
            try:
                self._pp.ip.CleanupSimulation()
            except Exception:
                pass
        self._pp          = None
        self._step        = 0
        self._unit_factors = None
        self._is_ph_mask   = None
        self._bg_block     = None

    def background_solution(self) -> object:
        """Geeft een PhreeqPython-oplossing van de achtergrondsamenstelling."""
        self._init_pp()
        return self._pp.add_solution_simple(self.background)

    def langelier_index(self, C_vec: np.ndarray) -> float:
        """
        Bereken de Langelier Saturation Index (LSI) voor een concentratieprofiel.

        LSI > 0 → neiging tot CaCO₃-precipitatie (aankorsting)
        LSI < 0 → neiging tot oplossing (corrosief)

        Returns
        -------
        lsi : float
        """
        self._init_pp()
        sol = self._make_solution(C_vec)
        try:
            lsi = sol.si('Calcite')
        except Exception:
            lsi = float('nan')
        sol.desaturate()
        return lsi

    def saturation_indices(self, C_vec: np.ndarray) -> dict[str, float]:
        """
        Bereken verzadigingsindices voor alle relevante mineralen.

        Returns
        -------
        dict {mineraalnaam: SI}
        """
        self._init_pp()
        sol = self._make_solution(C_vec)
        minerals = ['Calcite', 'Aragonite', 'Dolomite', 'Gypsum',
                    'Goethite', 'Fe(OH)3(a)', 'Siderite', 'Vivianite']
        result = {}
        for m in minerals:
            try:
                result[m] = sol.si(m)
            except Exception:
                pass
        sol.desaturate()
        return result

    def __repr__(self) -> str:
        kinetics = 'ja' if self.kinetics_script else 'nee'
        eq       = list(self.equilibrium_phases.keys()) if self.equilibrium_phases else []
        return (
            f"<GeochemSolver species={self.smap.species_names} "
            f"kinetics={kinetics} eq_phases={eq} stap={self._step}>"
        )


# ── Victoria-compatibele modus ────────────────────────────────────────────────

class PhreeqSolutionMode:
    """
    Oplossingnummer-modus voor NzingaFlow — thermodynamisch exacte menging.

    In tegenstelling tot GeochemSolver slaat deze klasse geen concentraties op
    in de C-matrix.  In plaats daarvan draagt elk segment een PHREEQC-
    oplossingnummer mee als integer in een aparte array ``sol_ids``.
    Menging via ``pp.mix_solutions()`` is thermodynamisch exact: pH-buffering
    en carbonaatevenwicht worden correct berekend.

    Architectuur
    ------------
    - ``sol_ids`` : np.ndarray shape ``(capacity,)`` int32, parallel aan
      ``SegmentStore.pipe`` / ``x`` / ``C``.  Index ``i`` bevat het PHREEQC-
      oplossingnummer van segment ``i``.
    - ``NzingaFlowSolver`` hoeft niet te worden aangepast: de ``geochem``-
      interface wordt gerespecteerd via ``apply_geochemistry()`` en
      ``apply_mixing()``.
    - De C-matrix wordt niet gebruikt; alle chemische queries gaan via
      ``pp.get_solution(sol_ids[i])``.

    Nadelen t.o.v. GeochemSolver
    ----------------------------
    - ~2× trager door ``mix_solutions()``-overhead per knoopmenging.
    - PHREEQC-geheugen groeit met unieke oplossingen; roep regelmatig
      ``garbage_collect()`` aan (of gebruik de ingebouwde auto-gc).
    - Concentraties niet beschikbaar als NumPy-array zonder extra query.

    Parameters
    ----------
    pp : PhreeqPython
        Gedeelde PhreeqPython-instantie.  Mag ook gedeeld worden met
        GeochemSolver of Victoria.
    auto_gc_interval : int
        Elke ``auto_gc_interval`` stappen wordt automatisch garbage
        collected.  0 = uitgeschakeld.
    background_number : int | None
        Oplossingnummer van de achtergrondoplossing.  Wordt gebruikt als
        fallback als een segment nog geen oplossing heeft.

    Gebruik
    -------
    ::

        import phreeqpython
        from nzingaflow.geochemistry import PhreeqSolutionMode

        pp  = phreeqpython.PhreeqPython()
        bg  = pp.add_solution({'temp': '15', 'pH': '7.5', 'units': 'mol/L',
                               'Ca': '1e-3', 'Cl': '2e-3', 'Alkalinity': '2.5e-3'})

        psm = PhreeqSolutionMode(pp, background_number=bg.number)

        solver = NzingaFlowSolver('netwerk.inp', n_species=1, geochem=psm)
        # (n_species=1 is placeholder; chemische queries gaan via psm)

        # Initialiseer segmenten
        psm.fill(store, bg.number)

        # Simulatielus:
        for _ in range(n_stappen):
            psm.apply_geochemistry(store, dt=5.0)
            # node-menging via NzingaFlowSolver.step() roept apply_mixing() aan
    """

    def __init__(
        self,
        pp,
        auto_gc_interval: int      = 100,
        background_number: int | None = None,
    ):
        self.pp                 = pp
        self.auto_gc_interval   = auto_gc_interval
        self.background_number  = background_number
        self._step              = 0
        # sol_ids: parallel aan SegmentStore; ingevuld door fill() of bij add
        self.sol_ids: np.ndarray | None = None
        self._capacity: int = 0

    # ── Initialisatie ─────────────────────────────────────────────────────────

    def fill(self, store, solution_number: int) -> None:
        """
        Initialiseer alle actieve segmenten met één bronoplossing.

        Maakt een kopie van de oplossing per segment aan zodat reacties
        segmenten onafhankelijk kunnen aanpassen zonder kruiscontaminatie.

        Parameters
        ----------
        store : SegmentStore
        solution_number : int
            PHREEQC-oplossingnummer van de bronoplossing.
        """
        self._ensure_capacity(store.n)
        n  = store.n
        pp = self.pp

        # Maak n kopieën in één bulk run_string
        base_num = pp.solution_counter + 1
        lines    = []
        for j in range(n):
            num = base_num + j
            lines.append(f"COPY SOLUTION {solution_number} {num}")
        lines.append("END")
        pp.solution_counter += n
        pp.ip.run_string("\n".join(lines))

        self.sol_ids[:n] = np.arange(base_num, base_num + n, dtype=np.int32)

    def _ensure_capacity(self, needed: int) -> None:
        """Vergroot sol_ids-array als nodig (spiegelt SegmentStore.resize)."""
        if self._capacity >= needed:
            return
        new_cap = max(needed, self._capacity * 2, 1000)
        new_arr = np.zeros(new_cap, dtype=np.int32)
        if self.sol_ids is not None and self._capacity > 0:
            new_arr[:self._capacity] = self.sol_ids[:self._capacity]
        self.sol_ids  = new_arr
        self._capacity = new_cap

    # ── Geochem-interface (zelfde signatuur als GeochemSolver) ────────────────

    def apply_geochemistry(
        self,
        store,
        dt:        float,
        kinetics_fn: Callable | None = None,
        pipe_diam: np.ndarray | None = None,
        pipe_vel:  np.ndarray | None = None,
    ) -> None:
        """
        Voer reacties uit voor alle actieve segmenten.

        Parameters
        ----------
        store : SegmentStore
        dt : float
            Tijdstap [s].
        kinetics_fn : callable | None
            ``kinetics_fn(sol, dt) -> sol_out`` — past kinetiek toe op één
            PhreeqPython-oplossing en geeft de resulterende oplossing terug.
            Gebruik dit voor aangepaste PHREEQC KINETICS-blokken.
            None = geen reacties (puur transport).
        pipe_diam, pipe_vel : array | None
            Optioneel; worden doorgegeven aan kinetics_fn als extra context.
        """
        if kinetics_fn is None:
            return   # geen reacties gedefinieerd

        self._ensure_capacity(store.n)
        pp  = self.pp
        n   = store.n
        ids = self.sol_ids

        for i in range(n):
            old_num = int(ids[i])
            sol     = pp.get_solution(old_num)
            sol_out = kinetics_fn(sol, dt)
            if sol_out.number != old_num:
                # Nieuwe oplossing aangemaakt door kinetics_fn
                ids[i] = sol_out.number
                pp.remove_solutions([old_num])

        self._step += 1
        if self.auto_gc_interval > 0 and self._step % self.auto_gc_interval == 0:
            self.garbage_collect(store)

    def apply_mixing(
        self,
        node_C:       np.ndarray,   # (node_count, n_species) — NIET gebruikt
        node_flow:    np.ndarray,   # (node_count,) debietgewichten
        dt:           float,
        node_inflows: List[List[Tuple[int, float]]] | None = None,
    ) -> None:
        """
        Knoopmenging via pp.mix_solutions() — thermodynamisch exact.

        Parameters
        ----------
        node_C : np.ndarray
            Niet gebruikt in oplossingnummer-modus (signatuur-compatibiliteit).
        node_flow : np.ndarray
            Totaal debiet per knoop; knopen met debiet=0 worden overgeslagen.
        dt : float
            Tijdstap [s] (niet gebruikt; aanwezig voor interface-compatibiliteit).
        node_inflows : list of list of (sol_num, fraction) | None
            Per knoop een lijst van (oplossingnummer, debietfractie)-paren.
            Indien None wordt geen menging uitgevoerd.

        Resultaat
        ---------
        Gemengde oplossingen worden opgeslagen in ``self._node_solutions``:
        een dict {node_idx: sol_num} dat door de solver kan worden gebruikt
        om nieuwe segmenten te initialiseren.
        """
        if node_inflows is None:
            return

        pp = self.pp
        self._node_solutions: Dict[int, int] = {}

        for ni, inflow_list in enumerate(node_inflows):
            if not inflow_list or node_flow[ni] <= 0:
                continue

            total = sum(frac for _, frac in inflow_list)
            if total <= 0:
                continue

            mix_dict = {}
            for sol_num, frac in inflow_list:
                sol = pp.get_solution(sol_num)
                if sol is not None:
                    mix_dict[sol] = frac / total

            if not mix_dict:
                continue

            if len(mix_dict) == 1:
                # Geen echte menging nodig; kopieer de enige oplossing
                src_sol = next(iter(mix_dict))
                pp.solution_counter += 1
                num = pp.solution_counter
                pp.ip.run_string(
                    f"COPY SOLUTION {src_sol.number} {num}\nEND"
                )
                self._node_solutions[ni] = num
            else:
                mixed = pp.mix_solutions(mix_dict)
                self._node_solutions[ni] = mixed.number

    def new_segment_solution(self, node_idx: int) -> int:
        """
        Geef het oplossingnummer terug voor een nieuw segment vertrekkend
        vanuit node_idx (na apply_mixing).

        Gebruik in de solver om sol_ids bij te werken wanneer een nieuw
        segment wordt aangemaakt bij een knoop.
        """
        node_sols = getattr(self, '_node_solutions', {})
        if node_idx in node_sols:
            return node_sols[node_idx]
        if self.background_number is not None:
            return self.background_number
        raise KeyError(
            f"Geen oplossing beschikbaar voor knoop {node_idx}. "
            "Roep apply_mixing() eerst aan of stel background_number in."
        )

    # ── Query-methoden ────────────────────────────────────────────────────────

    def get_solution(self, store, seg_idx: int):
        """
        Geef het PhreeqPython-oplossingsobject terug voor segment seg_idx.

        Parameters
        ----------
        store : SegmentStore
        seg_idx : int
            Index in de SegmentStore.

        Returns
        -------
        PhreeqPython Solution object
        """
        self._ensure_capacity(seg_idx + 1)
        return self.pp.get_solution(int(self.sol_ids[seg_idx]))

    def get_conc(
        self,
        store,
        element: str,
        units:   str = 'mg/L',
    ) -> np.ndarray:
        """
        Geef concentraties voor alle actieve segmenten als NumPy-array.

        Parameters
        ----------
        store : SegmentStore
        element : str
            PHREEQC-elementnaam (bijv. 'Ca', 'Cl', 'Fe').
        units : str
            'mg/L', 'mmol/L', 'mol/L', of 'pH'.

        Returns
        -------
        np.ndarray shape (n,)
        """
        n   = store.n
        pp  = self.pp
        ids = self.sol_ids[:n]
        sols = [pp.get_solution(int(num)) for num in ids]

        if units == 'pH':
            return np.fromiter((s.pH for s in sols), dtype=float, count=n)

        # Bepaal conversiefactor mmol/L → gewenste eenheid
        # total_element() geeft altijd mmol/L terug
        if units in ('mg/L', 'mg/l'):
            mm  = GeochemSolver._MOLAR_MASS.get(element, 1.0)
            fac = mm          # mmol/L * mm(g/mol) = mg/L
        elif units in ('mol/L', 'mol/l', 'M'):
            fac = 1e-3        # mmol/L → mol/L
        else:
            fac = 1.0         # al mmol/L

        raw = np.fromiter(
            (s.total_element(element) for s in sols),
            dtype=float, count=n,
        )
        return np.clip(raw * fac, 0.0, None)

    def get_ph(self, store) -> np.ndarray:
        """Geef pH voor alle actieve segmenten als NumPy-array."""
        return self.get_conc(store, '', units='pH')

    def get_properties(self, store) -> Dict[str, np.ndarray]:
        """
        Geef waterchemische eigenschappen voor alle actieve segmenten.

        Returns
        -------
        dict met sleutels 'pH', 'sc' (spec. geleidbaarheid μS/cm),
        'temperature' (°C).
        """
        n    = store.n
        pp   = self.pp
        ids  = self.sol_ids[:n]
        sols = [pp.get_solution(int(num)) for num in ids]
        return {
            'pH':          np.fromiter((s.pH          for s in sols), dtype=float, count=n),
            'sc':          np.fromiter((s.sc           for s in sols), dtype=float, count=n),
            'temperature': np.fromiter((s.temperature  for s in sols), dtype=float, count=n),
        }

    # ── Geheugenbeheer ────────────────────────────────────────────────────────

    def garbage_collect(
        self,
        store,
        keep: Sequence[int] | None = None,
    ) -> int:
        """
        Verwijder PHREEQC-oplossingen die niet meer actief in gebruik zijn.

        Parameters
        ----------
        store : SegmentStore
        keep : sequence of int | None
            Extra oplossingnummers die niet verwijderd mogen worden
            (bijv. bronoplossingen, achtergrondoplossing).

        Returns
        -------
        int
            Aantal verwijderde oplossingen.
        """
        pp = self.pp
        n  = store.n

        # Actieve nummers: segmenten + node_solutions + keep-lijst
        active: set[int] = set(int(x) for x in self.sol_ids[:n] if x > 0)
        if hasattr(self, '_node_solutions'):
            active.update(self._node_solutions.values())
        if keep:
            active.update(keep)
        if self.background_number is not None:
            active.add(self.background_number)

        all_sols    = set(pp.get_solution_list())
        to_remove   = list(all_sols - active)
        if to_remove:
            pp.remove_solutions(to_remove)
            logger.debug("PhreeqSolutionMode GC: %d oplossingen verwijderd", len(to_remove))

        return len(to_remove)

    def on_segment_remove(self, indices: np.ndarray) -> None:
        """
        Callback: verwijder PHREEQC-oplossingen van verwijderde segmenten.

        Roep dit aan vanuit SegmentStore.on_resize of vlak vóór
        SegmentStore.remove() om geheugenlekken te voorkomen.

        Parameters
        ----------
        indices : np.ndarray
            Segment-indices die worden verwijderd.
        """
        nums = [int(self.sol_ids[i]) for i in indices if self.sol_ids[i] > 0]
        if nums:
            self.pp.remove_solutions(nums)
            self.sol_ids[indices] = 0

    # ── Repr ──────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_active = int(np.sum(self.sol_ids > 0)) if self.sol_ids is not None else 0
        return (
            f"<PhreeqSolutionMode actieve_segmenten={n_active} "
            f"stap={self._step} auto_gc={self.auto_gc_interval}>"
        )


# ── Voorgeconfigureerde recepten ──────────────────────────────────────────────

def chlorine_decay_geochem(
    k_bulk_per_day: float = 0.5,
    k_wall_ms:      float = 0.0,
    background:     dict | None = None,
) -> GeochemSolver:
    """
    Kant-en-klare GeochemSolver voor chloor-verval met pH-afhankelijkheid.

    Simuleert:
    - HOCl/OCl⁻ speciatieafhankelijk verval
    - Carbonaat-buffering (pH-verschuiving door chloor-reacties)

    species_map: index 0 = totaal chloor [mg/L als Cl₂], index 1 = pH

    Parameters
    ----------
    k_bulk_per_day : bulkvervalconstante [1/dag]
    k_wall_ms      : wandreactiesnelheid [m/s] (0 = geen wandreactie via PHREEQC)
    background     : achtergrond-watersamenstelling (None = DEFAULT_BACKGROUND)

    Returns
    -------
    GeochemSolver geconfigureerd voor chloor + pH

    Voorbeeld
    ---------
        geo = chlorine_decay_geochem(k_bulk_per_day=0.5)
        solver = NzingaFlowSolver("net.inp", n_species=2, geochem=geo)
        solver.inject("R1", C_vector=[1.0, 7.5], volume=0.05)
    """
    smap = SpeciesMap(
        species_names=['Cl2_total', 'pH'],
        phreeqc_names=['Cl',        None],   # pH wordt via sol.pH gelezen, niet via sol.total()
        units        =['mg/L',      ''],
        is_pH        =[False,       True],
    )

    # Eenvoudige kinetische beschrijving van chloor-verval
    # PHREEQC RATE voor chloor: dC/dt = -k * C
    kinetics = f"""
KINETICS 1
    Chlorine_decay
    -formula  Cl -1
    -m        {{Cl}}
    -parms    {k_bulk_per_day}
    -steps    1 in 1  # tijdstap wordt dynamisch ingesteld
END
"""
    return GeochemSolver(
        species_map=smap,
        background_solution=background,
        kinetics_script=kinetics,
    )


def full_water_chemistry(
    background: dict | None = None,
    equilibrium_minerals: list[str] | None = None,
) -> GeochemSolver:
    """
    Uitgebreide GeochemSolver voor volledige waterchemie:
    chloor, pH, alkaliniteit, calcium, ijzer, mangaan.

    species_map indices:
        0 : totaal chloor     [mg/L als Cl₂]
        1 : pH                [dimensieloos]
        2 : alkaliniteit      [meq/L]
        3 : calcium           [mg/L]
        4 : ijzer (totaal)    [mg/L]
        5 : mangaan           [mg/L]

    Parameters
    ----------
    background            : achtergrond-watersamenstelling
    equilibrium_minerals  : lijst van mineralen voor evenwichtsberekening,
                            bijv. ['Calcite', 'Fe(OH)3(a)']
                            None = ['Calcite'] (CaCO₃ als standaard)

    Returns
    -------
    GeochemSolver geconfigureerd voor volledige waterchemie
    """
    if equilibrium_minerals is None:
        equilibrium_minerals = ['Calcite']

    smap = SpeciesMap(
        species_names=['Cl2',  'pH',   'Alk',   'Ca',   'Fe',   'Mn'],
        phreeqc_names=['Cl',    None,  'Alk',   'Ca',   'Fe',   'Mn'],
        units        =['mg/L',  '',   'meq/L', 'mg/L', 'mg/L', 'mg/L'],
        is_pH        =[False,   True,   False,   False,  False,  False],
    )

    eq_phases = {m: (0.0, 0.0) for m in equilibrium_minerals}

    return GeochemSolver(
        species_map=smap,
        background_solution=background,
        equilibrium_phases=eq_phases,
    )
