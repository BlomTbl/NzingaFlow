# inzingaflow/geochemistry.py
"""
PhreeqPython-integratie voor inzingaflow.

Vervangt de eenvoudige eerste-orde bulkverval (bulk_first_order_multi) door
volledige geochemische reacties per segment via PHREEQC/PhreeqPython.

Wat PhreeqPython toevoegt t.o.v. eerste-orde verval
----------------------------------------------------
- pH-afhankelijk chloor-verval (HOCl/OCl⁻ evenwicht)
- Carbonaat/bicarbonaatbuffering (alkaliniteit, TIC)
- Corrosieproducten: Fe²⁺/Fe³⁺, Mn²⁺
- Precipitatie/oplossing: CaCO₃ (Langelier Saturation Index)
- Nitrificatie (NH₄⁺ → NO₂⁻ → NO₃⁻)
- Willekeurige PHREEQC-reacties via user_script

Ontwerp
-------
- GeochemSolver houdt één PhreeqPython-instantie bij voor de hele simulatie.
- apply_geochemistry() verwerkt alle actieve segmenten in één aanroep:
    1. Maak per segment een PHREEQC-oplossing aan.
    2. Voer KINETICS/EQUILIBRIUM_PHASES uit (dt-afhankelijk).
    3. Lees concentraties terug → C-matrix bijwerken.
- Vectorisatie: PHREEQC-oplossingen worden in bulk aangemaakt via
  pp.add_solution_simple(); de Python-loop over segmenten is onvermijdelijk
  maar wordt beperkt door segmentmerging (typisch < 5000 segs).
- species_map koppelt InzingaFlow-stofindices aan PHREEQC-elementnamen.

Vereisten
---------
    pip install phreeqpython

Gebruik
-------
    from geochemistry import GeochemSolver, SpeciesMap

    # Definieer welke InzingaFlow-stoffen overeenkomen met PHREEQC-elementen
    smap = SpeciesMap(
        species_names=['Cl2', 'pH', 'Alk', 'Ca', 'Fe'],
        phreeqc_names=['Cl',  None, 'Alk', 'Ca', 'Fe'],   # None = geen PHREEQC-koppeling
        units       =['mg/L','',   'meq/L','mg/L','mg/L'],
        is_pH       =[False, True, False, False, False],
    )

    geo = GeochemSolver(
        species_map=smap,
        background_solution={'temp': 15, 'pH': 7.5, 'Alkalinity': 2.5e-3,
                              'Ca': 1e-3, 'Mg': 5e-4, 'Na': 1e-3, 'Cl': 1e-3},
        kinetics_script=\"\"\"
            KINETICS 1
                Chlorine_decay
                -m  {Cl}
                -parms  0.5   # k_bulk [1/dag]
        \"\"\",
    )

    # In de simulatielus (vervangt bulk_first_order_multi):
    geo.apply_geochemistry(store, dt=5.0)
"""

from __future__ import annotations
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ── Gegevensklassen ───────────────────────────────────────────────────────────

@dataclass
class SpeciesMap:
    """
    Koppeling tussen InzingaFlow-stofindices en PHREEQC-elementnamen.

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
        assert len(self.phreeqc_names) == n, "phreeqc_names moet zelfde lengte hebben als species_names"
        assert len(self.units)         == n, "units moet zelfde lengte hebben als species_names"
        assert len(self.is_pH)         == n, "is_pH moet zelfde lengte hebben als species_names"

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
    species_map        : SpeciesMap — koppeling InzingaFlow ↔ PHREEQC
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
        solver = InzingaFlowSolver(..., geochem=geo)
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

        self._pp        = None   # PhreeqPython instantie (lazy init)
        self._db_path   = db_path
        self._step      = 0

    # ── Initialisatie ─────────────────────────────────────────────────────────

    def _init_pp(self):
        """Initialiseer PhreeqPython (lazy, alleen als echt nodig)."""
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
            self._pp = PhreeqPython()   # gebruikt ingebouwde llnl.dat

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

    def _process_batch(
        self,
        C:         np.ndarray,        # (n, n_species) view — wordt in-place bijgewerkt
        batch_idx: np.ndarray,        # indices in C te verwerken
        pipes:     np.ndarray,        # (n,) pipe-indices
        dt:        float,
        pipe_diam: np.ndarray | None,
        pipe_vel:  np.ndarray | None,
    ) -> None:
        """Verwerk één batch segmenten via PHREEQC."""
        solutions = []

        # Aanmaken van PHREEQC-oplossingen
        for i in batch_idx:
            sol = self._make_solution(C[i])
            solutions.append(sol)

        # Reacties uitvoeren
        result_solutions = []
        for k, (i, sol) in enumerate(zip(batch_idx, solutions)):
            p = int(pipes[i])
            diam = float(pipe_diam[p]) if pipe_diam is not None else None
            vel  = float(pipe_vel[p])  if pipe_vel  is not None else None
            sol_out = self._run_reactions(sol, dt, diam, vel)
            result_solutions.append((i, sol_out))

        # Terugschrijven
        for i, sol_out in result_solutions:
            C[i] = self._read_back(C[i], sol_out)
            # Geheugen vrijgeven
            sol_out.desaturate()

        # Originele oplossingen opruimen
        for sol in solutions:
            try:
                sol.desaturate()
            except Exception:
                pass

    def _make_solution(self, C_vec: np.ndarray):
        """
        Maak een PhreeqPython-oplossing op basis van een concentratievector.

        Begint altijd vanuit de achtergrondoplossing en overschrijft
        de gespecificeerde stoffen met de waarden uit C_vec.
        """
        sol_dict = dict(self.background)

        for i, (pname, unit, is_ph) in enumerate(zip(
            self.smap.phreeqc_names,
            self.smap.units,
            self.smap.is_pH,
        )):
            if pname is None:
                continue
            val = float(C_vec[i])
            if val < 0:
                val = 0.0
            if is_ph:
                sol_dict['pH'] = max(1.0, min(14.0, val))
            else:
                sol_dict[pname] = (val, unit)

        return self._pp.add_solution_simple(sol_dict)

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
        Lees PHREEQC-resultaten terug naar de InzingaFlow-concentratiematrix.

        Eenheden worden omgezet vanuit PHREEQC (mol/L) naar de eenheden
        in SpeciesMap.units.
        """
        C_out = C_vec.copy()

        for i, (pname, unit, is_ph) in enumerate(zip(
            self.smap.phreeqc_names,
            self.smap.units,
            self.smap.is_pH,
        )):
            if pname is None:
                continue
            try:
                if is_ph:
                    C_out[i] = sol.pH
                elif unit in ('mg/L', 'mg/l'):
                    C_out[i] = sol.total(pname, 'mg/L')
                elif unit in ('mol/L', 'mol/l', 'M'):
                    C_out[i] = sol.total(pname, 'mol/L')
                elif unit in ('mmol/L', 'mmol/l', 'mM'):
                    C_out[i] = sol.total(pname, 'mol/L') * 1000.0
                elif unit in ('meq/L', 'meq/l'):
                    C_out[i] = sol.total(pname, 'eq/L') * 1000.0
                else:
                    C_out[i] = sol.total(pname, 'mg/L')
            except Exception:
                pass   # houd oorspronkelijke waarde bij als PHREEQC faalt

        # Zorg dat concentraties niet negatief worden
        C_out = np.maximum(C_out, 0.0)
        return C_out

    # ── Hulpfuncties ──────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset PHREEQC-instantie (nuttig bij herstart simulatie)."""
        if self._pp is not None:
            try:
                self._pp.ip.CleanupSimulation()
            except Exception:
                pass
        self._pp   = None
        self._step = 0

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
        solver = InzingaFlowSolver("net.inp", n_species=2, geochem=geo)
        solver.inject("R1", C_vector=[1.0, 7.5], volume=0.05)
    """
    smap = SpeciesMap(
        species_names=['Cl2_total', 'pH'],
        phreeqc_names=['Cl',        None],   # pH wordt via is_pH gelezen
        units        =['mg/L',      ''],
        is_pH        =[False,       True],
    )
    # Voeg pH toe als expliciete stof die PHREEQC teruggeeft
    smap.phreeqc_names[1] = None   # pH via sol.pH, niet via sol.total()
    smap.is_pH[1]         = True

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
