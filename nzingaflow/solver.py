# nzingaflow/solver.py
"""
NzingaFlowSolver: volledige LTA-solver gekoppeld aan EPANET via epynet.

Nieuw in deze versie
---------------------
- Wandreacties via twee-film-model (k_wall per leiding per stof)
- Tank-knoopmodel (CSTR, impliciet Euler)
- Automatische CFL-tijdstapcontrole
- Massabalansregistratie via MassBalanceTracker
- Pipe-overgang via vectorized routing (node_type/node_out0 cache)
"""

from __future__ import annotations
from collections import defaultdict
from typing import List, Tuple
import numpy as np



def _read_tank_mix_models(inp_path: str) -> dict:
    """
    Lees het [MIXING]-blok van een .inp: {tanknaam: model} (model in hoofdletters).

    NzingaFlow rekent elke tank als MIXED (CSTR). Andere modellen (FIFO, LIFO,
    2COMP) worden hier wel gelezen zodat de solver kan waarschuwen.
    """
    models: dict = {}
    try:
        with open(inp_path, encoding='utf-8', errors='replace') as fh:
            in_sec = False
            for raw in fh:
                line = raw.split(';', 1)[0].strip()
                if not line:
                    continue
                if line.startswith('['):
                    in_sec = line.upper().startswith('[MIXING')
                    continue
                if in_sec:
                    parts = line.split()
                    if len(parts) >= 2:
                        models[parts[0]] = parts[1].upper()
    except OSError:
        pass
    return models


class NzingaFlowSolver:
    """
    Vectorized Lagrangian Transport Approach solver voor EPANET-netwerken.

    Kenmerken
    ----------
    Multi-species     C heeft shape (n, n_species); alle stoffen in één operatie.
    Wandreacties      Twee-film-model per leiding; afhankelijk van Re en k_wall.
    Tanks             Massabehoudend CSTR-model (impliciet Euler, stabiel voor grote dt):
                      nemen aankomende segmenten op en geven water als nieuwe
                      segmenten af aan hun uitgaande leidingen.
    EPS               Hydraulica bijgewerkt via update_hydraulics(simtime).
    Splitsingen       Segmenten proportioneel opgesplitst naar debiet.
    Auto-resize       SegmentStore verdubbelt capaciteit automatisch.
    Stabiliteitscheck CFL + reactie-dt gecontroleerd bij initialisatie en stap.
    Massabalans       MassBalanceTracker bijgehouden per stap (opt-in).

    Gebruik
    -------
        solver = NzingaFlowSolver(
            "netwerk.inp",
            n_species=2,
            k_wall=np.array([[1e-5, 0.0]] * n_pipes),  # [m/s] per pipe per stof
        )
        solver.inject("R1", C_vector=[1.0, 0.5], volume=0.05)
        for step in range(n_steps):
            node_C = solver.step(dt=5.0, decay_k=[0.001, 0.0])
    """

    def __init__(
        self,
        inp_path:         str,
        n_species:        int = 1,
        capacity:         int = 200_000,
        k_wall:           np.ndarray | None = None,
        D_mol:            float = 1.3e-9,
        nu:               float = 1e-6,
        track_mass:       bool = False,
        geochem           = None,
        temperature:      float | None = None,
        leakage_fraction: float = 0.0,
        wall_mode:        str   = 'two_film',
        include_valves:   bool  = False,
        skip_short_valves: bool = True,
        cache_links:      bool  = False,
        reversal_deadband: float = 1e-5,
        auto_reservoir_sources: bool = True,
    ):
        """
        Parameters
        ----------
        inp_path          : pad naar EPANET .inp bestand
        n_species         : aantal te simuleren stoffen
        capacity          : initiële SegmentStore capaciteit (auto-resize)
        k_wall            : wandreactiesnelheid [m/s]; None = geen wandreacties.
                            shape (n_pipes, n_species) of (n_species,) voor uniform
        D_mol             : moleculaire diffusiviteit bij 20°C [m²/s]
        nu                : kinematische viscositeit [m²/s]
        track_mass        : bijhouden van massabalans via MassBalanceTracker
        geochem           : GeochemSolver instantie; vervangt eerste-orde bulkverval.
                            Vereist: pip install phreeqpython
        temperature       : watertemperatuur [°C]. Activeert Arrhenius/Hayduk-Laudie
                            correctie op D_mol en k_wall (Ea_D≈17 kJ/mol, θ_w=1.047).
                            None = geen temperatuurcorrectie (Rossman 1994-compatibel).
        leakage_fraction  : fractie van leiding-debiet dat lekt (0.0–1.0).
                            Elk segment verlaat per tijdstap proportioneel volume
                            zonder chemicaliënadditie (concentratie onveranderd).
                            Typisch 0.05–0.20 voor NL-distributienetwerken.
        wall_mode         : hoe k_wall wordt geïnterpreteerd:
            'two_film' (standaard) — EPANET-compatibel: k_eff=k_f·k_w/(k_f+k_w).
                Gebruik als k_wall uit EPANET-kalibratie komt.
            'direct' — k_wall is al k_eff: k_vol=k_wall·4/D, geen filmweerstand.
                Gebruik als k_wall al een effectieve waarde is.
        include_valves    : als True worden afsluiters (PRV, PSV, TCV, FCV, GPV,
                            PCV) meegenomen in de transporttopologie.  EPANET lost
                            de hydraulica voor afsluiters altijd correct op; deze
                            optie zorgt dat NzingaFlow de verblijftijd en het
                            stoftransport over afsluiters ook berekent.
                            Standaard False voor achterwaartse compatibiliteit.
        cache_links       : link-objectlijst niet na elke hydraulische oplossing opnieuw
                            opbouwen (zie HydraulicModel).  Zelfde resultaten, sneller
                            bij grote netwerken.  Standaard False.
        auto_reservoir_sources : bool
            Injecteer reservoirs met head > 0 automatisch: bron (CONCEN, uit
            [SOURCES] met patroon) of, zonder bron, de beginkwaliteit; het volume
            is het uitgaande debiet × dt. Reservoirs met head 0 worden overgeslagen.
            Zet op False als je reservoirs zelf via inject() wilt voeden (anders
            dubbele massa).
        skip_short_valves : (alleen met include_valves=True) open afsluiters waarvan
                            de lengte kleiner is dan de afstand die water in één
                            tijdstap aflegt (L < v·dt) worden overgeslagen: een
                            segment passeert ze binnen dezelfde stap, zonder een
                            extra tijdstap te verliezen.  Gesloten afsluiters (status
                            CLOSED of debiet ≈ 0) worden NOOIT overgeslagen en blokkeren.
        """
        from .hydraulics import HydraulicModel
        from .segments   import SegmentStore
        from .stability  import MassBalanceTracker

        self._tracker = None
        self._tank_release: list = []
        import warnings as _warnings
        self.tank_mix_models = _read_tank_mix_models(inp_path)
        for _tk, _mdl in self.tank_mix_models.items():
            if _mdl != 'MIXED':
                _warnings.warn(
                    f"Tank {_tk!r}: tankmodel {_mdl} in het .inp wordt niet ondersteund; "
                    f"NzingaFlow rekent deze tank als MIXED (CSTR). Resultaten "
                    f"kunnen afwijken van EPANET.",
                    stacklevel=2,
                )
        self.hyd = HydraulicModel(inp_path, include_valves=include_valves,
                                 cache_links=cache_links,
                                 reversal_deadband=reversal_deadband)
        self.hyd.solve()

        (
            self.pipe_start,
            self.pipe_end,
            self.pipe_length,
            self.pipe_area,
            self.node_count,
            self.pipe_ids,
            self.node_names,
        ) = self.hyd.get_topology()

        n_pipes = len(self.pipe_ids)
        self.n_species = n_species
        self.segments  = SegmentStore(capacity, n_species)
        self._D_mol            = D_mol
        self._nu               = nu
        self._temperature      = temperature          # [°C] of None
        self._leakage_fraction = float(leakage_fraction)  # 0.0 = geen lekkage
        if wall_mode not in ('two_film', 'direct'):
            raise ValueError(
                f"wall_mode moet 'two_film' of 'direct' zijn, niet {wall_mode!r}"
            )
        self._wall_mode = wall_mode

        # Naam → index
        self.node_index = {name: i for i, name in enumerate(self.node_names)}
        self.pipe_index = {pid:  i for i, pid  in enumerate(self.pipe_ids)}

        # Tank-knoopindices (epynet Tank-objecten)
        self._tank_nodes = self._detect_tank_nodes()
        self._tank_V     = self._get_tank_volumes()   # (n_tanks,) m³ — gecached
        self._tank_C     = np.zeros((len(self._tank_nodes), n_species))  # CSTR toestand

<<<<<<< HEAD
=======
        # Automatische reservoir- en junctionbronnen (zie network_sources.py)
        self._qual_clock: float = 0.0            # kwaliteitstijd [s], telt mee met elke step()
        self._res_sources: list = []
        self._junc_sources: list = []
        self._warned_auto_res: set[str] = set()
        self._in_auto_inject: bool = False
        if auto_reservoir_sources:
            self._res_sources = self._build_reservoir_sources(n_species)
            self._junc_sources = self._build_junction_sources(n_species)

>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d
        # Uitgaande leidingen per knoop (herbouwd na flow-reversal)
        self._node_outpipes: dict[int, list[int]] = defaultdict(list)

        # Cache van in/out leidingen per tank — herbouwd bij update_hydraulics()
        self._tank_in_pipes:  list[np.ndarray] = []   # per tank: array van inkomende pipe-indices
        self._tank_out_pipes: list[np.ndarray] = []   # per tank: array van uitgaande pipe-indices

        # Per-knoop exit-routing caches (herbouwd bij update_hydraulics)
        # _node_type[nd]  : 0=eindknoop, 1=doorgaand, >=2=splitsing
        # _node_out0[nd]  : index van eerste (hoogste-debiet) uitgangsleiding
        # Hiermee wordt het dominante pad in _handle_exits volledig vectorized.
        self._node_type: np.ndarray | None = None   # (node_count,) int32
        self._node_out0: np.ndarray | None = None   # (node_count,) int32
<<<<<<< HEAD
=======

        # Externe instroom op knopen met negatieve vraag (EPANET). Concentratie
        # van het ingevoerde water per knoop; standaard 0 (zoals EPANET zonder
        # SOURCE). Zie set_source_quality().
        self._node_demand: np.ndarray | None = None  # (node_count,) m³/s, + = uit
        self._ext_C: np.ndarray | None = None        # (node_count, n_species)
        # Persistente knoopconcentratie (laatste aankomst) en aankomstvlag
        self._node_C_hold: np.ndarray | None = None  # (node_count, n_species)
        self._res_init_C: np.ndarray | None = None   # beginkwaliteit per knoop (vast, voor reservoirs)
        self.node_arrived: np.ndarray | None = None  # (node_count,) bool, deze stap
        # Uitlaat-concentratie per leiding (laatst aangekomen), voor knoopmenging
        self._pipe_out_C: np.ndarray | None = None   # (n_pipes, n_species)
        self._arrived_nodes: np.ndarray | None = None
        # Gemengde leidingen (tank-achtig, geen segmenten): masker, concentratie
        # en volume per leiding. Zie set_mixed_pipes().
        self._mixed_mask: np.ndarray | None = None
        self._mixed_C: np.ndarray | None = None      # (n_pipes, n_species)
        self._mixed_V: np.ndarray | None = None      # (n_pipes,) m3
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d

        # Wandreacties
        self._k_wall_ms  = None    # (n_pipes, n_species) [m/s], None = geen
        self._k_wall_vol = None    # (n_pipes, n_species) [1/s], berekend bij solve
        if k_wall is not None:
            self._k_wall_ms = self._broadcast_kwall(np.asarray(k_wall, dtype=float),
                                                     n_pipes, n_species)

        # Gecombineerde decay-factoren: exp(-(k_bulk + k_wall) * dt)
        # Gecached per (n_pipes, n_species); herbouwd bij nieuwe dt of hydraulica.
        self._combined_exp: np.ndarray | None = None
        self._combined_exp_dt: float = -1.0   # dt waarvoor gecached
        self._combined_exp_k: np.ndarray | None = None  # decay_k waarvoor gecached

        # Hydraulica-cache
        self._flow:     np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._reversed: np.ndarray | None = None   # (n_pipes,) bool — t.o.v. EPANET-definitie
<<<<<<< HEAD
=======
        # Afsluiters overslaan (zie skip_short_valves)
        self._skip_short_valves = bool(skip_short_valves) and include_valves
        self._valve_mask   = (self.hyd.get_valve_mask() if include_valves
                              else np.zeros(n_pipes, dtype=bool))
        self._valve_closed = np.zeros(n_pipes, dtype=bool)
        self._skip_mask    = np.zeros(n_pipes, dtype=bool)
        self._skip_target  = np.arange(n_pipes, dtype=np.int32)
        self._skip_time    = np.zeros(n_pipes, dtype=np.float64)
        self._skip_dt      = -1.0
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d
        # Corrigeer start/end voor de initiële stromingsrichting vóór routing-caches.
        self._apply_flow_reversal()
        self._rebuild_tank_pipe_cache()
        self._rebuild_node_routing_cache()
        self._update_kwall_vol()   # bereken initiële k_wall_vol

        # Massabalans
        self._track_mass = track_mass
        self._tracker    = MassBalanceTracker(n_species) if track_mass else None

        # Geochemie (PhreeqPython)
        self._geochem    = geochem   # GeochemSolver of None

        self._step_counter = 0
        self._last_dt: float = 1.0   # bijgehouden per step()-aanroep; gebruikt door booster_inject()

        # ── Pre-allocatie van herbruikbare werkbuffers ────────────────────────
        # Doel: geen heap-allocaties meer in de hot-path (step() per tijdstap).
        # Alle buffers worden aangemaakt op basis van de maximale verwachte grootte;
        # ze worden in-place overschreven zonder nieuwe array-objecten aan te maken.
        nc = self.node_count
        self._buf_node_C    = np.zeros((nc, n_species), dtype=np.float64)  # node_mixing uitvoer
        self._buf_node_flow = np.zeros(nc,              dtype=np.float64)  # debiet per knoop (geochem)
        self._buf_exit_C    = np.zeros((capacity, n_species), dtype=np.float64)  # geëxiteerde concentraties
        self._buf_exit_V    = np.zeros(capacity,              dtype=np.float64)  # geëxiteerde volumes
        self._buf_n_exit    = 0   # aantal geldig gevulde rijen in _buf_exit_*
        self._buf_wC        = np.zeros((capacity, n_species), dtype=np.float64)  # gewogen massa in node_mixing
        self._buf_exit_mask = np.zeros(capacity, dtype=np.bool_)                  # exit-detectie resultaat
        # Tank-buffers (klein; n_tanks typisch < 10)
        n_tanks = len(self._tank_nodes)
        self._buf_Q_in  = np.zeros(max(n_tanks, 1), dtype=np.float64)
        self._buf_Q_out = np.zeros(max(n_tanks, 1), dtype=np.float64)
        self._buf_C_in  = np.zeros((max(n_tanks, 1), n_species), dtype=np.float64)

        # Koppel resize-callback: als SegmentStore zijn capaciteit verdubbelt,
        # schalen de exit-buffers mee zodat ze nooit te klein zijn.
        self.segments.on_resize = self._on_store_resize

    # ── Numba JIT warmup ──────────────────────────────────────────────────────

    def warmup_numba(self) -> None:
        """
        Trigger Numba JIT-compilatie vóór de eerste simulatiestap.

        Roep eenmalig aan na het aanmaken van de solver (~0.5-2 s eenmalig).
        Daarna start elke tijdstap zonder compilatie-overhead.
        Doet niets als Numba niet geïnstalleerd is.
        """
        from .lta     import warmup_numba as _warmup
        from .merging import warmup_numba_merging as _warmup_merge
        _warmup(self.n_species)
        _warmup_merge(self.n_species)

    # ── Beginkwaliteit ───────────────────────────────────────────────────────

    def set_initial_quality(
        self,
        node_quality: np.ndarray | dict,
        simtime:      int = 0,
        dt:           float | None = None,
    ) -> None:
        """
        Initialiseer het netwerk met beginconcentraties per knoop.

        Vult elke leiding met één segment dat de gehele leidinginhoud
        vertegenwoordigt (volume = L × A) en de concentraties van de
        upstream knoop draagt.  Dit simuleert de EPANET-MSX [QUALITY]
        GLOBAL/NODE initialisatie.

        Roep aan vóór EPSRunner.run().  De methode roept intern
        update_hydraulics(simtime) aan zodat de stroomrichting bekend is
        vóór de segmenten worden aangemaakt.

        Parameters
        ----------
        node_quality : ndarray (node_count, n_species)  of  dict
            Beginconcentraties per knoop.
            Als dict: {knoopnaam: ndarray(n_species)} — alle niet-genoemde
            knopen krijgen concentratie 0.
            Als ndarray: directe concentratiematrix, rij-index = node_index.
        simtime : int
            EPANET-simulatietijd [s] voor de hydraulische initialisatie.
            Standaard 0 (beginconditie).
        dt : float, optional
            Kwaliteitstijdstap [s] (zelfde als EPSRunner/step). Elke leiding
            wordt opgedeeld in segmenten van ongeveer v·dt, zodat het
            beginwater met de juiste aankomsttijden de uitgang bereikt (punt 3).
            Zonder dt (en zonder eerdere step) blijft één segment per leiding,
            met een waarschuwing.

        Voorbeelden
        -----------
        # Globale beginconcentraties (EPANET-MSX QUALITY GLOBAL)
        C0 = np.zeros((solver.node_count, solver.n_species))
        C0[:, SP['ALK']] = 0.004
        C0[:, SP['H']]   = 2.818e-8
        solver.set_initial_quality(C0)

        # Mix van globaal en node-specifiek via dict
        C0 = np.zeros((solver.node_count, solver.n_species))
        C0[:, SP['ALK']] = 0.004         # globaal
        solver.set_initial_quality(C0)
        # Daarna node-overschrijvingen via losse inject_pipe aanroepen,
        # of geef ze meteen mee als dict:
        solver.set_initial_quality({
            '__global__': C_global,       # speciale sleutel voor alle knopen
            '4':          C_node4,        # overschrijft knoop '4'
            '5':          C_node5,
        })
        """
        # ── Zet hydraulica op voor simtime ─────────────────────────────────
        self.update_hydraulics(simtime=simtime)
        flow, velocity = self._get_hydraulics()

        # ── Bouw concentratiematrix ─────────────────────────────────────────
        n_sp = self.n_species
        nc   = self.node_count

        if isinstance(node_quality, dict):
            C_node = np.zeros((nc, n_sp), dtype=np.float64)
            # '__global__' sleutel vult alle knopen
            if '__global__' in node_quality:
                C_node[:] = np.asarray(node_quality['__global__'],
                                       dtype=np.float64)
            # Knoop-specifieke overschrijvingen
            for uid, c_vec in node_quality.items():
                if uid == '__global__':
                    continue
                idx = self.node_index.get(uid)
                if idx is None:
                    raise KeyError(
                        f"set_initial_quality: onbekende knoopnaam {uid!r}"
                    )
                C_node[idx] = np.asarray(c_vec, dtype=np.float64)
        else:
            C_node = np.asarray(node_quality, dtype=np.float64)
            if C_node.shape != (nc, n_sp):
                raise ValueError(
                    f"set_initial_quality: verwacht shape ({nc}, {n_sp}), "
                    f"gekregen {C_node.shape}"
                )

        # ── Wis eventuele bestaande segmenten ──────────────────────────────
        if self._tracker:
            # Eerder geboekte massa die hier verdwijnt, terugboeken (anders telt
            # een tweede aanroep de beginmassa dubbel).
            self._tracker.record_injection_mass(-self._tracker.current_mass(self.segments))
        self.segments.n = 0

        # Tanks starten met de concentratie van hun knoop.
        if len(self._tank_nodes) > 0:
            C_t_new = C_node[self._tank_nodes].copy()
            if self._tracker and len(self._tank_V) == len(self._tank_nodes):
                self._tracker.record_injection_mass(
                    ((C_t_new - self._tank_C) * self._tank_V[:, np.newaxis]).sum(axis=0)
                )
            self._tank_C[:] = C_t_new

        # Persistente knoopconcentratie start met de beginconditie (punt 5):
        # knopen die pas later een aankomst krijgen, tonen tot dan deze waarde.
        self._node_C_hold = C_node.copy()
        # Beginkwaliteit blijft bewaard: een reservoir zonder bron houdt in EPANET
        # altijd deze waarde, ook als het zelf water ontvangt.
        self._res_init_C = C_node.copy()
        # Uitlaatconcentratie per leiding = begininhoud (stroomafwaarts, zie hieronder)
        self._pipe_out_C = C_node[self.pipe_end].copy()

        # ── Vul elke leiding met beginsegmenten van lengte ≈ v·dt ─────────────
        # Het water van de leiding komt op één positie per segment te staan:
        # het segment op x_k = k·ℓ krijgt aankomst na (L − x_k)/v, zoals de
        # echte inhoud. Eén blok op x = 0 kwam als één lading aan.
        dt_seg = dt if dt is not None else getattr(self, '_last_dt', None)
        if dt is None and dt_seg is None:
            import warnings
            warnings.warn(
                "set_initial_quality: geen dt opgegeven; elke leiding krijgt één "
                "beginsegment (aankomsttijd onnauwkeurig). Geef dt= mee.",
                stacklevel=2,
            )
        vel_all = velocity
        n_added = 0
        for pi in range(len(self.pipe_ids)):
            L   = float(self.pipe_length[pi])
            A   = float(self.pipe_area[pi])
<<<<<<< HEAD
            vol = L * A    # totale leidinginhoud [m³]
            src_node = int(self.pipe_start[pi])
            self.segments.add(
                pipe=pi, x=0.0, volume=vol,
                C_vector=C_node[src_node],
            )
            self._sync_geochem_new_segment()
            n_added += 1
=======
            vol_total = L * A    # totale leidinginhoud [m³]
            # EPANET vult een leiding met de beginkwaliteit van de stroomafwaartse
            # knoop (gecontroleerd op pulse4.inp: leiding R1→J1 = 510 = J1,
            # leiding J10→T1 = 530 = T1), niet van de upstream knoop.
            src_node = int(self.pipe_end[pi])
            v = float(vel_all[pi])
            if dt_seg and dt_seg > 0.0 and v > self._V_EPS:
                # Begrensd: bij (bijna) stilstaand water geeft L/(v·dt) miljarden
                # segmenten (wst.inp: v tot 1e-12). Stilstaand water arriveert
                # binnen de simulatie toch nauwelijks; zie _MAX_INIT_SEGMENTS.
                n_seg = int(min(self._MAX_INIT_SEGMENTS,
                                max(1, np.ceil(L / (v * dt_seg)))))
            else:
                n_seg = 1
            cell = L / n_seg
            vol  = vol_total / n_seg
            for k in range(n_seg):
                self.segments.add(
                    pipe=pi, x=k * cell, volume=vol,
                    C_vector=C_node[src_node],
                )
                self._sync_geochem_new_segment()
                n_added += 1
            if self._tracker:
                # Beginmassa boeken als 'injectie'; anders blijft injected = 0 en
                # meldt het rapport (total_in = 0 → fout 0) onterecht 'ok'.
                self._tracker.record_injection(C_node[src_node], vol_total)
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d

        # Invalideer hydraulica-cache (segmenten zijn nieuw)
        self._combined_exp_dt = -1.0
        self._combined_exp_k = None

    # ── Hydraulica ────────────────────────────────────────────────────────────

    def update_hydraulics(self, simtime: int = 0) -> None:
        """
        Herbereken hydraulica voor een EPS-tijdstip [s].

        Detecteert automatisch flow reversals en past pipe_start, pipe_end
        en _node_outpipes aan zodat advectie altijd in de juiste richting loopt.
        """
        self.hyd.solve(simtime=simtime)
        # Kwaliteitsklok volgt de hydraulische tijd (patronen van automatische bronnen);
        # binnen een hydraulische stap telt step() er dt bij op.
        self._qual_clock = float(simtime)
        self._flow     = None
        self._velocity = None
<<<<<<< HEAD
=======
        # Knoopvraag is optioneel: een hydraulische bron zonder get_node_demand()
        # (bv. een eenvoudige stub of een andere solver) krijgt geen externe
        # instroom. Zo blijft de hydraulische bron verwisselbaar (punt 7).
        _get_demand = getattr(self.hyd, 'get_node_demand', None)
        self._node_demand = _get_demand() if _get_demand is not None else None
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d
        # _reversed niet wissen: _apply_flow_reversal() vergelijkt met de
        # vorige masker om segmenten alleen te spiegelen bij échte richtingswissel.
        self._apply_flow_reversal()
        self._update_kwall_vol()
        self._combined_exp_dt = -1.0   # invalideer na hydraulica-update
        self._combined_exp_k = None
<<<<<<< HEAD
=======
        self._skip_dt = -1.0           # skip-cache hangt af van flow/status/dt
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d
        # Tankvolumes kunnen veranderen tijdens EPS (peil varieert)
        if len(self._tank_nodes) > 0:
            new_V = self._get_tank_volumes()
            if self._tracker and len(new_V) == len(self._tank_V):
                # EPANET zet het tankvolume terug op de eigen waarde: dat is water
                # dat met de tankconcentratie bijkomt/verdwijnt → als injectie boeken.
                self._tracker.record_injection_mass(
                    (self._tank_C * (new_V - self._tank_V)[:, np.newaxis]).sum(axis=0)
                )
            self._tank_V = new_V
            self._rebuild_tank_pipe_cache()
        self._rebuild_node_routing_cache()

    def _combined_exp_needs_rebuild(self, dt: float, decay_k: np.ndarray) -> bool:
        """
        True als de gecachede combined_exp niet meer geldt voor (dt, decay_k).

        Vergelijkt decay_k exact (np.array_equal). Niet np.allclose: de
        standaard atol=1e-8 is in 1/s zo groot (≈ 0,001/dag) dat een
        verandering van bijv. [0.0] naar [5e-9] als "gelijk" gold en het
        nieuwe verval stilzwijgend genegeerd werd.
        """
        return (
            self._combined_exp is None
            or self._combined_exp_dt != dt
            or self._combined_exp_k is None
            or self._combined_exp_k.shape != decay_k.shape
            or not np.array_equal(self._combined_exp_k, decay_k)
        )

    def _apply_flow_reversal(self) -> None:
        """
        Pas pipe_start, pipe_end en _node_outpipes aan op basis van de
        actuele stromingsrichting. Wordt aangeroepen na elke hydraulica-update.

        Leidingen waarvan EPANET een negatief debiet rapporteert stromen
        feitelijk in omgekeerde richting. We wisselen pipe_start/pipe_end om
        zodat de LTA-advectie (x loopt van 0 naar pipe_length) altijd klopt.

        Segmentposities worden alleen gespiegeld voor leidingen waarvan de
        stromingsrichting *sinds de vorige aanroep* is omgedraaid. Spiegelen
        op de actuele reversed-masker zou x bij elke hydraulica-update
        opnieuw omklappen zolang het debiet negatief blijft, en zou x niet
        terugzetten wanneer de stroom weer de definitierichting op gaat.
        """
        _, _, reversed_mask = self.hyd.get_hydraulic_state()
        prev_mask = self._reversed

        if not reversed_mask.any():
            # Geen reversals: zorg dat topologie terug op EPANET-waarden staat
            (
                self.pipe_start, self.pipe_end,
                self.pipe_length, self.pipe_area,
                self.node_count, self.pipe_ids, self.node_names,
            ) = self.hyd.get_topology()
        else:
            (
                self.pipe_start, self.pipe_end,
                self.pipe_length, self.pipe_area,
                self.node_count, self.pipe_ids, self.node_names,
            ) = self.hyd.get_topology_with_reversal()

        # Herbereken uitgaande leidingen per knoop
        self._node_outpipes.clear()
        for p, s in enumerate(self.pipe_start):
            self._node_outpipes[int(s)].append(p)

        # Spiegelen: x_new = L - x_old, alleen bij échte richtingswissel.
        n = self.segments.n
        if n > 0:
            if prev_mask is None:
                prev_mask = np.zeros_like(reversed_mask)
            changed = prev_mask != reversed_mask
            if changed.any():
                seg_pipe   = self.segments.pipe[:n]
                chg_pipes  = np.where(changed)[0]
                in_chg     = np.isin(seg_pipe, chg_pipes)
                if in_chg.any():
                    L_seg = self.pipe_length[seg_pipe[in_chg]]
                    self.segments.x[:n][in_chg] = L_seg - self.segments.x[:n][in_chg]
                    np.clip(self.segments.x[:n], 0.0, None,
                            out=self.segments.x[:n])
<<<<<<< HEAD
=======
                    # Uitlaatconcentratie van omgekeerde leidingen opnieuw bepalen:
                    # na de omkering is het water dat nu naar de knoop stroomt het
                    # water dat het dichtst bij het nieuwe uiteinde staat. Zonder
                    # dit gebruikt de knoopmenging een verouderde waarde (één
                    # stap fout bij elke omkering).
                    if self._pipe_out_C is not None:
                        for p in chg_pipes:
                            ip = np.where(in_chg & (seg_pipe == p))[0]
                            if ip.size:
                                j = ip[np.argmax(self.segments.x[ip])]
                                self._pipe_out_C[p] = self.segments.C[j]
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d

        self._reversed = reversed_mask.copy()

    def _get_hydraulics(self) -> tuple:
        if self._flow is None:
            self._flow, self._velocity, self._reversed = self.hyd.get_hydraulic_state()
        return self._flow, self._velocity

    def _update_kwall_vol(self) -> None:
        """
        Herbereken k_wall_vol [1/s] na elke hydraulica-update.

        wall_mode='two_film'  (standaard, EPANET-compatibel):
            Twee-film serieweerstand: k_eff = k_f·k_w/(k_f+k_w)
            k_f afhankelijk van Re via Sherwood-correlatie.
            Gebruik als k_wall afkomstig is uit EPANET-kalibratie —
            EPANET past intern hetzelfde model toe bij orde-1 wandreacties.

        wall_mode='direct':
            k_wall direct als k_eff: k_vol=k_wall·4/D, geen filmweerstand.
            Gebruik als k_wall al een effectieve waarde is (bijv. teruggerekend
            zonder twee-film model). Voorkomt dubbele filmweerstand.
        """
        if self._k_wall_ms is None:
            self._k_wall_vol = None
            return

        diam = np.sqrt(4 * self.pipe_area / np.pi)   # m

        if self._wall_mode == 'direct':
            # k_wall is al k_eff — geen filmweerstand berekenen.
            # Optioneel: temperatuurcorrectie θ_w=1.047 (Rossman 2000).
            k_w = np.asarray(self._k_wall_ms, dtype=np.float64).copy()
            if self._temperature is not None:
                k_w *= 1.047 ** (self._temperature - 20.0)
            self._k_wall_vol = k_w * (4.0 / diam[:, np.newaxis])
            return

        # two_film (standaard): serieweerstand film + wandreactie
        from .lta import compute_wall_k
        _, vel = self._get_hydraulics()
        self._k_wall_vol = compute_wall_k(
            diam, vel, self._k_wall_ms,
            D_mol        = self._D_mol,
            nu           = self._nu,
            pipe_length  = self.pipe_length,
            temperature  = self._temperature,
        )

    # ── Tijdstap-stabiliteitscontrole ─────────────────────────────────────────

    def check_stability(self, dt: float, decay_k, warn: bool = True) -> dict:
        """
        Controleer CFL en reactie-stabiliteit voor de opgegeven tijdstap.

        Returns een rapport-dict; zie stability.check_dt() voor inhoud.
        """
        from .stability import check_dt
        _, vel = self._get_hydraulics()
        return check_dt(
            dt, self.pipe_length, vel,
            np.asarray(decay_k, dtype=float),
            self._k_wall_vol, warn=warn,
        )

    def recommended_dt(self, decay_k) -> float:
        """Geeft de aanbevolen tijdstap terug op basis van CFL en reactie."""
        from .stability import recommended_dt
        _, vel = self._get_hydraulics()
        return recommended_dt(
            self.pipe_length, vel,
            np.asarray(decay_k, dtype=float),
            self._k_wall_vol,
        )

    # ── Injectie ──────────────────────────────────────────────────────────────

    def _build_reservoir_sources(self, n_species: int) -> list:
        """Lijst (naam, knoopindex, sterkte, patroonwaarden, patroonstap, patroonstart)."""
        from .network_sources import reservoir_sources, pattern_values, pattern_timing
        net = self.hyd.net
        found: list = []
        pstep, pstart = pattern_timing(net)
        for r in reservoir_sources(net):
            if r.name not in self.node_index:
                continue
            if r.kind is not None and r.kind != "CONCEN":
                raise NotImplementedError(
                    f"Reservoir '{r.name}': bronsoort {r.kind} wordt niet automatisch "
                    f"ondersteund. Gebruik auto_reservoir_sources=False en injecteer "
                    f"zelf met inject().")
            if r.kind == "CONCEN" and n_species != 1:
                raise NotImplementedError(
                    f"Reservoir '{r.name}' heeft een CONCEN-bron; automatische "
                    f"reservoirbronnen werken alleen met n_species=1. Gebruik "
                    f"auto_reservoir_sources=False.")
            values: list = pattern_values(net, r.pattern) if r.pattern is not None else []
            found.append((r.name, self.node_index[r.name], r.strength, values,
                          pstep, pstart))
        return found

    def _build_junction_sources(self, n_species: int) -> list:
        """Bronnen op junctions (negatieve vraag = instroom). Zelfde tuple-vorm als reservoirs."""
        from .network_sources import junction_sources, pattern_values, pattern_timing
        net = self.hyd.net
        found: list = []
        pstep, pstart = pattern_timing(net)
        for r in junction_sources(net):
            if r.name not in self.node_index:
                continue
            if r.kind != "CONCEN":
                import warnings
                warnings.warn(
                    f"Knoop '{r.name}': bronsoort {r.kind} op een junction wordt niet "
                    f"automatisch toegepast (alleen CONCEN). Gebruik set_source_quality().",
                    UserWarning, stacklevel=3)
                continue
            if n_species != 1:
                raise NotImplementedError(
                    f"Knoop '{r.name}' heeft een CONCEN-bron; automatische bronnen werken "
                    f"alleen met n_species=1. Gebruik auto_reservoir_sources=False.")
            values: list = pattern_values(net, r.pattern) if r.pattern is not None else []
            found.append((r.name, self.node_index[r.name], r.strength, values, pstep, pstart))
        return found

    def _apply_junction_sources(self, t: float) -> None:
        """Stel de instroomconcentratie in van junctions met bron en negatieve vraag."""
        if not self._junc_sources or self._node_demand is None:
            return
        from .network_sources import pattern_multiplier
        for name, ni, strength, values, pstep, pstart in self._junc_sources:
            if self._node_demand[ni] < 0.0:
                c = strength * pattern_multiplier(values, t, pstep, pstart)
                self.set_source_quality(name, np.array([c]))

    def _auto_inject_reservoirs(self, dt: float) -> None:
        """Voed actieve reservoirs met hun bron; wordt aan het begin van elke step() aangeroepen."""
        t = self._qual_clock
        self._qual_clock += float(dt)
        self._apply_junction_sources(t)
        if not self._res_sources:
            return
        from .network_sources import pattern_multiplier
        flow, _ = self._get_hydraulics()
        for name, ni, strength, values, pstep, pstart in self._res_sources:
            vol = float(flow[self.pipe_start == ni].sum()) * float(dt)
            if vol <= 0.0:
                continue
            if strength is None:
                # Geen bron: vaste beginkwaliteit van het reservoir (zoals EPANET)
                c = (self._res_init_C[ni].copy() if self._res_init_C is not None
                     else np.zeros(self.n_species))
            else:
                c = np.array([strength * pattern_multiplier(values, t, pstep, pstart)])
            self._in_auto_inject = True
            try:
                self.inject(name, c, vol)
            finally:
                self._in_auto_inject = False

    def inject(self, node_uid: str, C_vector, volume: float) -> None:
        """
        Injecteer vanuit een knoop proportioneel over alle uitgaande leidingen.

        Het opgegeven volume wordt verdeeld naar rato van het debiet per leiding,
        zodat de massa-injectie overeenkomt met de werkelijke stroomverdeling.
        Bij slechts één uitgaande leiding gaat het volledige volume daarheen.

        Parameters
        ----------
        node_uid : EPANET knoopnaam (bijv. 'R1', 'J1')
        C_vector : concentraties per stof, lengte n_species
        volume   : totaal segmentvolume [m³]
        """
        node_idx = self.node_index.get(node_uid)
        if node_idx is None:
            raise KeyError(f"Onbekende knoopnaam: '{node_uid}'")
        if not self._in_auto_inject \
                and any(src[1] == node_idx for src in self._res_sources) \
                and node_uid not in self._warned_auto_res:
            self._warned_auto_res.add(node_uid)
            import warnings
            warnings.warn(
                f"Reservoir '{node_uid}' wordt automatisch gevoed "
                f"(auto_reservoir_sources=True). Deze inject() voegt daar extra "
                f"water toe. Verwijder de handmatige injectie of zet "
                f"auto_reservoir_sources=False.", UserWarning, stacklevel=2)
        out = self._node_outpipes.get(node_idx, [])
        if not out:
            # Geen uitgaande leidingen op dit tijdstip (eindknoop of flow
            # reversal): booster-injectie stilt overgeslagen, geen fout.
            # Dit is consistent met EPANET-MSX gedrag bij Q_out = 0.
            return
        flow, _ = self._get_hydraulics()
        C_arr = np.asarray(C_vector, dtype=np.float64)

        if len(out) == 1:
            self.segments.add(pipe=out[0], x=0.0, volume=volume, C_vector=C_arr)
            self._sync_geochem_new_segment()
            if self._tracker:
                self._tracker.record_injection(C_arr, volume)
        else:
            # Verdeel volume proportioneel over uitgaande leidingen
            flows   = np.array([flow[p] for p in out], dtype=np.float64)
            tot_q   = flows.sum()
            if tot_q <= 0:
                # Geen debiet: injecteer in eerste leiding als fallback
                self.segments.add(pipe=out[0], x=0.0, volume=volume, C_vector=C_arr)
                self._sync_geochem_new_segment()
                if self._tracker:
                    self._tracker.record_injection(C_arr, volume)
                return
            for p, q in zip(out, flows):
                vol_p = volume * q / tot_q
                if vol_p > 0:
                    self.segments.add(pipe=p, x=0.0, volume=vol_p, C_vector=C_arr)
                    self._sync_geochem_new_segment()
            if self._tracker:
                self._tracker.record_injection(C_arr, volume)

    def inject_pipe(self, pipe_uid: str, C_vector, volume: float,
                    x: float = 0.0) -> None:
        """Injecteer direct in een leiding op positie x [m]."""
        pipe_idx = self.pipe_index[pipe_uid]
        C_arr = np.asarray(C_vector, dtype=np.float64)
        self.segments.add(pipe=pipe_idx, x=x, volume=volume, C_vector=C_arr)
        self._sync_geochem_new_segment()
        if self._tracker:
            self._tracker.record_injection(C_arr, volume)

    def _sync_geochem_new_segment(self, sol_num: int | None = None) -> None:
        """
        Koppel het laatst toegevoegde segment aan een oplossingnummer voor
        oplossingnummer-gebaseerde geochemie-modi (bijv. PhreeqSolutionMode).

        inject()/inject_pipe()/booster_inject() werken met een C_vector; die
        wordt door PhreeqSolutionMode genegeerd (het gebruikt geen C-matrix).
        Zonder deze koppeling zou ``sol_ids`` op de nieuwe index een
        verweesd of hergebruikt oplossingnummer bevatten na een eerdere
        remove()/add(). Standaard wordt de achtergrondoplossing gebruikt —
        de geïnjecteerde concentratie wordt dan NIET chemisch gerepresenteerd.
        Voor scheikundig correcte injecties: maak zelf een PHREEQC-oplossing
        aan en geef het nummer door via ``sol_num``, of roep na deze aanroep
        ``self._geochem.on_segment_added(self.segments.n - 1, sol_num)`` aan.
        """
        if self._geochem is not None and hasattr(self._geochem, 'sol_ids'):
            self._geochem.on_segment_added(self.segments.n - 1, sol_num)

    def booster_inject(
        self,
        node_uid:  str,
        C_set:     np.ndarray | list,
        flow_frac: float = 1.0,
    ) -> None:
        """
        Booster-injectie op een knoop: stel de concentratie in op een vaste
        waarde voor alle uitgaande leidingen (zoals een chloor-boosterstation).

        In tegenstelling tot inject() wordt hier geen nieuw segment aangemaakt
        op basis van een extern volume. In plaats daarvan worden de concentraties
        van bestaande segmenten aan het begin (x=0) van de uitgaande leidingen
        van de knoop overschreven.

        Als er nog geen segment aan het begin van een uitgaande leiding staat,
        wordt er één aangemaakt met een volume gelijk aan flow × qual_dt.
        Gebruik inject() als je liever een volume-gebaseerde injectie wilt.

        Parameters
        ----------
        node_uid  : EPANET knoopnaam (bijv. 'B1', 'J5')
        C_set     : (n_species,) doelconcentratie per stof [mg/L of dimensieloos]
                    Stoffen met C_set[s] < 0 worden niet gewijzigd.
        flow_frac : fractie van de uitgaande debieten waarop de booster werkt
                    (default 1.0 = alle uitgaande leidingen)

        Raises
        ------
        KeyError   : als node_uid niet bestaat
        ValueError : als de knoop geen uitgaande leidingen heeft
        """
        node_idx = self.node_index.get(node_uid)
        if node_idx is None:
            raise KeyError(f"Onbekende knoopnaam: '{node_uid}'")

        out = self._node_outpipes.get(node_idx, [])
        if not out:
            # Geen uitgaande leidingen op dit tijdstip (eindknoop of flow
            # reversal): booster-injectie stilt overgeslagen, geen fout.
            # Dit is consistent met EPANET-MSX gedrag bij Q_out = 0.
            return

        flow, _ = self._get_hydraulics()
        C_set   = np.asarray(C_set, dtype=np.float64)
        active  = C_set >= 0          # stoffen die we wél instellen

        # Selecteer de leidingen met het hoogste debiet (flow_frac)
        if flow_frac < 1.0:
            sorted_out = sorted(out, key=lambda p: -flow[p])
            cum = 0.0
            tot = sum(flow[p] for p in out)
            selected = []
            for p in sorted_out:
                selected.append(p)
                cum += flow[p]
                if cum >= flow_frac * tot:
                    break
        else:
            selected = out

        n = self.segments.n
        for pipe_idx in selected:
            # Zoek bestaand segment aan begin (x < kleine drempel)
            threshold = self.pipe_length[pipe_idx] * 0.01
            if n > 0:
                at_start = (
                    (self.segments.pipe[:n] == pipe_idx) &
                    (self.segments.x[:n] < threshold)
                )
                if at_start.any():
                    # Overschrijf concentraties voor actieve stoffen.
                    # Let op: dubbele fancy indexing (C[:n][mask]) levert een
                    # kopie op — schrijf daarom via expliciete integer-indices.
                    rows = np.where(at_start)[0]
                    cols = np.where(active)[0]
                    if self._tracker:
                        # Massa-wijziging door het overschrijven boeken als injectie
                        # (kan negatief zijn), anders klopt de massabalans niet.
                        old_C = self.segments.C[np.ix_(rows, cols)]
                        vol_r = self.segments.volume[rows][:, np.newaxis]
                        dm = np.zeros(self.n_species, dtype=np.float64)
                        dm[cols] = ((C_set[active] - old_C) * vol_r).sum(axis=0)
                        self._tracker.record_injection_mass(dm)
                    self.segments.C[np.ix_(rows, cols)] = C_set[active]
                    continue

            # Geen bestaand segment: maak nieuw segment aan.
            # Volume = flow [m³/s] × tijdstap [s] → correcte eenheid [m³].
            # _last_dt wordt bijgehouden door step(); vóór de eerste step()-aanroep
            # is _last_dt=1.0 (conservatieve standaard).
            vol = max(flow[pipe_idx] * self._last_dt, 1e-9)
            C_new = np.zeros(self.n_species, dtype=np.float64)
            C_new[active] = C_set[active]
            self.segments.add(pipe=pipe_idx, x=0.0, volume=vol, C_vector=C_new)
            self._sync_geochem_new_segment()
            if self._tracker:
                self._tracker.record_injection(C_new, vol)

    # ── Tijdstap ──────────────────────────────────────────────────────────────

    def step(
        self,
        dt:             float,
        decay_k,
        merge_interval: int   = 10,
        merge_tol:      float = 1e-6,
        check_cfl:      bool  = False,
        merge_span:     float | None = 1.0,
        merge_rel_tol:  float = 0.0,
    ) -> np.ndarray:
        """
        Voer één kwaliteitstijdstap uit.

        Volgorde
        --------
        1. CFL-check (optioneel)
        2. Bulk + wandverval (in-place)
        3. Advectie (in-place)
        4. Exit-detectie
        5. Knoopmenging → node_C
        5b. Geochemisch evenwicht na menging (alleen bij geochem ≠ None)
        6. Tank-update (massabehoudend CSTR; instroom = aangekomen segmenten)
        7. Pipe-overgang / splitsing / eindknoop-verwijdering
           (overshoot voorbij het leidingeinde gaat als tijd mee)
        7b. Tankafgifte: tankwater gaat als nieuwe segmenten (x = 0) de
            uitgaande leidingen in
        8. Segment merging (elke merge_interval stappen)
        9. Massabalans registreren (indien track_mass=True)

        Parameters
        ----------
        dt             : tijdstap [s]
        decay_k        : (n_species,) bulkvervalconstanten [1/s]
        merge_interval : voer merging uit elke N stappen
        merge_tol      : concentratietolerantie voor merging
        merge_rel_tol  : relatieve concentratietolerantie voor merging (0 = uit)
        check_cfl      : als True, controleer CFL en waarschuw indien nodig
        merge_span     : samenvoegen alleen als de span van het paar blijft
                         onder merge_span·v·dt. None of 0 schakelt de
                         positie-voorwaarde niet in (oud, niet aanbevolen).

        Returns
        -------
        node_C : (node_count, n_species)
            Persistente knoopconcentratie: bij een aankomst de (gemengde)
            aankomstconcentratie, anders de laatst bekende waarde. Zie
            ``node_arrived`` voor de aankomstvlag van deze stap.
        """
        from .lta     import (combined_decay_multi, build_combined_exp,
                             advect, exit_detect, node_mixing_multi)
        from .merging import merge_segments

        flow, velocity = self._get_hydraulics()
        decay_k = np.asarray(decay_k, dtype=np.float64)
        if self._mixed_mask is not None:
            self._mixed_prepare()     # vóór alle index-gebruik: segmenten kunnen verdwijnen
        n = self.segments.n
        self._last_dt = float(dt)   # bewaar voor gebruik in booster_inject()

        if check_cfl:
            self.check_stability(dt, decay_k, warn=True)
        self._auto_inject_reservoirs(dt)
        n = self.segments.n

        # Zonder segmenten én zonder tanks valt er niets te doen. Mét tanks moet
        # de stap doorlopen: een tank kan een leeg net opnieuw vullen.
        if n == 0 and len(self._tank_nodes) == 0 and not self._has_external_inflow():
            return self._report_nodes(
                np.zeros((self.node_count, self.n_species), dtype=np.float64),
                np.zeros(self.node_count, dtype=np.float64),
            )

        # 1+2. Bulk + wandverval gecombineerd in één pass
        # combined_exp[p,s] = exp(-(k_bulk[s] + k_wall[p,s]) * dt)
        # Gecached; alleen herbouwd als dt of hydraulica verandert.
        # Massa vóór verval (incl. tussentijdse injecties) voor exacte boekhouding.
        if self._tracker:
            _mass_pre = self.segments.C[:n] * self.segments.volume[:n, np.newaxis]

        if self._geochem is not None:
            pipe_diam = np.sqrt(4 * self.pipe_area / np.pi)
            _, velocity = self._get_hydraulics()
            self._geochem.apply_geochemistry(
                self.segments, dt,
                pipe_diam=pipe_diam,
                pipe_vel=velocity,
            )
        else:
<<<<<<< HEAD
            need_rebuild = (
                self._combined_exp is None or
                self._combined_exp_dt != dt or
                self._combined_exp_k is None or
                not np.allclose(self._combined_exp_k, decay_k)
            )
            if need_rebuild:
=======
            if self._combined_exp_needs_rebuild(dt, decay_k):
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d
                self._combined_exp = build_combined_exp(
                    decay_k, self._k_wall_vol, dt, len(self.pipe_ids)
                )
                self._combined_exp_dt = dt
                self._combined_exp_k = decay_k.copy()
            combined_decay_multi(
                self.segments.C, self.segments.pipe,
                self._combined_exp, n=n,
            )

        decay_loss = None
        if self._tracker:
            loss_seg = _mass_pre - self.segments.C[:n] * self.segments.volume[:n, np.newaxis]
            if self._geochem is not None:
                # Netto-verandering door geochemie: alles onder 'bulk' boeken.
                decay_loss = (loss_seg.sum(axis=0), np.zeros(self.n_species))
            else:
                kw = (self._k_wall_vol[self.segments.pipe[:n]]
                      if self._k_wall_vol is not None else 0.0)
                k_tot = decay_k[np.newaxis, :] + kw
                with np.errstate(divide='ignore', invalid='ignore'):
                    share_b = np.where(k_tot > 0.0, decay_k[np.newaxis, :] / k_tot, 0.0)
                bulk = (loss_seg * share_b).sum(axis=0)
                decay_loss = (bulk, loss_seg.sum(axis=0) - bulk)

        # 3. Advectie
        advect(self.segments.x, self.segments.pipe, velocity, dt, n=n)

        # 3b. Lekkage: proportioneel volume-verlies per segment
        # Elk segment verlaat een fractie _leakage_fraction van zijn volume
        # per tijdstap, overeenkomend met het debietverlies door emitters/lekkage.
        # De concentratie blijft constant (geconserveerd mengmodel): het segment
        # krimpt, maar er stroomt geen extern water in. Bij zuigslag (negatief
        # druk) zou concentratie stijgen — dat is buiten het bereik van dit model.
        if self._leakage_fraction > 0.0:
            # Effectieve lekfractie per tijdstap: f_lek = Q_lek/Q * dt/T_verblijf
            # Benadering: proportioneel aan verblijftijd in het segment
            # volume_nieuw = volume_oud * exp(-lambda_lek * dt)
            # lambda_lek = leakage_fraction / verblijftijd_leiding
            # verblijftijd ≈ pipe_length[pipe] / velocity[pipe]
            safe_vel = np.maximum(velocity[self.segments.pipe[:n]], 1e-6)
            t_verblijf = self.pipe_length[self.segments.pipe[:n]] / safe_vel
            lam_lek    = self._leakage_fraction / np.maximum(t_verblijf, 1.0)
            if self._tracker:
                _v_old = self.segments.volume[:n].copy()
            self.segments.volume[:n] *= np.exp(-lam_lek * dt)
            if self._tracker:
                dV = _v_old - self.segments.volume[:n]
                self._tracker.record_leakage(
                    (self.segments.C[:n] * dV[:, np.newaxis]).sum(axis=0)
                )

        self._ensure_skip_cache(dt)

        # 4. Exit-detectie — in-place in pre-allocated buffer
        exit_detect(
            self.segments.x, self.segments.pipe, self.pipe_length,
            self._buf_exit_mask, n=n,
        )
        exit_mask = self._buf_exit_mask
        # Stilstaande links (gesloten afsluiter, dode tak) laten niets door:
        # zonder dit verlaat een segment een lengte-0 element (x=0 >= L=0)
        # óók als het gesloten is.
        exit_mask[:n] &= velocity[self.segments.pipe[:n]] > self._V_EPS

        # 5. Knoopmenging: volumegewogen (zie _mix_nodes) met externe instroom.
        #    Een knoop met meerdere bronnen (of externe instroom) geeft het
        #    uitgaande water één gemengde concentratie; een doorgaande knoop
        #    met één bron houdt de Lagrangiaanse segmentconcentraties.
        ext_V = self._external_volume(dt)
        node_C, node_V_tot, mixed_nodes = self._mix_nodes(
            exit_mask, n, ext_V, flow, dt,
        )
        if self._mixed_mask is not None:
            self._mixed_update(flow, node_C, dt)
        if self._geochem is None:
            self._apply_mixed_outflow(exit_mask, n, node_C, mixed_nodes)

        # 5b. Geochemisch evenwicht na knoopmenging (alleen bij GeochemSolver).
        #
        # Lineaire concentratiemenging (stap 5) is exact voor conservatieve
        # stoffen maar een benadering voor pH en het koolzuursysteem: het
        # mengsel van twee waters is pas in evenwicht na PHREEQC-berekening.
        # apply_mixing() corrigeert dit in-place voor knopen met debiet > 0,
        # analoog aan Victoria's pp.mix_solutions() bij elke uitvraag.
        if self._geochem is not None:
            # Totaal inkomend debiet per knoop — gebruik pre-allocated buffer
            node_flow = self._buf_node_flow
            node_flow[:] = 0.0
            np.add.at(node_flow, self.pipe_end, np.abs(flow))

            if hasattr(self._geochem, 'sol_ids'):
                # PhreeqSolutionMode (of vergelijkbare oplossingnummer-modus):
                # apply_mixing() doet NIETS zonder node_inflows (zie docstring
                # PhreeqSolutionMode.apply_mixing) — die moeten we hier zelf
                # opbouwen uit de PHREEQC-oplossingnummers van de segmenten
                # die deze stap een knoop bereiken.
                node_inflows = self._build_node_inflows(exit_mask, flow, n)
                self._geochem.apply_mixing(
                    node_C, node_flow, dt, node_inflows=node_inflows,
                )
            else:
                self._geochem.apply_mixing(node_C, node_flow, dt)

        # 6. Tank-update
        if len(self._tank_nodes) > 0:
            node_C = self._update_tanks(node_C, flow, decay_k, dt,
                                    exit_mask=exit_mask, n=n)

        # 7. Pipe-overgang
        exited_C, exited_V = self._handle_exits(exit_mask, flow)

        # 7a. Externe instroom verlaat de knoop als nieuwe segmenten (x = 0)
        #     in de uitgaande leidingen, met de gemengde knoopconcentratie.
        self._release_external(flow, ext_V, node_C)

        # 7b. Tanks geven water af aan hun uitgaande leidingen
        self._release_tanks(flow, dt)

        # 8. Merging — alleen als het samengevoegde segment kort genoeg blijft:
        #    span ≤ merge_span·v·dt (punt 2). Zo blijft de aankomsttijd binnen
        #    ~één tijdstap van de werkelijke.
        self._step_counter += 1
        if self._step_counter % merge_interval == 0:
            if self._geochem is not None and hasattr(self._geochem, 'sol_ids'):
                # merge_segments() herschikt/compacteert store-rijen volledig
                # (lexsort + reductie) zonder enige sol_ids-remap-hook. Voor
                # een oplossingnummer-modus zoals PhreeqSolutionMode zou dit
                # sol_ids stilzwijgend aan de verkeerde segmenten koppelen.
                # Overslaan tot merge_segments zelf een parallelle-array-hook
                # krijgt.
                pass
            else:
                span = None
                if merge_span is not None and merge_span > 0.0:
                    span = merge_span * velocity * dt
                merge_segments(self.segments, tol=merge_tol,
                              n_pipes=len(self.pipe_ids), max_span=span,
                              rel_tol=merge_rel_tol)

        # 9. Massabalans
        if self._tracker:
            self._tracker.record_step(
                self.segments, decay_k, dt,
                self._k_wall_vol, exited_C, exited_V,
                decay_loss=decay_loss,
            )

        return self._report_nodes(node_C, node_V_tot)

    # ── Massabalansrapport ────────────────────────────────────────────────────

    def mass_balance(self) -> dict | None:
        """
        Geeft het massabalansrapport terug (alleen als track_mass=True).
        """
        if not self._tracker:
            return None
        tank_mass = None
        if len(self._tank_nodes) and len(self._tank_V) == len(self._tank_nodes):
            tank_mass = (self._tank_C * self._tank_V[:, np.newaxis]).sum(axis=0)
        return self._tracker.report(self.segments, tank_mass=tank_mass)

    # ── Tank-model ────────────────────────────────────────────────────────────

    def _detect_tank_nodes(self) -> np.ndarray:
        """Geeft 0-based indices van tankknopen terug."""
        tank_uids = [n.uid for n in self.hyd.net.tanks]
        return np.array(
            [self.node_index[uid] for uid in tank_uids if uid in self.node_index],
            dtype=np.int32,
        )

    def _on_store_resize(self, new_capacity: int) -> None:
        """
        Wordt aangeroepen door SegmentStore._resize() als de capaciteit verdubbelt.
        Schaalt alle exit- en werkbuffers mee zodat ze nooit te klein zijn.
        """
        # De buffers worden gevuld vóórdat een split-lus segmenten toevoegt
        # (en dus een resize kan triggeren): de bestaande inhoud moet mee,
        # anders gaan de exit-concentraties/volumes/mask van deze stap verloren.
        def _grow(old, shape, dtype):
            new = np.zeros(shape, dtype=dtype)
            if old is not None:
                m = min(len(old), len(new))
                new[:m] = old[:m]
            return new
        ns = self.n_species
        self._buf_exit_C    = _grow(getattr(self, '_buf_exit_C', None),    (new_capacity, ns), np.float64)
        self._buf_exit_V    = _grow(getattr(self, '_buf_exit_V', None),    new_capacity,       np.float64)
        self._buf_wC        = _grow(getattr(self, '_buf_wC', None),        (new_capacity, ns), np.float64)
        self._buf_exit_mask = _grow(getattr(self, '_buf_exit_mask', None), new_capacity,       np.bool_)

    def _rebuild_tank_pipe_cache(self) -> None:
        """
        Herbouw cache van inkomende/uitgaande leidingen per tank.

        Wordt aangeroepen bij __init__ en na elke update_hydraulics(), omdat
        flow reversals de topologie (pipe_start/pipe_end) kunnen wijzigen.
        Voorkomt O(n_tanks × n_pipes) np.where()-aanroepen per tijdstap.
        """
        self._tank_in_pipes  = []
        self._tank_out_pipes = []
        for tank_node in self._tank_nodes:
            self._tank_in_pipes.append(
                np.where(self.pipe_end   == tank_node)[0]
            )
            self._tank_out_pipes.append(
                np.where(self.pipe_start == tank_node)[0]
            )

    def _get_tank_volumes(self) -> np.ndarray:
        """Huidige tankvolumes [m³] via epynet.

        EPYnetDTD noemt deze property `tank_volume` (EN_TANKVOLUME, actueel
        berekend volume), niet `volume` zoals de oude epynet — `n.volume`
        bestaat niet op EPYnetDTD's Tank en gaf hier voorheen altijd de
        stille 1000.0-fallback, voor élke tank, ongeacht het echte volume.
        """
        from . import units as _u
        # EPANET geeft tankvolume in ft³ bij US-flow-eenheden (net als lengte in ft).
        get_units = getattr(self.hyd, '_get_flow_units', None)
        units = get_units() if callable(get_units) else 'CMH'
        volumes = []
        for n in self.hyd.net.tanks:
            try:
                volumes.append(float(_u.volume_to_m3(float(n.tank_volume), units)))
            except Exception as exc:
                # Geen stille standaardwaarde: een verkeerd tankvolume geeft
                # een verkeerde menging zonder dat de gebruiker het ziet.
                raise RuntimeError(
                    f"Tankvolume van tank '{getattr(n, 'uid', '?')}' kon niet "
                    f"worden uitgelezen of omgerekend ({type(exc).__name__}: {exc})."
                ) from exc
        return np.array(volumes, dtype=np.float64) if volumes else np.array([])

    def _update_tanks(
        self,
        node_C:  np.ndarray,   # (node_count, n_species)
        flow:    np.ndarray,
        decay_k: np.ndarray,
        dt:      float,
        exit_mask: np.ndarray | None = None,
        n:         int | None = None,
    ) -> np.ndarray:
        """
        Massabehoudend CSTR-tankmodel (impliciet Euler) met tankmassa als toestand.

        Per stap, per tank (V_start = huidig tankvolume):
            V_in    = Σ volume van de segmenten die de tank deze stap bereiken
            C_in    = volumegewogen concentratie van die segmenten
            V_after = V_start + V_in − Q_out·dt          (≥ 1e-9)
            M_new   = (M_old + V_in·C_in) / (1 + Q_out·dt/V_after + k·dt)
            C_new   = M_new / V_after

        Dit is exact massabehoudend: M_new − M_old = V_in·C_in − Q_out·dt·C_new
        − k·M_new·dt, met Q_out·dt·C_new de massa die via _release_tanks() als
        nieuw segment de uitgaande leidingen in gaat. Instroom komt uit de
        daadwerkelijk aangekomen segmenten (niet uit het hydraulische debiet:
        node_C[tank] was 0 op stappen zonder aankomst → verdunning uit het niets).

        Zonder exit_mask wordt de oude (niet massabehoudende) variant gebruikt:
        tank_step_implicit() met hydraulisch Q_in, zonder afgifte aan leidingen.
        """
        from .lta import tank_step_implicit

        n_tanks = len(self._tank_nodes)
        self._tank_release = []
        if n_tanks == 0:
            return node_C

        if exit_mask is None:
            Q_in  = self._buf_Q_in[:n_tanks];  Q_in[:]  = 0.0
            Q_out = self._buf_Q_out[:n_tanks]; Q_out[:] = 0.0
            C_in  = self._buf_C_in[:n_tanks];  C_in[:]  = 0.0
            for ti, tank_node in enumerate(self._tank_nodes):
                ip, op = self._tank_in_pipes[ti], self._tank_out_pipes[ti]
                Q_in[ti]  = flow[ip].sum() if len(ip) else 0.0
                Q_out[ti] = flow[op].sum() if len(op) else 0.0
                C_in[ti]  = node_C[tank_node]
            if len(self._tank_V) == n_tanks:
                tank_step_implicit(self._tank_C, Q_in, C_in, Q_out, self._tank_V, decay_k, dt)
                node_C[self._tank_nodes] = self._tank_C
            return node_C

        if len(self._tank_V) != n_tanks:
            return node_C

        n_act = self.segments.n if n is None else n
        em    = exit_mask[:n_act]
        exit_idx = np.where(em)[0] if em.any() else np.empty(0, dtype=np.int64)
        exit_end = self.pipe_end[self.segments.pipe[exit_idx]] if exit_idx.size else exit_idx

        V_in  = np.zeros(n_tanks)
        C_in  = np.zeros((n_tanks, self.n_species))
        Q_out = np.zeros(n_tanks)
        for ti, tank_node in enumerate(self._tank_nodes):
            op = self._tank_out_pipes[ti]
            Q_out[ti] = flow[op].sum() if len(op) else 0.0
            idx = exit_idx[exit_end == tank_node] if exit_idx.size else exit_idx
            if idx.size:
                vols = self.segments.volume[idx]
                v_tot = float(vols.sum())
                if v_tot > 0.0:
                    V_in[ti] = v_tot
                    C_in[ti] = (self.segments.C[idx] * vols[:, np.newaxis]).sum(axis=0) / v_tot

        V_start = np.asarray(self._tank_V, dtype=np.float64)
        M_old   = self._tank_C * V_start[:, np.newaxis]
        V_after = np.maximum(V_start + V_in - Q_out * dt, 1e-9)
        denom   = 1.0 + (Q_out * dt / V_after)[:, np.newaxis] + decay_k[np.newaxis, :] * dt
        M_new   = (M_old + V_in[:, np.newaxis] * C_in) / denom
        C_new   = M_new / V_after[:, np.newaxis]

        if self._tracker:
            self._tracker.record_tank_decay(
                (decay_k[np.newaxis, :] * M_new * dt).sum(axis=0)
            )

        self._tank_C[:] = C_new
        self._tank_V    = V_after
        node_C[self._tank_nodes] = self._tank_C

        # Afgifte wordt na _handle_exits toegevoegd (segmentarrays mogen tijdens
        # de exitverwerking niet groeien: exit_mask hoort bij de oude n).
        for ti in range(n_tanks):
            for p in self._tank_out_pipes[ti]:
                vol = float(flow[p]) * dt
                if vol > 0.0:
                    self._tank_release.append((int(p), vol, C_new[ti].copy()))
        return node_C

    def _release_tanks(self, flow: np.ndarray, dt: float) -> None:
        """Voeg het door _update_tanks berekende tankwater toe als segmenten (x = 0)."""
        for p, vol, C in getattr(self, '_tank_release', ()):
            self.segments.add(pipe=p, x=0.0, volume=vol, C_vector=C)
            self._sync_geochem_new_segment()
        self._tank_release = []

    # ── Oplossingnummer-modus (PhreeqSolutionMode) ─────────────────────────────

    def _build_node_inflows(
        self,
        exit_mask: np.ndarray,
        flow:      np.ndarray,
        n:         int,
    ) -> List[List[Tuple[int, float]]]:
        """
        Bouw per knoop een lijst van (oplossingnummer, debiet)-paren voor
        de segmenten die deze stap de knoop bereiken.

        Analoog aan ``node_mixing_multi`` / ``_node_mixing_kernel`` in
        ``lta.py``, maar met PHREEQC-oplossingnummers (``self._geochem.
        sol_ids``) in plaats van concentratievectoren als last te mengen
        grootheid — nodig omdat ``PhreeqSolutionMode`` geen gebruik maakt
        van de C-matrix.  Het debiet wordt hier ongenormaliseerd doorgegeven
        (net als ``flow`` in de kernel); ``PhreeqSolutionMode.apply_mixing``
        normaliseert zelf naar fracties per knoop.

        Parameters
        ----------
        exit_mask : (capacity,) bool — welke segmenten deze stap exiten
        flow      : (n_pipes,) debiet per leiding
        n         : aantal actieve segmenten

        Returns
        -------
        list van lengte ``node_count``; element ``nd`` is een lijst van
        ``(sol_num, debiet)``-paren voor knoop ``nd`` (leeg als er geen
        segment deze stap die knoop bereikt).
        """
        node_inflows: List[List[Tuple[int, float]]] = [
            [] for _ in range(self.node_count)
        ]
        em = exit_mask[:n]
        if not em.any():
            return node_inflows

        sol_ids  = self._geochem.sol_ids
        exit_idx = np.where(em)[0]
        for i in exit_idx:
            p  = int(self.segments.pipe[i])
            nd = int(self.pipe_end[p])
            w  = float(flow[p])
            node_inflows[nd].append((int(sol_ids[i]), w))
        return node_inflows

    def _phreeq_mixed_number(self, node_idx: int) -> int | None:
        """Oplossingnummer na apply_mixing voor deze knoop, of None."""
        node_sols = getattr(self._geochem, '_node_solutions', None)
        if not node_sols:
            return None
        return node_sols.get(int(node_idx))

    def _assign_phreeq_mixed(self, seg_indices, node_ids) -> None:
        """
        Koppel doorgaande exit-segmenten aan de gemengde knoopoplossing.

        Elk segment krijgt een eigen PHREEQC-kopie: delen van één nummer
        zou latere kinetiek de oplossing van een ander segment muteren.
        Ontbreekt een mengresultaat, dan blijft het bestaande sol_ids-nummer.
        """
        geo = self._geochem
        for seg_i, nd in zip(seg_indices, node_ids):
            mixed = self._phreeq_mixed_number(int(nd))
            if mixed is None:
                continue
            geo.sol_ids[int(seg_i)] = geo.copy_solution(mixed)

    # ── Exit-verwerking ───────────────────────────────────────────────────────

    # ── Externe instroom en knoopmenging ──────────────────────────────────────

    def set_source_quality(self, node_uid: str, C_vector) -> None:
        """
        Stel de concentratie in van water dat via een externe instroom
        (negatieve vraag, bv. J7 in pulse4.inp) een knoop binnenkomt.

        Standaard 0 (zonder SOURCE in het .inp). Hetzelfde als EPANET met een
        bron van die concentratie. Geldt voor alle stappen tot de volgende
        aanroep.
        """
        idx = self.node_index.get(node_uid)
        if idx is None:
            raise KeyError(f"set_source_quality: onbekende knoopnaam {node_uid!r}")
        if self._ext_C is None:
            self._ext_C = np.zeros((self.node_count, self.n_species), dtype=np.float64)
        self._ext_C[idx] = np.asarray(C_vector, dtype=np.float64)

    def _has_external_inflow(self) -> bool:
        """True als er een knoop met externe instroom en uitgaande leiding is."""
        if self._node_demand is None or self._node_type is None:
            return False
        ok = (self._node_type >= 1) & ~self._is_tank_node
        return bool(np.any(ok & (self._node_demand < 0.0)))

    def _external_volume(self, dt: float) -> np.ndarray:
        """
        Volume [m³] dat deze stap via externe instroom elke knoop binnenkomt.

        Alleen knopen met een uitgaande leiding (anders blijft het water
        nergens) en geen tanks. EPANET negatieve vraag = instroom.
        """
        nc = self.node_count
        ext = np.zeros(nc, dtype=np.float64)
        if self._node_demand is None or self._node_type is None:
            return ext
        if len(self._node_demand) != nc:
            raise ValueError(
                f"knoopvraag heeft lengte {len(self._node_demand)}, "
                f"verwacht {nc}"
            )
        q_in = np.maximum(-self._node_demand, 0.0)
        ok = (self._node_type >= 1) & ~self._is_tank_node
        ext[ok] = q_in[ok] * dt
        return ext

    def set_mixed_pipes(self, mask) -> None:
        """
        Behandel de leidingen in ``mask`` als volledig gemengde volumes (zoals
        een tank) in plaats van als Lagrangiaanse segmenten.

        Bedoeld voor leidingen die vaak omkeren (bijna stilstaand heen-en-weer
        water), waar segmenten niet betrouwbaar zijn. De concentratie C_p van de
        leiding verandert met het doorstromende water: C_p ← C_up + (C_p−C_up)·e^(−|Q|Δt/V).
        Uitgaand water heeft C_p. Mask heeft lengte n_pipes; None schakelt dit uit.
        """
        if mask is None:
            self._mixed_mask = None
            return
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (len(self.pipe_ids),):
            raise ValueError(f"mask moet lengte {len(self.pipe_ids)} hebben")
        self._mixed_mask = mask
        self._mixed_C = None
        self._mixed_V = None

    def _mixed_prepare(self) -> None:
        """Zet gemengde leidingen klaar: segmenten erin samenvoegen tot C_p."""
        mask = self._mixed_mask
        npipe, ns = len(self.pipe_ids), self.n_species
        V_pipe = self.pipe_length * self.pipe_area
        if self._mixed_C is None:
            # Eerste keer: C_p = volumegewogen gemiddelde van de segmenten die er nu zijn,
            # anders de uitlaatconcentratie.
            self._mixed_C = self._pipe_out_C.copy() if self._pipe_out_C is not None \
                else np.zeros((npipe, ns))
            self._mixed_V = V_pipe.copy()
        n = self.segments.n
        if n > 0:
            sel = np.where(mask[self.segments.pipe[:n]])[0]
            if sel.size:
                p = self.segments.pipe[sel]
                vs = np.zeros(npipe); np.add.at(vs, p, self.segments.volume[sel])
                vcs = np.zeros((npipe, ns))
                np.add.at(vcs, p, self.segments.C[sel] * self.segments.volume[sel][:, np.newaxis])
                hit = vs > 0.0
                self._mixed_C[hit] = (self._mixed_V[hit, None] * self._mixed_C[hit] + vcs[hit]) \
                    / (self._mixed_V[hit, None] + vs[hit, None])
                self._mixed_V[hit] += vs[hit]
                self.segments.remove(sel.tolist())
        self._pipe_out_C[mask] = self._mixed_C[mask]

    def _mixed_update(self, flow: np.ndarray, node_C: np.ndarray, dt: float) -> None:
        """Verversing van gemengde leidingen met het water van de bovenstroomse knoop."""
        mask = self._mixed_mask
        idx = np.where(mask)[0]
        if idx.size == 0:
            return
        up = self.pipe_start[idx]
        Cup = node_C[up]
        a = np.abs(flow[idx]) * dt / np.maximum(self._mixed_V[idx], 1e-12)
        dec = np.exp(-a)[:, np.newaxis]
        self._mixed_C[idx] = Cup + (self._mixed_C[idx] - Cup) * dec
        self._pipe_out_C[idx] = self._mixed_C[idx]

    def _mix_nodes(
        self,
        exit_mask: np.ndarray,
        n:         int,
        ext_V:     np.ndarray,
        flow:      np.ndarray,
        dt:        float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Knoopmenging: continu over de doorstroming, zoals EPANET.

        Per leiding bewaren we de concentratie aan het uitlopende einde
        (``_pipe_out_C``): de laatst aangekomen segmenten. Het gewicht van een
        inkomende leiding in een knoop is de hydraulische doorstroom Q·Δt,
        niet het aantal of het volume van de aankomende segmenten. Zo telt elke
        leiding één keer mee, ook als er in deze stap geen segment aankomt.
        Externe instroom weegt met Q_ext·Δt en concentratie C_ext.

        Returns
        -------
        node_C      : (node_count, n_species)  gemengde concentratie per knoop
        node_V_tot  : (node_count,)            gewicht (m³) per knoop: Σ Q·Δt + V_ext
        mixed_nodes : (node_count,) bool       knopen die menging nodig hebben:
                                               ≥2 inkomende leidingen of externe instroom
        """
        nc, ns = self.node_count, self.n_species
        npipe  = len(self.pipe_ids)
        if self._pipe_out_C is None or self._pipe_out_C.shape != (npipe, ns):
            self._pipe_out_C = np.zeros((npipe, ns), dtype=np.float64)

        # 1. Aankomsten van deze stap: nieuwe uitlaat-concentratie per leiding
        idx = np.where(exit_mask[:n])[0]
        pipe_idx = self.segments.pipe[idx]
        V_arr = self.segments.volume[idx]
        pV  = np.zeros(npipe, dtype=np.float64)
        pVC = np.zeros((npipe, ns), dtype=np.float64)
        np.add.at(pV, pipe_idx, V_arr)
        np.add.at(pVC, pipe_idx, self.segments.C[idx] * V_arr[:, np.newaxis])
        has = pV > 0.0
        self._pipe_out_C[has] = pVC[has] / pV[has, np.newaxis]

        # Welke knopen kregen deze stap daadwerkelijk aankomsten (voor de vlag)
        arr = np.zeros(nc, dtype=np.int32)
        np.add.at(arr, self.pipe_end[pipe_idx], 1)
        self._arrived_nodes = arr > 0

        # 2. Continu gewicht per inkomende leiding (Q·Δt) en knoopmenging
        active_in = flow[:npipe] > self._FLOW_EPS
        W_pipe = np.where(active_in, flow[:npipe] * dt, 0.0)
        node_W  = np.zeros(nc, dtype=np.float64)
        node_WC = np.zeros((nc, ns), dtype=np.float64)
        np.add.at(node_W, self.pipe_end, W_pipe)
        np.add.at(node_WC, self.pipe_end, W_pipe[:, np.newaxis] * self._pipe_out_C)

        ext_C = self._ext_C if self._ext_C is not None else np.zeros((nc, ns))
        node_W_tot  = node_W + ext_V
        node_WC_tot = node_WC + ext_V[:, np.newaxis] * ext_C

        node_C = (self._node_C_hold.copy() if self._node_C_hold is not None
                  else np.zeros((nc, ns), dtype=np.float64))
        pos = node_W_tot > 0.0
        node_C[pos] = node_WC_tot[pos] / node_W_tot[pos, np.newaxis]

        n_in = np.zeros(nc, dtype=np.int32)
        np.add.at(n_in, self.pipe_end[active_in], 1)
        mixed_nodes = (n_in >= 2) | (ext_V > 0.0)
        return node_C, node_W_tot, mixed_nodes

    def _apply_mixed_outflow(
        self,
        exit_mask:   np.ndarray,
        n:           int,
        node_C:      np.ndarray,
        mixed_nodes: np.ndarray,
    ) -> None:
        """
        Zet de concentratie van segmenten die een gemengde knoop bereiken op
        de gemengde knoopconcentratie. Zo heeft het uitgaande water één
        concentratie. Volume blijft ongemoeid, dus massa blijft behouden.
        """
        idx = np.where(exit_mask[:n])[0]
        if idx.size == 0:
            return
        nd = self.pipe_end[self.segments.pipe[idx]]
        sel = idx[mixed_nodes[nd]]
        if sel.size:
            self.segments.C[sel] = node_C[self.pipe_end[self.segments.pipe[sel]]]

    def _release_external(
        self,
        flow:  np.ndarray,
        ext_V: np.ndarray,
        node_C: np.ndarray,
    ) -> None:
        """
        Laat het externe water de uitgaande leidingen in als nieuwe segmenten
        (x = 0), verdeeld naar debiet, met de gemengde knoopconcentratie.
        Dezelfde volumelogica als _release_tanks.
        """
        nd_list = np.where(ext_V > 0.0)[0]
        if nd_list.size == 0:
            return
        mass_in = np.zeros(self.n_species, dtype=np.float64)
        for nd in nd_list:
            outs = self._node_active_out.get(int(nd), [])
            if not outs:
                continue
            w = np.array([flow[p] for p in outs], dtype=np.float64)
            if w.sum() <= 0.0:
                w = np.ones(len(outs), dtype=np.float64)
            w = w / w.sum()
            C_nd = node_C[nd]
            for p, wp in zip(outs, w):
                tp = int(self._skip_target[p]) if self._skip_target is not None else int(p)
                self.segments.add(pipe=tp, x=0.0, volume=float(ext_V[nd] * wp),
                                  C_vector=C_nd)
                self._sync_geochem_new_segment()
            if self._ext_C is not None:
                mass_in += ext_V[nd] * self._ext_C[nd]
        if self._tracker and np.any(mass_in != 0.0):
            self._tracker.record_injection_mass(mass_in)

    def _report_nodes(self, node_C: np.ndarray, node_V_tot: np.ndarray) -> np.ndarray:
        """
        Persistente knoopconcentratie (punt 5): bij aankomst de gemengde
        waarde, anders de laatst bekende. Zet ``node_arrived`` voor deze stap.
        """
        nc, ns = self.node_count, self.n_species
        if self._node_C_hold is None:
            self._node_C_hold = np.zeros((nc, ns), dtype=np.float64)
        flowing = node_V_tot > 0.0
        self._node_C_hold[flowing] = node_C[flowing]
        # Reservoir zonder bron: rapporteer de vaste beginkwaliteit (EPANET), ook als
        # het reservoir zelf water ontvangt.
        if self._res_sources and self._res_init_C is not None:
            for _name, _ni, _strength, *_rest in self._res_sources:
                if _strength is None:
                    self._node_C_hold[_ni] = self._res_init_C[_ni]
        arr = getattr(self, '_arrived_nodes', None)
        self.node_arrived = arr.copy() if arr is not None and arr.shape == (nc,) \
            else np.zeros(nc, dtype=bool)
        return self._node_C_hold.copy()

    def _rebuild_node_routing_cache(self) -> None:
        """
        Bouw gecachede routing-arrays per knoop.

        _node_type[nd] : 0 = eindknoop, 1 = doorgaand, >=2 = splitsing
        _node_out0[nd] : index van de uitgangsleiding met het hoogste debiet
                         (of 0 voor eindknopen — nooit gebruikt).

        Aanroepen bij __init__ en na elke update_hydraulics().
        Maakt het dominante pad in _handle_exits (doorgaand + eindknoop)
        volledig vectoriseerbaar zonder Python-loop.
        """
        nc = self.node_count
        # Alleen uitgaande leidingen met debiet tellen mee: een gesloten
        # afsluiter (of dode tak) is geen uitgang en maakt van een doorgaande
        # knoop geen splitsing.  Heeft een knoop uitsluitend stilstaande
        # uitgangen, dan blijven alle uitgangen staan (segment wacht daar).
        flow, _ = self._get_hydraulics()
        active: dict[int, list] = {}
        for nd in range(nc):
            outs = self._node_outpipes.get(nd, [])
            act = [p for p in outs if flow[p] > self._FLOW_EPS]
            active[nd] = act if act else list(outs)
        self._node_active_out = active
        node_type = np.array([len(active[i]) for i in range(nc)], dtype=np.int32)
        node_out0 = np.array(
            [max(active[i], key=lambda p: flow[p]) if active[i] else 0
             for i in range(nc)],
            dtype=np.int32,
        )
        # Tanks absorberen aankomende segmenten (water gaat de tank in, zie
        # _update_tanks) en geven water af via _release_tanks(): geen doorgaande
        # of splitsende routing, ook niet als er uitgaande leidingen zijn.
        is_tank = np.zeros(nc, dtype=bool)
        if len(self._tank_nodes):
            is_tank[self._tank_nodes] = True
            node_type[self._tank_nodes] = 0
        self._is_tank_node = is_tank
        self._node_type = node_type
        self._node_out0 = node_out0

    # ── Overshoot-carry bij leidingovergang ───────────────────────────────────

    _MAX_INIT_SEGMENTS = 50   # max beginsegmenten per leiding (set_initial_quality)
    _CARRY_EPS = 1e-12   # houdt x strikt < L zodat een segment niet in dezelfde stap opnieuw "exit"
    _V_EPS     = 1e-12   # [m/s] snelheid waaronder een link als stilstaand geldt
    _FLOW_EPS  = 1e-7    # [m³/s] onder dit debiet is een link stilstaand (EPANET zet gesloten links op ~3e-8)

    def _ensure_skip_cache(self, dt: float) -> None:
        """
        Bepaal welke afsluiters in deze tijdstap worden overgeslagen en waar
        een segment dat zo'n afsluiter zou binnengaan terechtkomt.

        Overslaan als de afsluiter open is (status ≠ CLOSED, debiet > _FLOW_EPS)
        én korter is dan de afstand per tijdstap (L < v·dt).  Alleen over een
        doorgaande knoop (precies één actieve uitgang); bij een splitsing of
        eindknoop stroomafwaarts blijft de afsluiter gewoon staan.
        _skip_target[p] = eerste niet-overgeslagen leiding vanaf p, en
        _skip_time[p]   = de tijd [s] die het segment onderweg in de
        overgeslagen afsluiters doorbrengt (L/v, 0 bij lengte 0).
        """
        if self._skip_dt == dt:
            return
        n_pipes = len(self.pipe_ids)
        tgt = np.arange(n_pipes, dtype=np.int32)
        tsk = np.zeros(n_pipes, dtype=np.float64)
        skip = np.zeros(n_pipes, dtype=bool)
        if self._skip_short_valves and self._valve_mask.any():
            flow, vel = self._get_hydraulics()
            self._valve_closed = self.hyd.get_closed_mask()
            skip = (
                self._valve_mask
                & ~self._valve_closed
                & (flow > self._FLOW_EPS)
                & (self.pipe_length < vel * dt)
            )
            if skip.any():
                with np.errstate(divide='ignore', invalid='ignore'):
                    t_own = np.where(skip, self.pipe_length / np.maximum(vel, 1e-12), 0.0)
                for _ in range(64):   # ketens van opeenvolgende afsluiters
                    cur  = tgt
                    sel  = skip[cur] & (self._node_type[self.pipe_end[cur]] == 1) \
                           & ~self._is_tank_node[self.pipe_end[cur]]
                    if not sel.any():
                        break
                    tsk[sel] += t_own[cur[sel]]
                    tgt[sel]  = self._node_out0[self.pipe_end[cur[sel]]]
        self._skip_mask   = skip
        self._skip_target = tgt
        self._skip_time   = tsk
        self._skip_dt     = dt

    def skip_info(self) -> dict:
        """Diagnostiek: aantal afsluiters, gesloten en (in de laatste stap) overgeslagen."""
        return {
            'n_valves':  int(self._valve_mask.sum()),
            'n_closed':  int((self._valve_mask & self._valve_closed).sum()),
            'n_skipped': int(self._skip_mask.sum()),
            'dt':        self._skip_dt,
        }

    def _excess_time(self, x: float, pipe: int, vel: np.ndarray) -> float:
        """Tijd [s] dat segment al voorbij het einde van `pipe` is (x > L)."""
        v = float(vel[pipe])
        if v <= 1e-12:
            return 0.0
        return max(x - float(self.pipe_length[pipe]), 0.0) / v

    def _x_after_carry(self, t_excess: float, new_pipe: int, vel: np.ndarray) -> float:
        """Startpositie in `new_pipe` na `t_excess` seconden stroming daarin."""
        L = float(self.pipe_length[new_pipe])
        x = t_excess * float(vel[new_pipe])
        return min(x, L * (1.0 - self._CARRY_EPS))

    def _carry_over_x(self, x, old_pipes, new_pipes, vel, t_skip=None) -> np.ndarray:
        """Vectorized variant van _excess_time + _x_after_carry.

        t_skip : optioneel (n,) tijd [s] die al in overgeslagen afsluiters is
                 doorgebracht en van de overshoot-tijd wordt afgetrokken.
        """
        v_old  = vel[old_pipes]
        excess = np.maximum(x - self.pipe_length[old_pipes], 0.0)
        t_ex   = np.where(v_old > 1e-12, excess / np.maximum(v_old, 1e-12), 0.0)
        if t_skip is not None:
            t_ex = np.maximum(t_ex - t_skip, 0.0)
        x_new  = t_ex * vel[new_pipes]
        return np.minimum(x_new, self.pipe_length[new_pipes] * (1.0 - self._CARRY_EPS))

    def _record_pipe_out_chem(self, pipes, seg_idx):
        """
        PhreeqSolutionMode: bewaar per pijp de chemie (pH, alkaliniteit, koolstof)
        van het laatste segment dat de pijp volledig heeft doorlopen. Dat is de
        uitlaat-chemie van die pijp, de tegenhanger van _pipe_out_C voor concentraties.
        Zonder geochemie is dit een no-op.
        """
        geo = self._geochem
        if geo is None or not hasattr(geo, 'sol_ids'):
            return
        if getattr(self, '_pipe_out_ph', None) is None:
            n = len(self.pipe_ids)
            self._pipe_out_ph  = np.full(n, np.nan)
            self._pipe_out_alk = np.full(n, np.nan)
            self._pipe_out_ct  = np.full(n, np.nan)
        for p, j in zip(np.asarray(pipes), np.asarray(seg_idx)):
            so = geo.get_solution(self.segments, int(j))
            alk = (so.species.get('HCO3-', 0) + 2 * so.species.get('CO3-2', 0)
                   + so.species.get('OH-', 0) - so.species.get('H+', 0)) * 1000.0
            self._pipe_out_ph[int(p)]  = so.pH
            self._pipe_out_alk[int(p)] = alk
            self._pipe_out_ct[int(p)]  = so.total_element('C')

    def _cascade_thru(self, thru_idx, old_p, first_p, vel):
        """
        Verplaats doorgaande segmenten over de resterende reistijd van deze stap.

        Na het verlaten van een pijp blijft er reistijd over. Die wordt op de
        volgende pijp gebruikt. Als de volgende pijp ook volledig wordt
        afgelegd en eindigt in een doorgaande knoop, gaat het segment door.
        Stopt bij een splitsing, een eindknoop of een pijp die niet volledig
        wordt afgelegd; daar blijft het segment aan het eind liggen en verlaat
        het de leiding in de volgende stap (zoals voorheen).
        """
        skip = self._skip_time[first_p]
        C_thru = self.segments.C[thru_idx].copy()   # concentratie per doorlopen segment
        v_old = vel[old_p]
        excess = np.maximum(self.segments.x[thru_idx] - self.pipe_length[old_p], 0.0)
        t_rem = np.where(v_old > 1e-12, excess / np.maximum(v_old, 1e-12), 0.0)
        t_rem = np.maximum(t_rem - skip, 0.0)

        cur = np.asarray(self._skip_target[first_p], dtype=np.int64).copy()
        out_p = cur.copy()
        out_x = np.zeros(len(cur), dtype=np.float64)
        active = np.arange(len(cur))
        for _ in range(256):
            if active.size == 0:
                break
            p = cur[active]
            L = self.pipe_length[p]
            v = vel[p]
            dist = np.where(v > 1e-12, t_rem[active] * v, 0.0)
            end_nd = self.pipe_end[p]
            through = (self._node_type[end_nd] == 1) & (~self._is_tank_node[end_nd])
            finish = (dist < L * (1.0 - self._CARRY_EPS)) | (~through) | (v <= 1e-12)
            fa = active[finish]
            out_p[fa] = p[finish]
            out_x[fa] = np.minimum(dist[finish], L[finish] * (1.0 - self._CARRY_EPS))
            # Water dat aan het eind van een pijp tot stilstand komt: de chemie van
            # die pijp is dat water (zie _record_pipe_out_chem).
            self._record_pipe_out_chem(out_p[fa], thru_idx[fa])
            ca = active[~finish]
            if ca.size == 0:
                break
            # volledige pijp doorlopen: het water is er doorheen gegaan, dus de
            # uitlaatconcentratie van die pijp is het concentratie van dit segment
            # (anders zien knopen onderweg de aankomst niet).
            if self._pipe_out_C is not None:
                self._pipe_out_C[cur[ca]] = C_thru[ca]
            self._record_pipe_out_chem(cur[ca], thru_idx[ca])
            # verbruik de reistijd en ga naar de volgende pijp
            t_rem[ca] = t_rem[ca] - self.pipe_length[cur[ca]] / vel[cur[ca]]
            nxt_first = self._node_out0[self.pipe_end[cur[ca]]]
            cur[ca] = self._skip_target[nxt_first]
            active = ca
        else:
            # Veiligheidsgrens bereikt: resterende segmenten aan het eind van hun pijp
            if active.size:
                pa = cur[active]
                out_p[active] = pa
                out_x[active] = self.pipe_length[pa] * (1.0 - self._CARRY_EPS)
        return out_p, out_x

    def _handle_exits(
        self,
        exit_mask: np.ndarray,
        flow:      np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Verwerk segmenten die het einde van hun leiding bereiken.

        Strategie (drie paden, oplopend in complexiteit):

        1. Doorgaand (node_type == 1) — vectorized
           Segment krijgt nieuwe pipe = node_out0[node] en x = 0.
           Geen Python-loop, geen allocaties.

        2. Eindknoop (node_type == 0) — vectorized kopieer + swap-with-last
           Concentraties/volumes naar exit-buffers; segmenten verwijderd via
           bulk swap-with-last in één vectorized pass.

        3. Splitsing (node_type >= 2) — kleine Python-loop
           Alleen voor splitsingspunten; typisch < 5% van alle exits.
           Segment wordt proportioneel opgesplist naar debiet.

        Returns
        -------
        exited_C : view op _buf_exit_C[:n_exit]  (geen nieuwe array)
        exited_V : view op _buf_exit_V[:n_exit]  (geen nieuwe array)
        """
        n = self.segments.n
        em = exit_mask[:n]

        if not em.any():
            return self._buf_exit_C[:0], self._buf_exit_V[:0]

        exit_idx   = np.where(em)[0]                              # (n_exit,)
        exit_pipes = self.segments.pipe[exit_idx]                 # (n_exit,)
        exit_nodes = self.pipe_end[exit_pipes]                    # (n_exit,)
        nd_type    = self._node_type[exit_nodes]                  # (n_exit,)

        # ── 1. Doorgaand: volledig vectorized ─────────────────────────────
        _, vel = self._get_hydraulics()
        thru_sel = nd_type == 1
        if thru_sel.any():
            thru_idx = exit_idx[thru_sel]
            thru_nd  = exit_nodes[thru_sel]
<<<<<<< HEAD
            self.segments.pipe[thru_idx] = self._node_out0[thru_nd]
            self.segments.x[thru_idx]    = 0.0
=======
            old_p    = self.segments.pipe[thru_idx].copy()
            first_p  = self._node_out0[thru_nd]
            new_p    = self._skip_target[first_p]       # open korte afsluiters overslaan
            # Overgeslagen leiding krijgt geen segmenten meer: haar uitlaat-
            # concentratie is het water dat er nu doorheen gaat (anders blijft de
            # knoop erachter op een verouderde waarde staan).
            skipped_p = new_p != first_p
            if skipped_p.any() and self._pipe_out_C is not None:
                self._pipe_out_C[first_p[skipped_p]] = self.segments.C[thru_idx[skipped_p]]
            # Een segment kan in één tijdstap over meerdere korte doorgaande pijpen
            # gaan; de rest-reistijd wordt door de keten gevolgd (zie _cascade_thru).
            new_p, x_new = self._cascade_thru(thru_idx, old_p, first_p, vel)
            self.segments.pipe[thru_idx] = new_p
            self.segments.x[thru_idx]    = x_new
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d
            if self._geochem is not None and hasattr(self._geochem, 'sol_ids'):
                self._assign_phreeq_mixed(thru_idx, thru_nd)

        # ── 2. Eindknopen: vectorized kopieer, dan bulk-remove ─────────────
        end_sel = nd_type == 0
        n_end   = int(end_sel.sum())
        n_out   = 0
        if n_end > 0:
            end_idx = exit_idx[end_sel]
            # Kopieer naar exit-buffers in één bulk-operatie (geen loop).
            # Aankomsten bij een tank verlaten het systeem niet: die massa zit
            # daarna in de tank (zie _update_tanks) en telt dus niet als outflow.
            out_sel = ~self._is_tank_node[exit_nodes[end_sel]]
            out_idx = end_idx[out_sel]
            n_out   = int(out_idx.size)
            self._buf_exit_C[:n_out] = self.segments.C[out_idx]
            self._buf_exit_V[:n_out] = self.segments.volume[out_idx]
            # Bij PhreeqSolutionMode (of vergelijkbare modus) moet sol_ids
            # exact dezelfde swap-with-last ondergaan als store.remove()
            # doet — anders raakt sol_ids[i] gekoppeld aan het verkeerde
            # segment. Moet vóór store.remove() gebeuren: die heeft de
            # segmentcount ván vóór de removal nodig.
            if self._geochem is not None and hasattr(self._geochem, 'sol_ids'):
                self._geochem.mirror_removal(end_idx.tolist(), self.segments.n)
            # Verwijder via swap-with-last
            self.segments.remove(end_idx.tolist())

        # ── 3. Splitsingen: kleine Python-loop ────────────────────────────
        split_sel = nd_type >= 2
        orig_C    = np.empty(self.n_species, dtype=np.float64)
        if split_sel.any():
            # Herbereken exit_idx na mogelijke remove() hierboven:
            # remove() gebruikt swap-with-last waardoor indices kunnen zijn
            # verschoven. Herdetecteer splitsingen op basis van x >= L.
            n2     = self.segments.n
            em2    = (self.segments.x[:n2] >= self.pipe_length[self.segments.pipe[:n2]]) \
                     & (vel[self.segments.pipe[:n2]] > self._V_EPS)
            s_idx  = np.where(em2)[0]
            s_pipe = self.segments.pipe[s_idx]
            s_nd   = self.pipe_end[s_pipe]
            s_type = self._node_type[s_nd]
            split_only = s_idx[s_type >= 2]

            use_phreeq = self._geochem is not None and hasattr(self._geochem, 'sol_ids')
            for seg_i in split_only:
                nd    = int(self.pipe_end[self.segments.pipe[seg_i]])
                out   = self._node_active_out.get(nd, [])
                if not out:
                    continue
                out_s    = sorted(out, key=lambda p: -flow[p])
                tot_flow = sum(flow[p] for p in out_s)
                orig_vol = self.segments.volume[seg_i]
                orig_C[:] = self.segments.C[seg_i]
<<<<<<< HEAD
                orig_sol = int(self._geochem.sol_ids[seg_i]) if use_phreeq else None

                self.segments.pipe[seg_i]   = out_s[0]
                self.segments.x[seg_i]      = 0.0
=======
                if self._pipe_out_C is not None:
                    for _p in out:
                        if int(self._skip_target[_p]) != _p:
                            self._pipe_out_C[_p] = orig_C
                if use_phreeq:
                    self._record_pipe_out_chem(out, [seg_i] * len(out))
                orig_sol = int(self._geochem.sol_ids[seg_i]) if use_phreeq else None
                old_p    = int(self.segments.pipe[seg_i])
                t_excess = self._excess_time(float(self.segments.x[seg_i]), old_p, vel)

                tgt0 = int(self._skip_target[out_s[0]])
                self.segments.pipe[seg_i]   = tgt0
                self.segments.x[seg_i]      = self._x_after_carry(
                    max(t_excess - self._skip_time[out_s[0]], 0.0), tgt0, vel)
>>>>>>> 6802704785e71ed64a409922e041a6e05c6e869d
                if tot_flow <= 0.0:
                    # Geen uitgaande debiet: behoud massabalans door het segment
                    # op het dominante uitgangspad te houden zonder te splitsen.
                    self.segments.volume[seg_i] = orig_vol
                    if use_phreeq and orig_sol is not None:
                        self._geochem.sol_ids[seg_i] = self._geochem.copy_solution(orig_sol)
                    continue
                self.segments.volume[seg_i] = orig_vol * flow[out_s[0]] / tot_flow
                for p in out_s[1:]:
                    tp = int(self._skip_target[p])
                    self.segments.add(
                        pipe=tp, x=self._x_after_carry(
                            max(t_excess - self._skip_time[p], 0.0), tp, vel),
                        volume=orig_vol * flow[p] / tot_flow,
                        C_vector=orig_C,
                    )
                    if use_phreeq and orig_sol is not None:
                        copy_num = self._geochem.copy_solution(orig_sol)
                        self._geochem.on_segment_added(self.segments.n - 1, copy_num)

        return self._buf_exit_C[:n_out], self._buf_exit_V[:n_out]

    # ── Hulpfuncties ──────────────────────────────────────────────────────────

    @staticmethod
    def _broadcast_kwall(k_wall, n_pipes, n_species):
        if k_wall.ndim == 1:
            return np.broadcast_to(k_wall[np.newaxis, :], (n_pipes, n_species)).copy()
        if k_wall.shape == (n_pipes, n_species):
            return k_wall
        raise ValueError(
            f"k_wall shape {k_wall.shape} past niet op "
            f"(n_pipes={n_pipes}, n_species={n_species})"
        )

    def geochem_report(self, C_vec: np.ndarray) -> dict:
        """
        Geeft geochemisch rapport terug voor een concentratieprofiel.

        Bevat Langelier Saturation Index en verzadigingsindices voor
        relevante mineralen. Vereist dat geochem is geconfigureerd.

        Parameters
        ----------
        C_vec : (n_species,) concentratieprofiel [zelfde eenheden als SpeciesMap]

        Returns
        -------
        dict met 'lsi', 'saturation_indices', 'pH', 'species'
        """
        if self._geochem is None:
            return {'error': 'Geen GeochemSolver geconfigureerd (geochem=None)'}

        lsi = self._geochem.langelier_index(C_vec)
        si  = self._geochem.saturation_indices(C_vec)

        # Lees pH terug als aanwezig in SpeciesMap
        ph_idx = self._geochem.smap.ph_index
        pH = float(C_vec[ph_idx]) if ph_idx is not None else None

        return {
            'lsi':               lsi,
            'saturation_indices': si,
            'pH':                pH,
            'species': {
                name: float(C_vec[i])
                for i, name in enumerate(self._geochem.smap.species_names)
            },
        }

    def __repr__(self) -> str:
        geo_str = f" geochem={self._geochem.smap.species_names}" if self._geochem else ''
        T_str = f" T={self._temperature}°C" if self._temperature is not None else ""
        L_str = f" lek={self._leakage_fraction:.0%}" if self._leakage_fraction > 0 else ""
        return (
            f"<NzingaFlowSolver "
            f"pipes={len(self.pipe_ids)} nodes={self.node_count} "
            f"n_species={self.n_species} "
            f"tanks={len(self._tank_nodes)} "
            f"wall={'yes' if self._k_wall_vol is not None else 'no'}"
            f"{geo_str}{T_str}{L_str} "
            f"segments={self.segments.n}>"
        )
