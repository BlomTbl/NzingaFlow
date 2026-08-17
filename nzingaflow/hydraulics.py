# nzingaflow/hydraulics.py
"""
Koppeling tussen epynet (EPANET via Python) en NzingaFlow.

Werkt native tegen de kale EPYnetDTD-`epynet.Network` — geen compat-laag
(`epynet.compat` bestaat niet en wordt hier ook niet nagebouwd). Alle
eenheidsconversies en EPANET-property-enums komen uit `nzingaflow.units`
(één bron, zie die module's docstring voor de bugs die dat oploste).

Laagverdeling
──────────────
- `Topology`        : onveranderlijke netwerktopologie (pipe_start/end/
                       length/area, node_names) — eenmalig berekend bij
                       laden, blijft geldig zolang het netwerk niet van
                       vorm verandert (leidingen toevoegen/verwijderen).
- `HydraulicState`   : snapshot van de laatste solve() — flow, velocity,
                       reversed. Wordt in één moeite herbouwd ná elke
                       solve (één property-read per leiding), zodat
                       downstream code (solver.py, eps.py) niet
                       herhaaldelijk live epynet-properties aanspreekt.
- `_NoSaveHydraulicSolver` : subklasse van epynet.solver.HydraulicSolver;
                       schrijft nooit naar .hyd (i.p.v. het EPYnetDTD-
                       default EN_SAVE), onderdrukt de trial-by-trial
                       statusregels (EN_setstatusreport(0)), en hergebruikt de
                       EN_openH()-sessie over meerdere solve()-aanroepen
                       heen (zie solve()/close() hieronder). Gebruikt
                       EN_INITFLOW (niet EN_NOSAVE!) voor de per-stap
                       EN_initH()-aanroep — zie de klassedocstring voor
                       waarom dat het enige correcte is zodra EN_openH()
                       wordt hergebruikt.
- `HydraulicModel`   : dunne façade met dezelfde publieke API als voorheen
                       (solve(), get_topology(), get_topology_with_reversal(),
                       get_hydraulic_state(), .net, summary()) — solver.py
                       en eps.py blijven ongewijzigd werken.

Waarom EPYnetDTD een andere aanpak vraagt dan de oude epynet
────────────────────────────────────────────────────────────
1.  `Network.solve(simtime)` bestaat niet meer. Hydraulica oplossen gaat nu
    via `epynet.solver.HydraulicSolver.solve_time_step(pattern_start_time)`.
    `simtime` (oud) en `pattern_start_time` (nieuw) sturen beide
    EN_PATTERNSTART (code 4) aan — 1-op-1 uitwisselbaar.

2.  `net.nodes`/`.links`/`.pipes`/`.pumps`/`.valves`/`.tanks`/`.reservoirs`
    zijn kale generators: geen `len()`, geen `in`, geen dict-toegang — wél
    al getypeerd via NodeFactory/LinkFactory, dus `isinstance(node, Tank)`
    werkt (i.p.v. een membership-check op een uid-verzameling).

3.  `net.ep.ENxxx(...)` bestaat niet meer — Network ÍS zelf het EPANET2-
    toolkitobject; elke ENxxx-functie heet nu EN_xxx (`net.EN_getflowunits()`).

4.  `EN_initH()` zonder argument valt terug op EN_SAVE (schrijft naar
    .hyd) — ongewenst bij een EPS-loop met veel stappen. Zie
    `_NoSaveHydraulicSolver` (en de klassedocstring daar voor waarom dat
    concreet EN_INITFLOW is, niet het voor de hand liggende EN_NOSAVE).

5.  Property-toegang is "live", zonder caching: `pipe.flow`/`pipe.velocity`
    is elke keer een aparte EN_getlinkvalue-toolkitcall. `HydraulicState`
    vangt dit in één keer op na elke solve(), i.p.v. dat downstream code
    (solver.py) herhaaldelijk los `lnk.flow` opvraagt.

Invarianten die niet zijn veranderd
─────────────────────────────────────
- node.uid / link.uid : unieke EPANET-naam (str)
- pipe.from_node / pipe.to_node : Node-objecten
- Standaard alleen Pipe-objecten in de topologie; pumps/valves via
  include_pumps/include_valves
- Negatief debiet → reversed_mask → pipe_start/pipe_end omwisselen
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from epynet import Network
from epynet.solver import HydraulicSolver

from . import units as u


# ═══════════════════════════════════════════════════════════════════════════
# Onveranderlijke/snapshot-datastructuren
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, eq=False)
class Topology:
    """Onveranderlijke netwerktopologie, eenmalig berekend bij laden."""
    pipe_start: np.ndarray    # (n_pipes,) int32  0-based knoopindex beginpunt
    pipe_end: np.ndarray      # (n_pipes,) int32  0-based knoopindex eindpunt
    pipe_length: np.ndarray   # (n_pipes,) float64  [m]
    pipe_area: np.ndarray     # (n_pipes,) float64  [m²]
    node_count: int
    pipe_ids: list
    node_names: list

    def as_tuple(self) -> tuple:
        """Zelfde 7-tuple als de oude get_topology()-return — voor
        achterwaartse compatibiliteit met solver.py/eps.py."""
        return (
            self.pipe_start, self.pipe_end, self.pipe_length, self.pipe_area,
            self.node_count, self.pipe_ids, self.node_names,
        )


@dataclass(frozen=True, eq=False)
class HydraulicState:
    """Snapshot van de laatst opgeloste hydraulica (na solve())."""
    flow: np.ndarray       # (n_pipes,) float64  m³/s, altijd ≥ 0
    velocity: np.ndarray   # (n_pipes,) float64  m/s,  altijd ≥ 0
    reversed: np.ndarray   # (n_pipes,) bool     True = omgekeerd t.o.v. EPANET-definitie
    simtime: int

    def as_tuple(self) -> tuple:
        return (self.flow, self.velocity, self.reversed)


# ═══════════════════════════════════════════════════════════════════════════
# Solver
# ═══════════════════════════════════════════════════════════════════════════

class _NoSaveHydraulicSolver(HydraulicSolver):
    """
    `HydraulicSolver`-variant met twee optimalisaties t.o.v. het EPYnetDTD-
    default gedrag, beide geverifieerd tegen "vers HydraulicModel per
    simtime" met np.allclose() (zie tests/test_epynet_networks.py
    ::TestSessionReuse):

    1.  Geen .hyd-disk-I/O per stap, plus EN_setstatusreport(0): de
        trial-by-trial statusregels van de hydraulische solver gaan
        rechtstreeks via de C-bibliotheek naar de OS-file-descriptor (niet
        met Python's contextlib.redirect_stdout te onderscheppen).
        Zelf gemeten (30 EN_initH+EN_runH-cycli, 20k-leiding grid-net):
        het verschil viel binnen de meetruis (~2%, niet consistent
        positief) — in deze omgeving dus geen aantoonbare winst op
        zichzelf. Blijft niettemin aan: kost niets, en de write-vermijding
        is onafhankelijk daarvan al nuttig (zie punt 3, .hyd-schrijven).

    2.  EN_openH() (matrixopbouw, O(netwerkgrootte)) wordt maar één keer
        per sessie aangeroepen, via `open_once()` — niet bij elke stap,
        in tegenstelling tot de EPYnetDTD-basisklasse
        (`Solver.solve_time_step()` doet initialise()
        [=EN_openH()+EN_initH()] + run() + close() bij ÉLKE aanroep).
        Zelf gemeten op een 100×100 grid-net (10.001 knopen, 19.801
        leidingen): EN_openH() kost ~2,8s per aanroep — dat eenmalig
        i.p.v. per stap doen scheelt dus evenredig met het aantal stappen.

    3.  Link-lijst gecached in HydraulicModel._get_links() (zie aldaar) —
        `list(net.pipes)` opnieuw opbouwen kostte zelf gemeten ~70ms per
        aanroep op hetzelfde 20k-leiding grid-net; hergebruik van de
        gecachede lijst kost ~0,04ms.

        Gecombineerd effect (zelfde grid-net, 5 EPS-stappen,
        HydraulicModel end-to-end via solve()+get_hydraulic_state()):
        ~837ms/stap met sessie- en cache-hergebruik tegen ~3088ms/stap
        voor "elke stap een vers HydraulicModel" (~73% sneller) — zelf
        gemeten, niet uit een eerdere sessie overgenomen.

        **Kritieke correctheidsval, empirisch gevonden**: de voor de hand
        liggende keuze voor de per-stap EN_initH()-flag is EN_NOSAVE (0)
        — dat is wat de oude epynet gebruikte, en wat je zou verwachten
        als "geen wijziging" t.o.v. het eerdere gedrag. Dat bleek **fout**
        zodra EN_openH() over meerdere stappen wordt hergebruikt: EN_NOSAVE
        betekent letterlijk "sla niet op; initialiseer de debieten NIET
        opnieuw" — d.w.z. de solver hergebruikt de laatst-geconvergeerde
        debieten van de vórige stap als startpunt (warm start) voor de
        Newton-Raphson-iteratie van de volgende stap. Bij een vers
        `HydraulicModel` per simtime is er niets om te hergebruiken (cold
        start, EN_openH() net aangeroepen) — bij een hergebruikte sessie
        wél. Beide convergeren binnen de hydraulische tolerantie naar
        "hetzelfde" antwoord, maar niet bit-identiek: het verschil bleek
        in de orde van 1e-8 relatief bij een testnetwerk met tank+patroon
        — klein, maar een reëel, meetbaar verschil in uitkomst, niet ruis.
        **EN_INITFLOW (10) is de juiste flag**: die dwingt een cold start
        af bij élke stap (debieten wél opnieuw geïnitialiseerd), ook al
        blijft EN_openH() zelf open. Tankniveaus/klok worden sowieso al bij
        élke EN_initH()-aanroep teruggezet naar de begincondities uit het
        .inp-bestand, ongeacht deze flag — dát deel van "steady-state-
        opname bij patroontijd X vanaf begincondities" stond dus niet ter
        discussie; alleen de iteratieve-solver-startwaarde wel.

    Sessiebeheer: `open_once()` is idempotent (mag na close() opnieuw).
    `close()` is eveneens idempotent en ongevaarlijk zonder voorafgaande
    open_once()-aanroep. Gebruik via HydraulicModel.solve()/.close(), niet
    rechtstreeks.
    """

    def __init__(self, network) -> None:
        super().__init__(network)
        self._opened = False

    def open_once(self) -> None:
        """Open de hydraulische solver-sessie (EN_openH). Idempotent."""
        if self._opened:
            return
        self.network.EN_setstatusreport(0)
        self.network.EN_openH()
        self._opened = True

    def initialise(self) -> None:
        """EN_initH alleen — EN_openH gebeurt via open_once(), dat hier
        (idempotent) wordt aangeroepen zodat deze klasse ook los van
        HydraulicModel bruikbaar blijft (bv. via het geërfde
        solve_time_step()).

        EN_INITFLOW (niet EN_NOSAVE!) — zie klasse-docstring hierboven
        voor waarom dat bij sessie-hergebruik het enige correcte is."""
        self.open_once()
        self.network.EN_initH(u.EN_InitHydOption.EN_INITFLOW)

    def close(self) -> None:
        """EN_closeH — sluit de sessie. Ongevaarlijk als er nooit geopend
        is, of als er al gesloten is (geen dubbele EN_closeH-aanroep)."""
        if self._opened:
            self.network.EN_closeH()
            self._opened = False

    def solve_step(self, pattern_start_time: int = 0) -> None:
        """
        Los één EPS-stap op, met hergebruik van de EN_openH()-sessie.

        Net als `Solver.solve_time_step()` (de EPYnetDTD-basisimplementatie)
        wordt EN_PATTERNSTART tijdelijk gezet en na afloop teruggezet naar
        de waarde van vóór de aanroep — maar in tegenstelling tot die
        basisimplementatie wordt `close()` hier NIET aangeroepen: de sessie
        blijft open voor de volgende stap. Roep `close()` expliciet aan
        (via HydraulicModel.close()) als de sessie echt beëindigd moet
        worden — bv. aan het einde van een EPS-run.
        """
        previous = self.network.EN_gettimeparam(u.EN_TimeParameter.EN_PATTERNSTART)
        self.network.EN_settimeparam(u.EN_TimeParameter.EN_PATTERNSTART, pattern_start_time)

        self.initialise()
        self.run()

        self.network.EN_settimeparam(u.EN_TimeParameter.EN_PATTERNSTART, previous)


# ═══════════════════════════════════════════════════════════════════════════
# Façade
# ═══════════════════════════════════════════════════════════════════════════

class HydraulicModel:
    """
    Koppeling tussen epynet (EPANET via Python) en NzingaFlow.

    Verantwoordelijkheden
    ──────────────────────
    - Laad het EPANET-netwerk en zet topologie om naar numpy-arrays.
    - Los hydraulica op voor een steady-state tijdstip (EPS: één stap per keer).
    - Lever flow [m³/s] en velocity [m/s] per leiding in SI-eenheden.
    - Corrigeer voor stroomrichting (negatief debiet → wissel start/end).

    Parameters
    ──────────────────────
    inp_path : str
        Pad naar het EPANET .inp invoerbestand.
    include_pumps : bool, default False
        Als True worden pompen ook als 'leidingen' opgenomen in de topologie.
        Standaard False: pompen worden overgeslagen (geen dispersie, geen verval).
    include_valves : bool, default False
        Als True worden afsluiters (PRV, PSV, TCV, FCV, GPV, PCV) ook als
        'leidingen' opgenomen in de topologie.  EPANET lost de hydraulica voor
        afsluiters altijd correct op; deze optie zorgt dat NzingaFlow de
        bijbehorende verblijftijd en stoftransport ook meeneemt.
        Standaard False voor achterwaartse compatibiliteit.

    Sessiebeheer
    ──────────────────────
    Roep `close()` aan wanneer je klaar bent met dit model (bv. aan het
    einde van een EPS-run), zodat de onderliggende EN_openH()-sessie netjes
    wordt afgesloten (EN_closeH). Een nieuwe `solve()`-aanroep na `close()`
    heropent de sessie automatisch. Ook bruikbaar als context manager:

        with HydraulicModel(path) as hm:
            hm.solve()
            ...
    """

    def __init__(self, inp_path: str, include_pumps: bool = False,
                 include_valves: bool = False):
        self.net              = Network(inp_path)
        self._solver          = _NoSaveHydraulicSolver(self.net)
        self._solved_for_simtime: int | None = None   # None = nog niet opgelost
        self._include_pumps   = include_pumps
        self._include_valves  = include_valves
        self._pipe_list       = None    # gecached na eerste _get_links()
        self._topology: Topology | None = None
        self._state: HydraulicState | None = None
        self._units           = None    # gecached na eerste _get_flow_units()

    def __enter__(self) -> "HydraulicModel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Hydraulische berekening
    # ═══════════════════════════════════════════════════════════════════════

    def solve(self, simtime: int = 0) -> None:
        """
        Los hydraulica op voor één tijdstip [s] (steady-state of EPS-stap).

        Parameters
        ──────────────────────
        simtime : int
            Simulatietijd in seconden (0 = beginconditie).

        Opmerking
        ──────────────────────
        Memoisatie: als het netwerk al opgelost is voor exact deze
        `simtime`, wordt er niet opnieuw gerekend.

        De EN_openH()-sessie blijft open tussen solve()-aanroepen (zie
        `_NoSaveHydraulicSolver`) — roep `close()` aan als je klaar bent.
        """
        if self._solved_for_simtime == simtime:
            return

        self._solver.solve_step(pattern_start_time=simtime)
        self._solved_for_simtime = simtime
        self._pipe_list = None    # invalideer link-cache na nieuwe oplossing
        self._state = self._read_hydraulic_state(simtime)

    def close(self) -> None:
        """Sluit de hydraulische solver-sessie (EN_closeH). Idempotent."""
        self._solver.close()

    # ═══════════════════════════════════════════════════════════════════════
    # Topologie
    # ═══════════════════════════════════════════════════════════════════════

    def get_topology(self) -> tuple:
        """
        Geeft netwerktopologie terug als numpy-arrays (gecached, zie Topology).

        De topologie is gebaseerd op de EPANET-definitierichting (van/naar knoop
        zoals opgegeven in het .inp bestand).  Voor de werkelijke stroomrichting
        na een hydraulische oplossing, zie get_topology_with_reversal().

        Returns
        ──────────────────────
        pipe_start  : (n_pipes,) int32    0-based knoopindex beginpunt
        pipe_end    : (n_pipes,) int32    0-based knoopindex eindpunt
        pipe_length : (n_pipes,) float64  lengte [m]
        pipe_area   : (n_pipes,) float64  dwarsdoorsnede [m²]
        node_count  : int                 totaal aantal knooppunten
        pipe_ids    : list[str]           EPANET leiding-namen
        node_names  : list[str]           knoopnamen in index-volgorde
        """
        if self._topology is not None:
            return self._topology.as_tuple()

        links    = self._get_links()
        units    = self._get_flow_units()

        # ── Knoopindex-tabel (0-based) ────────────────────────────────────────
        # node.index is 1-based (EPANET-conventie); wij willen 0-based.
        # De volgorde in self.net.nodes (kale generator) is de volgorde
        # van inlezen, die overeenkomt met de EPANET-interne indices.
        node_names  = [n.uid for n in self.net.nodes]
        node_index  = {uid: i for i, uid in enumerate(node_names)}
        node_count  = len(node_names)

        # ── Topologie-arrays ──────────────────────────────────────────────────
        pipe_start  = np.array(
            [node_index[lnk.from_node.uid] for lnk in links],
            dtype=np.int32,
        )
        pipe_end    = np.array(
            [node_index[lnk.to_node.uid] for lnk in links],
            dtype=np.int32,
        )
        # NB: epynet Valve-objecten hebben geen 'length' static_property
        # (EPANET kent geen lengte voor afsluiters: PRV/PSV/PBV/FCV/TCV/GPV).
        # getattr(..., 0.0) voorkomt een AttributeError zodra include_valves=True
        # en behandelt een valve in topologie-berekeningen als een lengteloos
        # verbindingselement (lengte 0 m).
        pipe_length = np.array(
            [getattr(lnk, 'length', 0.0) for lnk in links],
            dtype=np.float64,
        )

        # ── Diameter → m ───────────────────────────────────────────────────────
        # NB: epynet Pump-objecten hebben geen 'diameter' static_property
        # (EPANET kent geen diameter voor pompen). getattr(..., 0.0) voorkomt
        # een AttributeError zodra include_pumps=True en behandelt een pomp
        # in dwarsdoorsnede-afhankelijke berekeningen als nul-oppervlak
        # element (analoog aan de length-fallback voor valves).
        diam_raw   = np.array(
            [getattr(lnk, 'diameter', 0.0) for lnk in links],
            dtype=np.float64,
        )
        pipe_diam  = u.diameter_to_m(diam_raw, units)
        pipe_area  = np.pi * (pipe_diam / 2.0) ** 2

        pipe_ids   = [lnk.uid for lnk in links]

        self._topology = Topology(
            pipe_start=pipe_start, pipe_end=pipe_end,
            pipe_length=pipe_length, pipe_area=pipe_area,
            node_count=node_count, pipe_ids=pipe_ids, node_names=node_names,
        )
        return self._topology.as_tuple()

    def get_topology_with_reversal(self) -> tuple:
        """
        Geeft topologie terug waarbij pipe_start/pipe_end zijn omgewisseld
        voor leidingen met negatief debiet na solve().

        Flow reversal: EPANET rapporteert negatief debiet als de werkelijke
        stroomrichting omgekeerd is t.o.v. de .inp-definitierichting.
        Na omwisseling loopt de advectie altijd van start → end in de
        richting van de stroom.

        Let op: na solve() opnieuw aanroepen.

        Returns
        ──────────────────────
        Dezelfde tuple als get_topology(), maar pipe_start/pipe_end zijn
        gecorrigeerd voor de huidige hydraulische toestand.
        """
        (
            pipe_start_base, pipe_end_base,
            pipe_length, pipe_area,
            node_count, pipe_ids, node_names,
        ) = self.get_topology()

        _, _, reversed_mask = self.get_hydraulic_state()

        pipe_start = pipe_start_base.copy()
        pipe_end   = pipe_end_base.copy()

        if reversed_mask.any():
            pipe_start[reversed_mask] = pipe_end_base[reversed_mask]
            pipe_end[reversed_mask]   = pipe_start_base[reversed_mask]

        return (
            pipe_start, pipe_end, pipe_length, pipe_area,
            node_count, pipe_ids, node_names,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Hydraulische toestand (na solve)
    # ═══════════════════════════════════════════════════════════════════════

    def get_hydraulic_state(self) -> tuple:
        """
        Geeft debiet [m³/s], snelheid [m/s] en stroomrichting per leiding.

        Altijd in SI-eenheden, ongeacht de EPANET-eenheidsinstelling van het
        .inp bestand.  Negatieve debieten worden als absolute waarden
        teruggegeven; reversed geeft aan welke leidingen 'omgekeerd' stromen.

        Raises
        ──────────────────────
        RuntimeError als deze methode wordt aangeroepen vóórdat solve() ooit
        is uitgevoerd — voorheen gaven onopgeloste properties stilzwijgend
        onbetrouwbare (nul- of laatst-bekende) waarden terug.

        Returns
        ──────────────────────
        flow     : (n_pipes,) float64  m³/s, altijd ≥ 0
        velocity : (n_pipes,) float64  m/s,  altijd ≥ 0
        reversed : (n_pipes,) bool     True als werkelijke richting ≠ EPANET-definitie
        """
        if self._state is None:
            raise RuntimeError(
                "get_hydraulic_state() aangeroepen vóór solve() — er is nog "
                "geen hydraulische oplossing beschikbaar. Roep eerst "
                "HydraulicModel.solve(simtime=...) aan."
            )
        return self._state.as_tuple()

    def _read_hydraulic_state(self, simtime: int) -> HydraulicState:
        """Lees flow/velocity voor alle actieve links in één moeite in, ná
        een solve()-aanroep. Wordt gecachet in self._state; downstream code
        (solver.py) leest dus geen losse live epynet-properties meer per
        aanroep van get_hydraulic_state()."""
        links   = self._get_links()
        units   = self._get_flow_units()

        flow_raw = np.array([lnk.flow     for lnk in links], dtype=np.float64)
        vel_raw  = np.array([lnk.velocity for lnk in links], dtype=np.float64)

        flow_si  = u.flow_to_m3s(flow_raw, units)
        vel_si   = u.velocity_to_ms(vel_raw, units)

        reversed_mask = flow_si < 0.0

        return HydraulicState(
            flow=np.abs(flow_si), velocity=np.abs(vel_si),
            reversed=reversed_mask, simtime=simtime,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # Hulpfuncties (privé)
    # ═══════════════════════════════════════════════════════════════════════

    def _get_links(self) -> list:
        """
        Gecachede lijst van leiding-objecten.

        Standaard alleen Pipe-objecten.  Als include_pumps=True worden ook
        Pump-objecten meegenomen.  Als include_valves=True worden ook
        Valve-objecten (PRV, PSV, TCV, FCV, GPV, PCV) meegenomen.

        Deze lijst wordt gebouwd via `list(net.pipes)`/`list(net.pumps)`/
        `list(net.valves)` (elk al gefilterd op het juiste linktype door
        EPYnetDTD's eigen LinkFactory) en blijft geldig zolang de topologie
        van het netwerk niet verandert — de link-*objecten* zelf veranderen
        nooit tussen solve()-aanroepen, alleen hun live properties. Bij
        grote netwerken (10.000+ leidingen) scheelt dit cachen aanzienlijke
        tijd t.o.v. deze lijst bij elke solve() opnieuw opbouwen.
        """
        if self._pipe_list is None:
            pipes = list(self.net.pipes)
            if self._include_pumps:
                pipes = pipes + list(self.net.pumps)
            if self._include_valves:
                pipes = pipes + list(self.net.valves)
            self._pipe_list = pipes
        return self._pipe_list

    def _get_flow_units(self) -> str:
        """Lees de flow-eenheid uit het EPANET-project (gecached)."""
        if self._units is None:
            self._units = u.flow_units_label(self.net)
        return self._units

    # ═══════════════════════════════════════════════════════════════════════
    # Diagnostiek
    # ═══════════════════════════════════════════════════════════════════════

    def summary(self) -> str:
        """
        Geeft een korte samenvatting van het netwerk en de hydraulische toestand.

        Nuttig voor debuggen en verificatie van de eenheidsinstellingen.
        """
        links     = self._get_links()
        flow_units = self._get_flow_units()
        # net.nodes / .tanks / .reservoirs zijn kale generators in EPYnetDTD
        # (geen ObjectCollection meer) — dus expliciet naar list() voor len().
        # net.getNodeCount() is O(1) (EN_getcount) en dus sneller dan
        # len(list(net.nodes)) voor het totaal — voor tanks/reservoirs
        # bestaat geen losse O(1)-teller, dus die blijven list()-gebaseerd.
        n_nodes   = self.net.getNodeCount()
        n_pipes   = len(list(self.net.pipes))
        n_pumps   = len(list(self.net.pumps))
        n_valves  = len(list(self.net.valves))
        n_tanks   = len(list(self.net.tanks))
        n_res     = len(list(self.net.reservoirs))

        lines = [
            f"HydraulicModel — {self.net.inputfile or '(geen .inp)'}",
            f"  Flow-eenheden : {flow_units}",
            f"  Knopen        : {n_nodes}  "
            f"(junctions={n_nodes - n_tanks - n_res}, "
            f"tanks={n_tanks}, reservoirs={n_res})",
            f"  Leidingen     : {n_pipes} pipes, {n_pumps} pumps, {n_valves} valves",
            f"  Actieve links : {len(links)}  "
            f"(pipes"
            f"{'+pumps' if self._include_pumps else ''}"
            f"{'+valves' if self._include_valves else ''}"
            f")",
        ]

        if self._state is not None:
            try:
                flow, vel, rev = self.get_hydraulic_state()
                lines += [
                    f"  Hydraulica    : opgelost voor t={self._solved_for_simtime} s",
                    f"  Flow range    : {flow.min():.4f} – {flow.max():.4f} m³/s",
                    f"  Vel range     : {vel.min():.3f} – {vel.max():.3f} m/s",
                    f"  Reversals     : {rev.sum()} leidingen",
                ]
            except Exception as e:
                lines.append(f"  Hydraulica    : fout bij ophalen ({e})")
        else:
            lines.append("  Hydraulica    : nog niet opgelost")

        return "\n".join(lines)

    def __repr__(self) -> str:
        n_pipes = len(self._get_links())
        n_nodes = self.net.getNodeCount()
        units   = self._get_flow_units()
        return (
            f"<HydraulicModel pipes={n_pipes} nodes={n_nodes} "
            f"units={units} solved={self._state is not None}>"
        )
