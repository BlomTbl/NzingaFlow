# nzingaflow/hydraulics.py
"""
Koppeling tussen epynet (EPANET via Python) en NzingaFlow.

Wijzigingen t.o.v. vorige versie
──────────────────────────────────
Compatibel met de herziene epynet-codebase (2025):

1.  node.index  — is nu gecached via get_index() in BaseObject; blijft
                  1-based.  Aftrekken van 1 voor 0-based array-indexering
                  is nog steeds noodzakelijk.

2.  ObjectCollection.__iter__  — itereert over .values() (de objecten),
                  NIET over de sleutels.  list(net.pipes) geeft daardoor
                  een lijst van Pipe-objecten, precies wat we nodig hebben.
                  De oude code gebruikte list(net.pipes.values()) — beide
                  werken, maar list(net.pipes) is nu de idiomatische vorm.

3.  p.diameter  — ENgetlinkvalue(index, EN_DIAMETER=0) retourneert de
                  diameter in millimeter voor SI-eenheden (CMH, LPS, enz.)
                  en in inch voor US-eenheden (GPM, CFS, enz.).
                  De conversie naar meter hangt dus af van de eenheidsinstelling.
                  _get_diameter_to_m() handelt beide gevallen af.

4.  p.velocity  — wordt teruggegeven in m/s (SI) of ft/s (US) door EPANET.
                  get_hydraulic_state() converteert US-snelheid naar m/s.

5.  ENgetflowunits()  — beschikbaar via net.ep.ENgetflowunits(); retourneert
                  een int-code.  Dezelfde mapping als voorheen.

6.  weakref-netwerk  — node.network is een weakref.ref; gebruik altijd
                  node.network() om het Network-object te verkrijgen.
                  Niet relevant voor HydraulicModel zelf (we werken direct
                  op net.pipes / net.nodes), maar wel relevant als je ooit
                  node-methoden aanroept.

7.  solve() caches  — na network.solve() zijn link._values en node._values
                  leeggemaakt (reset() → _values.clear()).  Eigenschap-
                  toegang via p.flow / p.velocity werkt daarna correct omdat
                  get_property() opnieuw ENgetlinkvalue aanroept.

Invarianten die NIET zijn veranderd
─────────────────────────────────────
- node.uid / link.uid : unieke EPANET-naam (str)
- pipe.from_node / pipe.to_node : Node-objecten
- Standaard alleen Pipe-objecten in de topologie; pumps/valves via include_pumps/include_valves
- Negatief debiet → reversed_mask → pipe_start/pipe_end omwisselen
"""

from __future__ import annotations
import numpy as np


# ── Eenheidsafhankelijke conversiefactoren ────────────────────────────────────

# Flow-codes zoals geretourneerd door ENgetflowunits()
_FLOW_CODE_TO_LABEL = {
    0: 'CFS', 1: 'GPM', 2: 'MGD', 3: 'IMGD', 4: 'AFD',
    5: 'LPS', 6: 'LPM', 7: 'MLD', 8: 'CMH', 9: 'CMD',
}

# Conversiefactoren flow → m³/s
_FLOW_TO_CMS = {
    'CFS':  0.028317,
    'GPM':  6.30902e-5,
    'MGD':  0.043813,
    'IMGD': 0.052617,
    'AFD':  1.42764e-5,
    'LPS':  1e-3,
    'LPM':  1.0 / 60_000.0,
    'MLD':  1.0 / 86.4,
    'CMH':  1.0 / 3_600.0,
    'CMD':  1.0 / 86_400.0,
}

# US-eenheden gebruiken inch voor diameter en ft/s voor snelheid
_US_UNITS = {'CFS', 'GPM', 'MGD', 'IMGD', 'AFD'}

# Conversies
_INCH_TO_M  = 0.0254
_FT_S_TO_M_S = 0.3048


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
    """

    def __init__(self, inp_path: str, include_pumps: bool = False,
                 include_valves: bool = False):
        from epynet import Network
        self.net           = Network(inp_path)
        self._include_pumps  = include_pumps
        self._include_valves = include_valves
        self._pipe_list    = None   # gecached na eerste _get_links()
        self._topology     = None   # gecached na eerste get_topology()
        self._units        = None   # gecached na eerste _get_flow_units()

    # ═══════════════════════════════════════════════════════════════════════════
    # Hydraulische berekening
    # ═══════════════════════════════════════════════════════════════════════════

    def solve(self, simtime: int = 0) -> None:
        """
        Los hydraulica op voor één tijdstip [s] (steady-state of EPS-stap).

        Parameters
        ──────────────────────
        simtime : int
            Simulatietijd in seconden (0 = beginconditie).

        Opmerking
        ──────────────────────
        Na solve() maakt epynet intern link._values leeg via reset().
        Eigenschap-toegang (p.flow, p.velocity, p.diameter) werkt daarna
        correct via get_property() → ENgetlinkvalue().
        """
        self.net.solve(simtime=simtime)
        self._pipe_list = None   # invalideer link-cache na nieuwe oplossing

    # ═══════════════════════════════════════════════════════════════════════════
    # Topologie
    # ═══════════════════════════════════════════════════════════════════════════

    def get_topology(self) -> tuple:
        """
        Geeft netwerktopologie terug als numpy-arrays (gecached).

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
            return self._topology

        links    = self._get_links()
        units    = self._get_flow_units()

        # ── Knoopindex-tabel (0-based) ────────────────────────────────────────
        # node.index is 1-based (EPANET-conventie); wij willen 0-based.
        # De volgorde in self.net.nodes (ObjectCollection) is de volgorde
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
        # EPANET geeft diameter terug in:
        #   SI-eenheden (CMH, LPS, …) : mm
        #   US-eenheden (GPM, CFS, …) : inch
        # NB: epynet Pump-objecten hebben geen 'diameter' static_property
        # (EPANET kent geen diameter voor pompen). getattr(..., 0.0) voorkomt
        # een AttributeError zodra include_pumps=True en behandelt een pomp
        # in dwarsdoorsnede-afhankelijke berekeningen als lengteloos/nul-
        # oppervlak element (analoog aan de length-fallback voor valves).
        diam_raw   = np.array(
            [getattr(lnk, 'diameter', 0.0) for lnk in links],
            dtype=np.float64,
        )
        pipe_diam  = self._diameter_to_m(diam_raw, units)
        pipe_area  = np.pi * (pipe_diam / 2.0) ** 2

        pipe_ids   = [lnk.uid for lnk in links]

        self._topology = (
            pipe_start, pipe_end, pipe_length, pipe_area,
            node_count, pipe_ids, node_names,
        )
        return self._topology

    def get_topology_with_reversal(self) -> tuple:
        """
        Geeft topologie terug waarbij pipe_start/pipe_end zijn omgewisseld
        voor leidingen met negatief debiet na solve().

        Flow reversal: EPANET rapporteert negatief debiet als de werkelijke
        stroomrichting omgekeerd is t.o.v. de .inp-definitierichting.
        Na omwisseling loopt de advectie altijd van start → end in de
        richting van de stroom.

        Let op: na update_hydraulics() / solve() opnieuw aanroepen.

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

    # ═══════════════════════════════════════════════════════════════════════════
    # Hydraulische toestand (na solve)
    # ═══════════════════════════════════════════════════════════════════════════

    def get_hydraulic_state(self) -> tuple:
        """
        Geeft debiet [m³/s], snelheid [m/s] en stroomrichting per leiding.

        Altijd in SI-eenheden, ongeacht de EPANET-eenheidsinstelling van het
        .inp bestand.  Negatieve debieten worden als absolute waarden
        teruggegeven; reversed geeft aan welke leidingen 'omgekeerd' stromen.

        Returns
        ──────────────────────
        flow     : (n_pipes,) float64  m³/s, altijd ≥ 0
        velocity : (n_pipes,) float64  m/s,  altijd ≥ 0
        reversed : (n_pipes,) bool     True als werkelijke richting ≠ EPANET-definitie
        """
        links   = self._get_links()
        units   = self._get_flow_units()

        flow_raw = np.array([lnk.flow     for lnk in links], dtype=np.float64)
        vel_raw  = np.array([lnk.velocity for lnk in links], dtype=np.float64)

        # Converteernaar SI
        flow_si  = flow_raw * _FLOW_TO_CMS.get(units, 1.0 / 3600.0)
        vel_si   = vel_raw  * (_FT_S_TO_M_S if units in _US_UNITS else 1.0)

        reversed_mask = flow_si < 0.0

        return np.abs(flow_si), np.abs(vel_si), reversed_mask

    # ═══════════════════════════════════════════════════════════════════════════
    # Hulpfuncties (privé)
    # ═══════════════════════════════════════════════════════════════════════════

    def _get_links(self) -> list:
        """
        Gecachede lijst van leiding-objecten.

        Standaard alleen Pipe-objecten.  Als include_pumps=True worden ook
        Pump-objecten meegenomen.  Als include_valves=True worden ook
        Valve-objecten (PRV, PSV, TCV, FCV, GPV, PCV) meegenomen.

        Opmerking over ObjectCollection.__iter__:
            De nieuwe epynet itereert over .values() (de objecten zelf),
            zodat `for lnk in self.net.pipes` direct Pipe-objecten geeft.
            list(self.net.pipes) geeft dus een lijst van Pipe-objecten.
        """
        if self._pipe_list is None:
            pipes = list(self.net.pipes)   # list van Pipe-objecten
            if self._include_pumps:
                pipes = pipes + list(self.net.pumps)
            if self._include_valves:
                pipes = pipes + list(self.net.valves)
            self._pipe_list = pipes
        return self._pipe_list

    def _get_flow_units(self) -> str:
        """
        Lees de flow-eenheid uit het EPANET-project (gecached).

        Retourneert een string-label zoals 'CMH', 'LPS', 'GPM', enz.
        Fallback: 'CMH' (meest voorkomend in Nederlandse drinkwaternetwerken).
        """
        if self._units is not None:
            return self._units
        try:
            code = self.net.ep.ENgetflowunits()
            self._units = _FLOW_CODE_TO_LABEL.get(code, 'CMH')
        except Exception:
            self._units = 'CMH'
        return self._units

    @staticmethod
    def _diameter_to_m(diam_raw: np.ndarray, units: str) -> np.ndarray:
        """
        Converteer diameter van EPANET-eenheden naar meter.

        Parameters
        ──────────────────────
        diam_raw : ndarray  diameter zoals teruggegeven door ENgetlinkvalue
        units    : str      flow-eenheidsinstelling ('CMH', 'GPM', enz.)

        EPANET-conventie:
            SI  (CMH, LPS, LPM, MLD, CMD) : diameter in mm
            US  (CFS, GPM, MGD, IMGD, AFD): diameter in inch
        """
        if units in _US_UNITS:
            return diam_raw * _INCH_TO_M
        return diam_raw / 1000.0   # mm → m

    @staticmethod
    def _to_cms(flow: np.ndarray, units: str) -> np.ndarray:
        """
        Converteer flow-array naar m³/s.

        .. deprecated::
            Niet meer gebruikt binnen HydraulicModel. Gebruik get_hydraulic_state()
            voor gecombineerde flow + velocity conversie. Deze methode wordt in een
            toekomstige versie verwijderd.
        """
        import warnings
        warnings.warn(
            "HydraulicModel._to_cms() is deprecated en wordt in een toekomstige "
            "versie verwijderd. Gebruik get_hydraulic_state() als alternatief.",
            DeprecationWarning,
            stacklevel=2,
        )
        return flow * _FLOW_TO_CMS.get(units.upper(), 1.0 / 3600.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Diagnostiek
    # ═══════════════════════════════════════════════════════════════════════════

    def summary(self) -> str:
        """
        Geeft een korte samenvatting van het netwerk en de hydraulische toestand.

        Nuttig voor debuggen en verificatie van de eenheidsinstellingen.
        """
        links     = self._get_links()
        units     = self._get_flow_units()
        n_nodes   = len(self.net.nodes)
        n_pipes   = len(list(self.net.pipes))
        n_pumps   = len(list(self.net.pumps))
        n_valves  = len(list(self.net.valves))
        n_tanks   = len(self.net.tanks)
        n_res     = len(self.net.reservoirs)

        lines = [
            f"HydraulicModel — {self.net.inputfile or '(geen .inp)'}",
            f"  Flow-eenheden : {units}",
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

        if self.net.solved:
            try:
                flow, vel, rev = self.get_hydraulic_state()
                simtime_s = getattr(self.net, "solved_for_simtime", "?")
                lines += [
                    f"  Hydraulica    : opgelost voor t={simtime_s} s",
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
        n_nodes = len(self.net.nodes)
        units   = self._get_flow_units()
        return (
            f"<HydraulicModel pipes={n_pipes} nodes={n_nodes} "
            f"units={units} solved={self.net.solved}>"
        )
