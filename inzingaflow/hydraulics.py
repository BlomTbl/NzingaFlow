# inzingaflow/hydraulics.py

from __future__ import annotations
import numpy as np


class HydraulicModel:
    """
    Koppeling tussen epynet (EPANET via Python) en inzingaflow.

    Verantwoordelijkheden
    ----------------------
    - Laad het EPANET-netwerk en zet topologie om naar numpy-arrays.
    - Los hydraulica op voor een steady-state tijdstip (EPS: één stap per keer).
    - Lever flow [m³/s] en velocity [m/s] per leiding; corrigeer voor
      stroomrichting (negatief debiet → wissel start/end).
    - Zet EPANET-eenheden automatisch om naar SI.

    Correcties t.o.v. originele versie
    ------------------------------------
    - `.uid` i.p.v. `.id`  (epynet objecten gebruiken .uid)
    - node.index is 1-based in epynet → aftrekken van 1
    - diameter in epynet = mm → /1000 voor m
    - flow in epynet = m³/h (CMH) → /3600 voor m³/s; andere eenheden via _to_cms()
    - Negatieve flow: pipe_start/pipe_end worden omgedraaid zodat advectie
      altijd van lage index naar hoge index loopt
    """

    def __init__(self, inp_path: str):
        from epynet import Network
        self.net        = Network(inp_path)
        self._pipe_list = None        # gecached na solve()
        self._topology  = None        # gecached na eerste get_topology()
        self._units     = None        # gecached na eerste _get_flow_units()

    # ── Hydraulische berekening ───────────────────────────────────────────────

    def solve(self, simtime: int = 0) -> None:
        """
        Los hydraulica op voor één tijdstip [s] (steady-state of EPS-stap).

        Parameters
        ----------
        simtime : simulatietijd in seconden (0 = beginconditie)
        """
        self.net.solve(simtime=simtime)
        self._pipe_list = None   # invalideer pipe-cache na nieuwe oplossing

    # ── Topologie ─────────────────────────────────────────────────────────────

    def get_topology(self) -> tuple:
        """
        Geeft netwerktopologie terug als numpy-arrays (gecached).

        Returns
        -------
        pipe_start  : (n_pipes,) int32    0-based knoopindex beginpunt
        pipe_end    : (n_pipes,) int32    0-based knoopindex eindpunt
        pipe_length : (n_pipes,) float64  lengte [m]
        pipe_area   : (n_pipes,) float64  dwarsdoorsnede [m²]
        node_count  : int
        pipe_ids    : list[str]           EPANET pipe-namen
        node_names  : list[str]           knoopnamen in index-volgorde
        """
        if self._topology is not None:
            return self._topology

        pipes = self._get_pipes()

        # 1-based → 0-based
        pipe_start = np.array(
            [self.net.nodes[p.from_node.uid].index - 1 for p in pipes],
            dtype=np.int32,
        )
        pipe_end = np.array(
            [self.net.nodes[p.to_node.uid].index - 1 for p in pipes],
            dtype=np.int32,
        )
        pipe_length = np.array([p.length   for p in pipes], dtype=np.float64)
        pipe_diam   = np.array([p.diameter for p in pipes], dtype=np.float64) / 1000.0  # mm→m
        pipe_area   = np.pi * (pipe_diam / 2.0) ** 2

        node_count  = len(self.net.nodes)
        pipe_ids    = [p.uid for p in pipes]
        node_names  = [n.uid for n in self.net.nodes]

        self._topology = (
            pipe_start, pipe_end, pipe_length, pipe_area,
            node_count, pipe_ids, node_names,
        )
        return self._topology

    def get_topology_with_reversal(self) -> tuple:
        """
        Geeft topologie terug waarbij pipe_start/pipe_end zijn omgewisseld
        voor leidingen met negatief debiet (na solve()).

        Flow reversal correctie: EPANET kan negatieve debieten teruggeven
        als de werkelijke stroomrichting omgekeerd is t.o.v. de EPANET-definitie.
        Deze methode corrigeert pipe_start/pipe_end zodat advectie altijd
        van start → end loopt in de richting van de stroom.

        Returns
        -------
        Zelfde tuple als get_topology(), maar pipe_start/pipe_end zijn
        gecorrigeerd voor de huidige hydraulische toestand.

        Let op: de gecorrigeerde topologie is geldig voor het huidige tijdstip.
        Na update_hydraulics() opnieuw aanroepen.
        """
        (
            pipe_start_base, pipe_end_base,
            pipe_length, pipe_area,
            node_count, pipe_ids, node_names,
        ) = self.get_topology()

        pipes    = self._get_pipes()
        units    = self._get_flow_units()
        flow_raw = np.array([p.flow for p in pipes], dtype=np.float64)
        flow_si  = self._to_cms(flow_raw, units)

        # Negatief debiet → wissel start en end om
        reversed_mask = flow_si < 0
        pipe_start = pipe_start_base.copy()
        pipe_end   = pipe_end_base.copy()
        pipe_start[reversed_mask] = pipe_end_base[reversed_mask]
        pipe_end[reversed_mask]   = pipe_start_base[reversed_mask]

        return (
            pipe_start, pipe_end, pipe_length, pipe_area,
            node_count, pipe_ids, node_names,
        )

    # ── Hydraulische toestand ─────────────────────────────────────────────────

    def get_hydraulic_state(self) -> tuple:
        """
        Geeft debiet [m³/s], snelheid [m/s] en richting per leiding terug.

        Negatieve flow (omgekeerde stroming) wordt afgevangen:
        - flow en velocity worden als absolute waarden teruggegeven.
        - reversed geeft aan welke leidingen omgekeerd stromen t.o.v. de
          EPANET-definitie. Gebruik get_topology_with_reversal() voor
          correcte pipe_start/pipe_end bij flow reversals.

        Returns
        -------
        flow     : (n_pipes,) float64  m³/s, altijd ≥ 0
        velocity : (n_pipes,) float64  m/s,  altijd ≥ 0
        reversed : (n_pipes,) bool     True als EPANET-richting omgekeerd is
        """
        pipes    = self._get_pipes()
        units    = self._get_flow_units()

        flow_raw = np.array([p.flow     for p in pipes], dtype=np.float64)
        vel_raw  = np.array([p.velocity for p in pipes], dtype=np.float64)

        flow_si       = self._to_cms(flow_raw, units)
        reversed_mask = flow_si < 0

        return np.abs(flow_si), np.abs(vel_raw), reversed_mask

    # ── Hulpfuncties ──────────────────────────────────────────────────────────

    def _get_pipes(self) -> list:
        """Gecachede lijst van pipe-objecten (alleen pipes, geen pompen/kleppen)."""
        if self._pipe_list is None:
            self._pipe_list = list(self.net.pipes.values())
        return self._pipe_list

    def _get_flow_units(self) -> str:
        """Lees de flow-eenheid uit het EPANET-project (gecached)."""
        if self._units is not None:
            return self._units
        try:
            code = self.net.ep.ENgetflowunits()
            unit_map = {
                0: 'CFS', 1: 'GPM', 2: 'MGD', 3: 'IMGD', 4: 'AFD',
                5: 'LPS', 6: 'LPM', 7: 'MLD', 8: 'CMH', 9: 'CMD',
            }
            self._units = unit_map.get(code, 'CMH')
        except Exception:
            self._units = 'CMH'
        return self._units

    @staticmethod
    def _to_cms(flow: np.ndarray, units: str) -> np.ndarray:
        """Conversiefactoren naar m³/s."""
        factors = {
            'CFS':  0.028317,
            'GPM':  6.309e-5,
            'MGD':  0.043813,
            'IMGD': 0.052616,
            'AFD':  1.427e-5,
            'LPS':  1e-3,
            'LPM':  1 / 60_000,
            'MLD':  0.011574,
            'CMH':  1 / 3_600,
            'CMD':  1 / 86_400,
        }
        return flow * factors.get(units.upper(), 1 / 3_600)

    def __repr__(self) -> str:
        n_pipes = len(self._get_pipes())
        n_nodes = len(self.net.nodes)
        return f"<HydraulicModel pipes={n_pipes} nodes={n_nodes}>"
