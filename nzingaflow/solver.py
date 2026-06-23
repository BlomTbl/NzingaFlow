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
import numpy as np


class NzingaFlowSolver:
    """
    Vectorized Lagrangian Transport Approach solver voor EPANET-netwerken.

    Kenmerken
    ----------
    Multi-species     C heeft shape (n, n_species); alle stoffen in één operatie.
    Wandreacties      Twee-film-model per leiding; afhankelijk van Re en k_wall.
    Tanks             CSTR-model met impliciet Euler (stabiel voor grote dt).
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
                            Elk segment verliest per tijdstap proportioneel volume
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
        """
        from .hydraulics import HydraulicModel
        from .segments   import SegmentStore
        from .stability  import MassBalanceTracker

        self.hyd = HydraulicModel(inp_path, include_valves=include_valves)
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

        # Uitgaande leidingen per knoop
        self._node_outpipes: dict[int, list[int]] = defaultdict(list)
        for p, s in enumerate(self.pipe_start):
            self._node_outpipes[int(s)].append(p)

        # Cache van in/out leidingen per tank — herbouwd bij update_hydraulics()
        self._tank_in_pipes:  list[np.ndarray] = []   # per tank: array van inkomende pipe-indices
        self._tank_out_pipes: list[np.ndarray] = []   # per tank: array van uitgaande pipe-indices
        self._rebuild_tank_pipe_cache()

        # Per-knoop exit-routing caches (herbouwd bij update_hydraulics)
        # _node_type[nd]  : 0=eindknoop, 1=doorgaand, >=2=splitsing
        # _node_out0[nd]  : index van eerste (hoogste-debiet) uitgangsleiding
        # Hiermee wordt het dominante pad in _handle_exits volledig vectorized.
        self._node_type: np.ndarray | None = None   # (node_count,) int32
        self._node_out0: np.ndarray | None = None   # (node_count,) int32
        self._rebuild_node_routing_cache()

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

        # Hydraulica-cache
        self._flow:     np.ndarray | None = None
        self._velocity: np.ndarray | None = None
        self._reversed: np.ndarray | None = None   # (n_pipes,) bool — flow reversal per leiding
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

    # ── Beginkwaliteit ────────────────────────────────────────────────────────

    def set_initial_quality(
        self,
        node_quality: np.ndarray | dict,
        simtime:      int = 0,
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
        self.segments.n = 0

        # ── Vul elke leiding met één beginsegment ───────────────────────────
        # Upstream knoop = pipe_start (al gecorrigeerd voor flow-richting)
        n_added = 0
        for pi in range(len(self.pipe_ids)):
            v = float(velocity[pi])
            if v < 1e-9:
                continue   # stilstaand water — geen segment
            L   = float(self.pipe_length[pi])
            A   = float(self.pipe_area[pi])
            vol = L * A    # totale leidinginhoud [m³]
            src_node = int(self.pipe_start[pi])
            self.segments.add(
                pipe=pi, x=0.0, volume=vol,
                C_vector=C_node[src_node],
            )
            n_added += 1

        # Invalideer hydraulica-cache (segmenten zijn nieuw)
        self._combined_exp_dt = -1.0

    # ── Hydraulica ────────────────────────────────────────────────────────────

    def update_hydraulics(self, simtime: int = 0) -> None:
        """
        Herbereken hydraulica voor een EPS-tijdstip [s].

        Detecteert automatisch flow reversals en past pipe_start, pipe_end
        en _node_outpipes aan zodat advectie altijd in de juiste richting loopt.
        """
        self.hyd.solve(simtime=simtime)
        self._flow     = None
        self._velocity = None
        self._reversed = None
        self._apply_flow_reversal()
        self._update_kwall_vol()
        self._combined_exp_dt = -1.0   # invalideer na hydraulica-update
        # Tankvolumes kunnen veranderen tijdens EPS (peil varieert)
        if len(self._tank_nodes) > 0:
            self._tank_V = self._get_tank_volumes()
            self._rebuild_tank_pipe_cache()
        self._rebuild_node_routing_cache()

    def _apply_flow_reversal(self) -> None:
        """
        Pas pipe_start, pipe_end en _node_outpipes aan op basis van de
        actuele stromingsrichting. Wordt aangeroepen na elke hydraulica-update.

        Leidingen waarvan EPANET een negatief debiet rapporteert stromen
        feitelijk in omgekeerde richting. We wisselen pipe_start/pipe_end om
        zodat de LTA-advectie (x loopt van 0 naar pipe_length) altijd klopt.
        """
        _, _, reversed_mask = self.hyd.get_hydraulic_state()

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

        self._reversed = reversed_mask

        # Spiegelen van segmentposities in omgekeerde leidingen.
        # Na een flow reversal loopt x nog steeds van het OUDE startpunt.
        # Maar pipe_start en pipe_end zijn nu omgewisseld, dus x=0 is nu
        # het NIEUWE startpunt (= het oude eindpunt).
        # Correctie: x_new = pipe_length - x_old  voor elk segment in een omgekeerde leiding.
        n = self.segments.n
        if n > 0 and reversed_mask.any():
            seg_pipe   = self.segments.pipe[:n]
            rev_pipes  = np.where(reversed_mask)[0]
            in_rev     = np.isin(seg_pipe, rev_pipes)
            if in_rev.any():
                L_seg = self.pipe_length[seg_pipe[in_rev]]
                self.segments.x[:n][in_rev] = L_seg - self.segments.x[:n][in_rev]
                # Klamp negatieve waarden (numerieke ruis) op 0
                np.clip(self.segments.x[:n], 0.0, None,
                        out=self.segments.x[:n])

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
            k_wall direct als k_eff: k_vol = k_wall·4/D, geen filmweerstand.
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
            if self._tracker:
                self._tracker.record_injection(C_arr, volume)
        else:
            # Verdeel volume proportioneel over uitgaande leidingen
            flows   = np.array([flow[p] for p in out], dtype=np.float64)
            tot_q   = flows.sum()
            if tot_q <= 0:
                # Geen debiet: injecteer in eerste leiding als fallback
                self.segments.add(pipe=out[0], x=0.0, volume=volume, C_vector=C_arr)
                if self._tracker:
                    self._tracker.record_injection(C_arr, volume)
                return
            for p, q in zip(out, flows):
                vol_p = volume * q / tot_q
                if vol_p > 0:
                    self.segments.add(pipe=p, x=0.0, volume=vol_p, C_vector=C_arr)
            if self._tracker:
                self._tracker.record_injection(C_arr, volume)

    def inject_pipe(self, pipe_uid: str, C_vector, volume: float,
                    x: float = 0.0) -> None:
        """Injecteer direct in een leiding op positie x [m]."""
        pipe_idx = self.pipe_index[pipe_uid]
        C_arr = np.asarray(C_vector, dtype=np.float64)
        self.segments.add(pipe=pipe_idx, x=x, volume=volume, C_vector=C_arr)
        if self._tracker:
            self._tracker.record_injection(C_arr, volume)

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
        6. Tank-update (CSTR)
        7. Pipe-overgang / splitsing / eindknoop-verwijdering
        8. Segment merging (elke merge_interval stappen)
        9. Massabalans registreren (indien track_mass=True)

        Parameters
        ----------
        dt             : tijdstap [s]
        decay_k        : (n_species,) bulkvervalconstanten [1/s]
        merge_interval : voer merging uit elke N stappen
        merge_tol      : concentratietolerantie voor merging
        check_cfl      : als True, controleer CFL en waarschuw indien nodig

        Returns
        -------
        node_C : (node_count, n_species)
        """
        from .lta     import (combined_decay_multi, build_combined_exp,
                             advect, exit_detect, node_mixing_multi)
        from .merging import merge_segments

        flow, velocity = self._get_hydraulics()
        decay_k = np.asarray(decay_k, dtype=np.float64)
        n = self.segments.n
        self._last_dt = float(dt)   # bewaar voor gebruik in booster_inject()

        if check_cfl:
            self.check_stability(dt, decay_k, warn=True)

        if n == 0:
            return np.zeros((self.node_count, self.n_species), dtype=np.float64)

        # 1+2. Bulk + wandverval gecombineerd in één pass
        # combined_exp[p,s] = exp(-(k_bulk[s] + k_wall[p,s]) * dt)
        # Gecached; alleen herbouwd als dt of hydraulica verandert.
        if self._geochem is not None:
            pipe_diam = np.sqrt(4 * self.pipe_area / np.pi)
            _, velocity = self._get_hydraulics()
            self._geochem.apply_geochemistry(
                self.segments, dt,
                pipe_diam=pipe_diam,
                pipe_vel=velocity,
            )
        else:
            if self._combined_exp is None or self._combined_exp_dt != dt:
                self._combined_exp    = build_combined_exp(
                    decay_k, self._k_wall_vol, dt, len(self.pipe_ids)
                )
                self._combined_exp_dt = dt
            combined_decay_multi(
                self.segments.C, self.segments.pipe,
                self._combined_exp, n=n,
            )

        # 3. Advectie
        advect(self.segments.x, self.segments.pipe, velocity, dt, n=n)

        # 3b. Lekkage: proportioneel volume-verlies per segment
        # Elk segment verliest een fractie _leakage_fraction van zijn volume
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
            self.segments.volume[:n] *= np.exp(-lam_lek * dt)

        # 4. Exit-detectie — in-place in pre-allocated buffer
        exit_detect(
            self.segments.x, self.segments.pipe, self.pipe_length,
            self._buf_exit_mask, n=n,
        )
        exit_mask = self._buf_exit_mask

        # 5. Knoopmenging — schrijf in pre-allocated buffer, geen nieuwe array
        node_C = node_mixing_multi(
            exit_mask,
            self.segments.pipe,
            self.segments.C,
            flow, self.pipe_end, self.node_count,
            out=self._buf_node_C,
            wC_buf=self._buf_wC,
            node_flow_buf=self._buf_node_flow,
            n=n,
        )

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
            self._geochem.apply_mixing(node_C, node_flow, dt)

        # 6. Tank-update
        if len(self._tank_nodes) > 0:
            node_C = self._update_tanks(node_C, flow, decay_k, dt)

        # 7. Pipe-overgang
        exited_C, exited_V = self._handle_exits(exit_mask, flow)

        # 8. Merging
        self._step_counter += 1
        if self._step_counter % merge_interval == 0:
            merge_segments(self.segments, tol=merge_tol,
                          n_pipes=len(self.pipe_ids))

        # 9. Massabalans
        if self._tracker:
            self._tracker.record_step(
                self.segments, decay_k, dt,
                self._k_wall_vol, exited_C, exited_V,
            )

        return node_C

    # ── Massabalansrapport ────────────────────────────────────────────────────

    def mass_balance(self) -> dict | None:
        """
        Geeft het massabalansrapport terug (alleen als track_mass=True).
        """
        if not self._tracker:
            return None
        return self._tracker.report(self.segments)

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
        self._buf_exit_C    = np.zeros((new_capacity, self.n_species), dtype=np.float64)
        self._buf_exit_V    = np.zeros(new_capacity,                   dtype=np.float64)
        self._buf_wC        = np.zeros((new_capacity, self.n_species), dtype=np.float64)
        self._buf_exit_mask = np.zeros(new_capacity,                   dtype=np.bool_)

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
        """Huidige tankvolumes [m³] via epynet."""
        volumes = []
        for n in self.hyd.net.tanks:
            try:
                volumes.append(float(n.volume))
            except Exception:
                volumes.append(1000.0)   # veilige standaard
        return np.array(volumes, dtype=np.float64) if volumes else np.array([])

    def _update_tanks(
        self,
        node_C:  np.ndarray,   # (node_count, n_species)
        flow:    np.ndarray,
        decay_k: np.ndarray,
        dt:      float,
    ) -> np.ndarray:
        """
        Pas CSTR-tankmodel toe en schrijf tankconcentraties naar node_C.

        Gebruikt gecachede in/out pipe-lijsten per tank (zie _rebuild_tank_pipe_cache)
        zodat geen O(n_tanks × n_pipes) np.where()-aanroepen nodig zijn per tijdstap.
        """
        from .lta import tank_step_implicit

        if len(self._tank_nodes) == 0:
            return node_C

        n_tanks = len(self._tank_nodes)
        # Gebruik pre-allocated buffers — geen nieuwe arrays per tijdstap
        Q_in  = self._buf_Q_in[:n_tanks];  Q_in[:]  = 0.0
        Q_out = self._buf_Q_out[:n_tanks]; Q_out[:] = 0.0
        C_in  = self._buf_C_in[:n_tanks];  C_in[:]  = 0.0

        for ti, tank_node in enumerate(self._tank_nodes):
            in_pipes  = self._tank_in_pipes[ti]
            out_pipes = self._tank_out_pipes[ti]
            Q_in[ti]  = flow[in_pipes].sum()  if len(in_pipes)  else 0.0
            Q_out[ti] = flow[out_pipes].sum() if len(out_pipes) else 0.0
            C_in[ti]  = node_C[tank_node]

        # CSTR update (in-place op self._tank_C) — gebruik gecachede volumes
        if len(self._tank_V) == len(self._tank_nodes):
            tank_step_implicit(self._tank_C, Q_in, C_in, Q_out, self._tank_V, decay_k, dt)
            # Schrijf tankconcentraties terug naar node_C
            node_C[self._tank_nodes] = self._tank_C

        return node_C

    # ── Exit-verwerking ───────────────────────────────────────────────────────

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
        node_type = np.array(
            [len(self._node_outpipes.get(i, [])) for i in range(nc)],
            dtype=np.int32,
        )
        node_out0 = np.array(
            [self._node_outpipes.get(i, [0])[0] for i in range(nc)],
            dtype=np.int32,
        )
        self._node_type = node_type
        self._node_out0 = node_out0

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
        thru_sel = nd_type == 1
        if thru_sel.any():
            thru_idx = exit_idx[thru_sel]
            thru_nd  = exit_nodes[thru_sel]
            self.segments.pipe[thru_idx] = self._node_out0[thru_nd]
            self.segments.x[thru_idx]    = 0.0

        # ── 2. Eindknopen: vectorized kopieer, dan bulk-remove ─────────────
        end_sel = nd_type == 0
        n_end   = int(end_sel.sum())
        if n_end > 0:
            end_idx = exit_idx[end_sel]
            # Kopieer naar exit-buffers in één bulk-operatie (geen loop)
            self._buf_exit_C[:n_end] = self.segments.C[end_idx]
            self._buf_exit_V[:n_end] = self.segments.volume[end_idx]
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
            em2    = self.segments.x[:n2] >= self.pipe_length[self.segments.pipe[:n2]]
            s_idx  = np.where(em2)[0]
            s_pipe = self.segments.pipe[s_idx]
            s_nd   = self.pipe_end[s_pipe]
            s_type = self._node_type[s_nd]
            split_only = s_idx[s_type >= 2]

            for seg_i in split_only:
                nd    = int(self.pipe_end[self.segments.pipe[seg_i]])
                out   = self._node_outpipes.get(nd, [])
                if not out:
                    continue
                out_s    = sorted(out, key=lambda p: -flow[p])
                tot_flow = sum(flow[p] for p in out_s)
                orig_vol = self.segments.volume[seg_i]
                orig_C[:]                   = self.segments.C[seg_i]
                self.segments.pipe[seg_i]   = out_s[0]
                self.segments.x[seg_i]      = 0.0
                self.segments.volume[seg_i] = orig_vol * flow[out_s[0]] / tot_flow
                for p in out_s[1:]:
                    self.segments.add(
                        pipe=p, x=0.0,
                        volume=orig_vol * flow[p] / tot_flow,
                        C_vector=orig_C,
                    )

        return self._buf_exit_C[:n_end], self._buf_exit_V[:n_end]

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
