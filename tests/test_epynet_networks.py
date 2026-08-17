# tests/test_epynet_networks.py
"""
Integratietests tegen echte EPANET .inp-netwerken via epynet.

In tegenstelling tot tests/test_inzingaflow.py en tests/test_validation.py
(die volledig op synthetische numpy-topologieën draaien) laden deze tests
daadwerkelijke .inp-bestanden via HydraulicModel/NzingaFlowSolver, zodat
epynet-specifieke edge cases worden gedekt die met arrays-only tests niet
zichtbaar zijn — met name het gedrag van include_valves en include_pumps.

Structuur
---------
E1  include_valves=True — topologie
    Controleert dat een PRV-klep correct wordt opgenomen: lengte 0 m
    (getattr-fallback), positieve diameter/oppervlak, geldige flow/velocity.

E2  include_valves=True — transport door de klep
    End-to-end test met NzingaFlowSolver: een geïnjecteerde tracer moet,
    via de klep heen, het benedenstroomse eindknooppunt bereiken met de
    verwachte (onverdunde) concentratie — regressietest voor de v1.2.1
    include_valves-feature (zie CHANGELOG.md).

E3  include_pumps=True — topologie mag niet crashen
    Regressietest voor de bugfix in hydraulics.py: epynet Pump-objecten
    hebben geen 'diameter' static_property. Vóór de fix gaf dit een
    onbehandelde AttributeError zodra include_pumps=True werd gebruikt op
    een netwerk met een echte pomp. Na de fix valt dit terug op 0.0,
    net als de bestaande length-fallback voor valves.

E4  Tankvolume via EPYnetDTD
    Regressietest voor de bugfix in solver.py::_get_tank_volumes(): de
    oude epynet-property heette `volume`, EPYnetDTD noemt 'm `tank_volume`
    (EN_TANKVOLUME). `n.volume` bestaat niet op EPYnetDTD's Tank en werd
    stilzwijgend opgevangen door de brede `except Exception`, waardoor
    ELKE tank altijd de fallback-waarde 1000.0 m³ kreeg — nooit het echte,
    door EPANET berekende volume. Deze test rekent het verwachte volume
    (cilinder: π·r²·h) uit en vergelijkt dat met wat de solver teruggeeft.

E5  Sessie-hergebruik (EN_openH één keer per sessie) — numerieke regressie
    Regressietest voor de performance-optimalisatie in
    `_NoSaveHydraulicSolver` (hydraulics.py): EN_openH() wordt maar één
    keer per HydraulicModel-sessie aangeroepen i.p.v. bij elke solve()
    opnieuw. Empirisch gevonden valkuil tijdens het bouwen hiervan: de voor
    de hand liggende EN_initH()-flag EN_NOSAVE bleek bij sessie-hergebruik
    een klein maar reëel numeriek verschil te geven t.o.v. "vers
    HydraulicModel per simtime" (~1e-8 relatief) — omdat EN_NOSAVE de
    laatst-geconvergeerde debieten van de vorige stap hergebruikt als
    startpunt voor de volgende Newton-Raphson-iteratie (warm start), iets
    wat een verse sessie niet heeft (cold start). EN_INITFLOW forceert een
    cold start bij elke stap, ook binnen een hergebruikte sessie — dat is
    de flag die hier daadwerkelijk gebruikt wordt. Deze test vergelijkt,
    op een netwerk met tank + patroon (zodat EN_initH() elke stap
    daadwerkelijk iets terugzet om warm/cold-verschil zichtbaar te maken),
    (a) één sessie met 4 opeenvolgende solve()-aanroepen tegen (b) 4 losse
    HydraulicModel-instanties — met np.allclose(rtol=1e-9, atol=1e-12),
    niet alleen "geen crash".

Alle tests zijn gemarkeerd met @pytest.mark.requires_epynet.
"""
from __future__ import annotations
import math
import textwrap

import numpy as np
import pytest

from nzingaflow.hydraulics import HydraulicModel
from nzingaflow import NzingaFlowSolver

pytestmark = pytest.mark.requires_epynet


# ── Netwerk met een PRV-klep tussen J1 en J2, dead-end J3 met demand ────────
_VALVE_NETWORK = textwrap.dedent("""\
    [TITLE]
    Testnetwerk met PRV-klep

    [JUNCTIONS]
    ;ID              Elev        Demand      Pattern
     J1               0           0
     J2               0           2
     J3               0           1

    [RESERVOIRS]
    ;ID              Head        Pattern
     R1               50

    [PIPES]
    ;ID              Node1           Node2           Length      Diameter    Roughness   MinorLoss   Status
     P1               R1              J1              100         200         100         0           Open
     P2               J2              J3              100         150         100         0           Open

    [VALVES]
    ;ID              Node1           Node2           Diameter    Type        Setting     MinorLoss
     V1               J1              J2              150         PRV         30

    [TIMES]
     Duration           0

    [OPTIONS]
     Units              LPS

    [COORDINATES]
     J1               0               0
     J2               100             0
     J3               200             0
     R1               -100            0

    [END]
    """)

# ── Netwerk met een pomp (geen diameter in EPANET) ───────────────────────────
_PUMP_NETWORK = textwrap.dedent("""\
    [TITLE]
    Testnetwerk met pomp

    [JUNCTIONS]
    ;ID              Elev        Demand      Pattern
     J1               0           5

    [RESERVOIRS]
    ;ID              Head        Pattern
     R1               50

    [PUMPS]
    ;ID              Node1           Node2           Parameters
     PU1              R1              J1              HEAD 1

    [CURVES]
    ;ID              X-Value          Y-Value
     1                0                60
     1                10               40
     1                20               0

    [TIMES]
     Duration           0

    [OPTIONS]
     Units              LPS

    [COORDINATES]
     J1                0                0
     R1                -100             0

    [END]
    """)

# ── Netwerk met een tank (voor tank_volume-regressietest) ────────────────────
_TANK_NETWORK = textwrap.dedent("""\
    [TITLE]
    Testnetwerk met tank

    [JUNCTIONS]
    ;ID              Elev        Demand      Pattern
     J1               0           5

    [RESERVOIRS]
    ;ID              Head        Pattern
     R1               50

    [TANKS]
    ;ID              Elevation   InitLevel   MinLevel    MaxLevel    Diameter    MinVol      VolCurve
     T1               10          5           0           10          20          0

    [PIPES]
    ;ID              Node1           Node2           Length      Diameter    Roughness   MinorLoss   Status
     P1               R1              J1              100         200         100         0           Open
     P2               J1              T1              100         150         100         0           Open

    [TIMES]
     Duration           0

    [OPTIONS]
     Units              LPS

    [COORDINATES]
     J1                0                0
     R1                -100             0
     T1                100              0

    [END]
    """)

# ── Netwerk met tank + patroon (voor sessie-hergebruik-regressietest) ────────
# Een patroon-gestuurde demand zorgt dat elke simtime een ander debiet geeft
# (anders zou EN_INITFLOW vs EN_NOSAVE toevallig hetzelfde resultaat geven,
# los van de bug die dit netwerk moet vangen).
_EPS_TANK_NETWORK = textwrap.dedent("""\
    [TITLE]
    EPS-testnetwerk met tank en patroon

    [JUNCTIONS]
     J1               0           5

    [RESERVOIRS]
     R1               50

    [TANKS]
     T1               10          5           0           10          20          0

    [PIPES]
     P1               R1              J1              200         200         100         0           Open
     P2               J1              T1              150         150         100         0           Open

    [PATTERNS]
     PAT1             1.0  1.5  0.5  2.0  0.8  1.2

    [DEMANDS]
     J1               5.0             PAT1

    [TIMES]
     Duration           6
     Hydraulic Timestep 1
     Pattern Timestep   1

    [OPTIONS]
     Units              LPS

    [COORDINATES]
     J1                0                0
     R1                -100             0
     T1                100              0

    [END]
    """)


@pytest.fixture
def valve_inp(tmp_path):
    """Schrijft _VALVE_NETWORK weg naar een tijdelijk .inp-bestand."""
    p = tmp_path / "valve_network.inp"
    p.write_text(_VALVE_NETWORK)
    return str(p)


@pytest.fixture
def pump_inp(tmp_path):
    """Schrijft _PUMP_NETWORK weg naar een tijdelijk .inp-bestand."""
    p = tmp_path / "pump_network.inp"
    p.write_text(_PUMP_NETWORK)
    return str(p)


@pytest.fixture
def tank_inp(tmp_path):
    """Schrijft _TANK_NETWORK weg naar een tijdelijk .inp-bestand."""
    p = tmp_path / "tank_network.inp"
    p.write_text(_TANK_NETWORK)
    return str(p)


@pytest.fixture
def eps_tank_inp(tmp_path):
    """Schrijft _EPS_TANK_NETWORK weg naar een tijdelijk .inp-bestand."""
    p = tmp_path / "eps_tank_network.inp"
    p.write_text(_EPS_TANK_NETWORK)
    return str(p)


class TestIncludeValvesTopology:
    """E1 — topologie-opbouw met include_valves=True."""

    def test_valve_included_with_zero_length(self, valve_inp):
        hm = HydraulicModel(valve_inp, include_valves=True)
        hm.solve()
        (pipe_start, pipe_end, pipe_length, pipe_area,
         node_count, pipe_ids, node_names) = hm.get_topology()

        assert "V1" in pipe_ids
        v_idx = pipe_ids.index("V1")
        # Valve heeft geen EPANET-lengte → getattr-fallback naar 0.0
        assert pipe_length[v_idx] == 0.0
        # Valve heeft wél een diameter → oppervlak > 0
        assert pipe_area[v_idx] > 0.0

    def test_valve_excluded_by_default(self, valve_inp):
        hm = HydraulicModel(valve_inp)   # include_valves=False (standaard)
        hm.solve()
        _, _, _, _, _, pipe_ids, _ = hm.get_topology()
        assert "V1" not in pipe_ids
        assert set(pipe_ids) == {"P1", "P2"}

    def test_hydraulic_state_valid_for_valve(self, valve_inp):
        hm = HydraulicModel(valve_inp, include_valves=True)
        hm.solve()
        flow, velocity, reversed_mask = hm.get_hydraulic_state()
        assert (flow >= 0.0).all()
        assert (velocity >= 0.0).all()
        assert flow.shape == velocity.shape == reversed_mask.shape


class TestIncludeValvesTransport:
    """E2 — end-to-end stoftransport door een klep."""

    def test_tracer_reaches_downstream_deadend(self, valve_inp):
        solver = NzingaFlowSolver(valve_inp, n_species=1, include_valves=True)
        solver.inject("R1", C_vector=[1.0], volume=0.0005)

        j3 = solver.node_index["J3"]
        arrived_C = None
        for _ in range(2000):
            node_C = solver.step(dt=2.0, decay_k=[0.0])
            if node_C[j3, 0] > 1e-6:
                arrived_C = node_C[j3, 0]
                break

        assert arrived_C is not None, (
            "Tracer heeft J3 niet bereikt binnen de simulatietijd — "
            "transport door de klep (V1) werkt niet zoals verwacht."
        )
        # Conservatieve tracer zonder verval/menging-verdunning op dit pad
        # moet vrijwel onverdund aankomen.
        assert arrived_C == pytest.approx(1.0, rel=1e-6)

    def test_no_transport_when_valves_excluded(self, valve_inp):
        # Zonder include_valves is J1→J2 topologisch niet verbonden, dus
        # een injectie bij R1 kan J2/J3 nooit bereiken.
        solver = NzingaFlowSolver(valve_inp, n_species=1, include_valves=False)
        solver.inject("R1", C_vector=[1.0], volume=0.0005)

        j3 = solver.node_index["J3"]
        for _ in range(500):
            node_C = solver.step(dt=2.0, decay_k=[0.0])
            assert node_C[j3, 0] == 0.0


class TestIncludePumpsDiameterFallback:
    """E3 — regressietest: include_pumps=True mag niet crashen op diameter."""

    def test_pump_topology_does_not_raise(self, pump_inp):
        hm = HydraulicModel(pump_inp, include_pumps=True)
        hm.solve()
        # Voor de fix: AttributeError('Nonexistant Attribute', 'diameter')
        (pipe_start, pipe_end, pipe_length, pipe_area,
         node_count, pipe_ids, node_names) = hm.get_topology()

        assert "PU1" in pipe_ids
        pu_idx = pipe_ids.index("PU1")
        # Pomp heeft geen EPANET-diameter → fallback naar 0.0 (net als
        # lengte bij valves), dus oppervlak is exact 0.
        assert pipe_area[pu_idx] == 0.0

    def test_pump_excluded_by_default(self, pump_inp):
        hm = HydraulicModel(pump_inp)   # include_pumps=False (standaard)
        hm.solve()
        _, _, _, _, _, pipe_ids, _ = hm.get_topology()
        assert pipe_ids == []


class TestTankVolume:
    """E4 — regressietest: tankvolume via EPYnetDTD's `tank_volume`-property."""

    def test_tank_volume_matches_computed_cylinder_volume(self, tank_inp):
        # T1: diameter 20 m, initieel peil 5 m boven de bodem (min_level=0,
        # elevation los van level) → verwacht volume = π·r²·h = π·10²·5.
        solver = NzingaFlowSolver(tank_inp, n_species=1)
        volumes = solver._get_tank_volumes()

        assert volumes.shape == (1,)
        expected = math.pi * (20.0 / 2.0) ** 2 * 5.0
        assert volumes[0] == pytest.approx(expected, rel=1e-6)
        # Vóór de fix (n.volume i.p.v. n.tank_volume) viel dit altijd terug
        # op de veilige standaardwaarde, ongeacht het echte tankvolume.
        assert volumes[0] != pytest.approx(1000.0, rel=1e-6)


class TestSessionReuse:
    """E5 — numerieke regressie voor EN_openH()-sessiehergebruik.

    Vergelijkt (a) één HydraulicModel-sessie met 4 opeenvolgende
    solve()-aanroepen tegen (b) 4 losse HydraulicModel-instanties (één
    per simtime) — moet bit-identiek zijn. Zie de moduledocstring
    hierboven (E5) voor waarom dit niet vanzelfsprekend is (EN_NOSAVE vs
    EN_INITFLOW)."""

    _SIMTIMES = [0, 3600, 7200, 10800]

    def test_reused_session_matches_fresh_sessions(self, eps_tank_inp):
        # (a) één sessie, herhaalde solve()-aanroepen
        flows_reused, vels_reused = [], []
        with HydraulicModel(eps_tank_inp) as hm_reused:
            for st in self._SIMTIMES:
                hm_reused.solve(simtime=st)
                flow, vel, _ = hm_reused.get_hydraulic_state()
                flows_reused.append(flow.copy())
                vels_reused.append(vel.copy())

        # (b) vers HydraulicModel per simtime
        flows_fresh, vels_fresh = [], []
        for st in self._SIMTIMES:
            with HydraulicModel(eps_tank_inp) as hm_fresh:
                hm_fresh.solve(simtime=st)
                flow, vel, _ = hm_fresh.get_hydraulic_state()
                flows_fresh.append(flow.copy())
                vels_fresh.append(vel.copy())

        for i, st in enumerate(self._SIMTIMES):
            assert np.allclose(
                flows_reused[i], flows_fresh[i], rtol=1e-9, atol=1e-12
            ), f"flow-mismatch bij simtime={st}: {flows_reused[i]} vs {flows_fresh[i]}"
            assert np.allclose(
                vels_reused[i], vels_fresh[i], rtol=1e-9, atol=1e-12
            ), f"velocity-mismatch bij simtime={st}"

    def test_solve_is_memoized_for_identical_simtime(self, eps_tank_inp):
        # Tweede solve() met dezelfde simtime moet een no-op zijn (geen
        # nieuwe EN_runH()), dus de cached HydraulicState-instantie blijft
        # ongewijzigd (zelfde object, niet alleen gelijke waarden).
        with HydraulicModel(eps_tank_inp) as hm:
            hm.solve(simtime=3600)
            state_after_first = hm._state
            hm.solve(simtime=3600)
            assert hm._state is state_after_first
