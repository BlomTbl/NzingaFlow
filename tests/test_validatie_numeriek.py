"""
Numerieke validatie van het transport (massabehoud, analytische en EPANET-referenties).

V1  Eén leiding        — aankomsttijd = L/v (±1 stap), massabalans exact
V2  Leidingen in serie — massabalans exact
V3  Mengpunt           — stationaire concentratie = debietgewogen gemiddelde (analytisch)
V4  Mengpunt           — massabehoud met gevulde leidingen; lege leidingen = bekende beperking
V5  Afsluiter          — open: stof passeert; gesloten: stof blokkeert
V6  Stroomomkering     — knoopconcentratie volgt de nieuwe aanvoerrichting
V7  EPANET-referentie  — tijdreeks op het mengpunt vs. EPANET-kwaliteitsberekening

Alle tests gebruiken kleine netwerken uit tekst (tmp_path) en zijn gemarkeerd
met @pytest.mark.requires_epynet.
"""
from __future__ import annotations
import textwrap
import numpy as np
import pytest

from nzingaflow import NzingaFlowSolver
from nzingaflow.eps import EPSRunner
from nzingaflow.hydraulics import HydraulicModel

pytestmark = pytest.mark.requires_epynet

_NET_EEN = """\
[TITLE]
een leiding

[JUNCTIONS]
 J1 0 5

[RESERVOIRS]
 R1 50

[PIPES]
 P1 R1 J1 30 150 100 0 Open

[TIMES]
 Duration 0

[OPTIONS]
 Units LPS

[COORDINATES]
 J1 0 0
 R1 -100 0

[END]
"""

_NET_SERIE = """\
[TITLE]
een leiding

[JUNCTIONS]
 J1 0 0
 J2 0 5

[RESERVOIRS]
 R1 50

[PIPES]
 P1 R1 J1 30 150 100 0 Open
 P2 J1 J2 30 150 100 0 Open

[TIMES]
 Duration 0

[OPTIONS]
 Units LPS

[COORDINATES]
 J1 0 0
 J2 50 0
 R1 -100 0

[END]
"""

_NET_MIX = """\
[TITLE]
mengpunt

[JUNCTIONS]
;ID Elev Demand Pattern
 J1 0 0
 J2 0 5

[RESERVOIRS]
;ID Head Pattern
 R1 50
 R2 50

[PIPES]
;ID N1 N2 Length Diameter Roughness MinorLoss Status
 P1 R1 J1 30 200 100 0 Open
 P2 R2 J1 30 100 100 0 Open
 P3 J1 J2 30 150 100 0 Open

[TIMES]
 Duration 1800
 Hydraulic Timestep 0:05
 Quality Timestep 0:00:05
 Report Timestep 0:05

[OPTIONS]
 Units LPS

[COORDINATES]
 J1 0 0
 J2 100 0
 R1 -100 50
 R2 -100 -50

[END]
"""

_NET_VALVE_OPEN = """\
[TITLE]
gesloten afsluiter

[JUNCTIONS]
 J1 0 1
 J2 0 0
 J3 0 0.5

[RESERVOIRS]
 R1 50

[PIPES]
 P1 R1 J1 30 200 100 0 Open
 P2 J2 J3 30 150 100 0 Open

[VALVES]
 V1 J1 J2 150 TCV 0 0

[STATUS]
 V1 Open

[TIMES]
 Duration 0

[OPTIONS]
 Units LPS

[COORDINATES]
 J1 0 0
 J2 100 0
 J3 200 0
 R1 -100 0

[END]
"""

_NET_VALVE_CLOSED = """\
[TITLE]
gesloten afsluiter

[JUNCTIONS]
 J1 0 1
 J2 0 0
 J3 0 0

[RESERVOIRS]
 R1 50

[PIPES]
 P1 R1 J1 30 200 100 0 Open
 P2 J2 J3 30 150 100 0 Open

[VALVES]
 V1 J1 J2 150 TCV 0 0

[STATUS]
 V1 Closed

[TIMES]
 Duration 0

[OPTIONS]
 Units LPS

[COORDINATES]
 J1 0 0
 J2 100 0
 J3 200 0
 R1 -100 0

[END]
"""

_NET_REV = """\
[TITLE]
stroomomkering

[JUNCTIONS]
 J1 0 0

[RESERVOIRS]
 R1 50 HP
 R2 45

[PIPES]
 P1 R1 J1 20 200 100 0 Open
 P2 J1 R2 20 200 100 0 Open

[PATTERNS]
 HP 1 1 0.8 0.8 0.8 0.8

[TIMES]
 Duration 3600
 Hydraulic Timestep 0:10
 Pattern Timestep 0:10
 Quality Timestep 0:00:05

[OPTIONS]
 Units LPS

[COORDINATES]
 J1 0 0
 R1 -100 0
 R2 100 0

[END]
"""


def _inp(tmp_path, tekst, naam="net.inp"):
    p = tmp_path / naam
    p.write_text(tekst)
    return str(p)


def _run(inp, sched, duur, dt=5.0, hyd_dt=300.0, **kw):
    s = NzingaFlowSolver(inp, n_species=1, track_mass=True, **kw)
    r = EPSRunner(s, qual_dt=dt, hyd_dt=hyd_dt, duration=duur)
    res = r.run(decay_k=[0.0], inject_schedule=sched, check_cfl=False)
    return s, r, res


def _bron(**conc):
    return {k: [(0, 9e9, np.array([float(c)]))] for k, c in conc.items()}


def _epanet_tijdreeks(inp, bronnen, knopen, t_eind):
    """Referentie: EPANET-kwaliteitsberekening (stof, reservoir-INITQUAL = bron)."""
    from epynet.network import Network
    from epynet import enum
    n = Network(inp)
    n.EN_setqualtype(enum.EN_QualityType.EN_CHEM, "chem", "mg/L", "")
    idx = {nd.uid: nd.index for nd in n.nodes}
    NP = enum.EN_NodeProperty
    for uid, c in bronnen.items():
        n.EN_setnodevalue(idx[uid], NP.EN_INITQUAL, float(c))
    n.EN_solveH()
    n.EN_openQ()
    n.EN_initQ(enum.EN_InitQOption.EN_NOSAVE)
    ts, q = [], []
    while True:
        t = n.EN_runQ()
        t = t.total_seconds() if hasattr(t, "total_seconds") else float(t)
        if t > t_eind:
            break
        ts.append(t)
        q.append([n.EN_getnodevalue(idx[k], NP.EN_QUALITY) for k in knopen])
        left = n.EN_stepQ()
        left = left.total_seconds() if hasattr(left, "total_seconds") else float(left)
        if left <= 0:
            break
    n.EN_closeQ()
    return np.array(ts), np.array(q)


# ── V1 ────────────────────────────────────────────────────────────────────────
def test_een_leiding_aankomsttijd_en_massabehoud(tmp_path):
    inp = _inp(tmp_path, _NET_EEN)
    dt = 5.0
    s, r, res = _run(inp, _bron(R1=1.0), 1200.0, dt=dt)
    flow, vel = s._get_hydraulics()
    t_aank = 30.0 / float(vel[0])                    # L / v
    j1 = s.node_index["J1"]
    t = r.time_axis("s")
    voor = res[t < t_aank - 2 * dt, j1, 0]
    na = res[t > t_aank + 2 * dt, j1, 0]
    assert np.all(voor == 0.0)
    assert np.allclose(na, 1.0, atol=1e-9)
    assert s.mass_balance()["balance_error"][0] < 1e-9


# ── V2 ────────────────────────────────────────────────────────────────────────
def test_serie_massabehoud(tmp_path):
    inp = _inp(tmp_path, _NET_SERIE)
    for dt in (5.0, 2.5):
        s, r, res = _run(inp, _bron(R1=1.0), 900.0, dt=dt)
        assert s.mass_balance()["balance_error"][0] < 1e-9


# ── V3 ────────────────────────────────────────────────────────────────────────
def test_mengpunt_stationair_is_debietgewogen_gemiddelde(tmp_path):
    inp = _inp(tmp_path, _NET_MIX)
    s, r, res = _run(inp, _bron(R1=1.0, R2=3.0), 900.0)
    flow, _ = s._get_hydraulics()
    q = np.abs(flow)
    verwacht = (q[0] * 1.0 + q[1] * 3.0) / (q[0] + q[1])
    for knoop in ("J1", "J2"):
        assert res[-1, s.node_index[knoop], 0] == pytest.approx(verwacht, rel=1e-9)


# ── V4 ────────────────────────────────────────────────────────────────────────
def _mix_run(inp, gevuld):
    s = NzingaFlowSolver(inp, n_species=1, track_mass=True)
    if gevuld:   # leidingen vooraf gevuld met water van concentratie 0 (zoals [QUALITY] GLOBAL)
        s.set_initial_quality(np.zeros((len(s.node_index), 1)), dt=5.0)
    r = EPSRunner(s, qual_dt=5.0, hyd_dt=300.0, duration=900.0)
    r.run(decay_k=[0.0], inject_schedule=_bron(R1=1.0, R2=1.0), check_cfl=False)
    return s.mass_balance()


def test_mengpunt_massabehoud_met_gevulde_leidingen(tmp_path):
    """Met beginwater in de leidingen blijft de massa behouden (< 0,1 %)."""
    mb = _mix_run(_inp(tmp_path, _NET_MIX), gevuld=True)
    assert mb["balance_error"][0] < 1e-3
    assert mb["ok"]


def test_mengpunt_lege_leidingen_verliest_massa_bekende_beperking(tmp_path):
    """
    Zonder beginwater (lege leidingen) verdunt het mengpunt de aankomende
    stof met 'water' dat niet bestaat: de knoop deelt door Σ Q, maar de nog
    niet gevulde leiding levert geen segmenten. Dat kost een vaste hoeveelheid
    massa (~0,07 m³·C hier, constant na het front). Vul de leidingen dus altijd
    met set_initial_quality(). Deze test legt het gedrag vast; slaagt hij niet
    meer (verlies ~0), dan is de beperking opgelost.
    """
    mb = _mix_run(_inp(tmp_path, _NET_MIX), gevuld=False)
    verlies = float(mb["injected"][0] - mb["accounted"][0])
    assert 0.01 < verlies < 0.1


# ── V5 ────────────────────────────────────────────────────────────────────────
def test_open_afsluiter_laat_stof_door(tmp_path):
    inp = _inp(tmp_path, _NET_VALVE_OPEN)
    s, r, res = _run(inp, _bron(R1=1.0), 3000.0,
                     include_valves=True, skip_short_valves=False)
    for knoop in ("J1", "J2", "J3"):
        assert res[-1, s.node_index[knoop], 0] == pytest.approx(1.0, abs=1e-9)


def test_gesloten_afsluiter_blokkeert_stof(tmp_path):
    inp = _inp(tmp_path, _NET_VALVE_CLOSED)
    hm = HydraulicModel(inp, include_valves=True)
    hm.solve()
    assert hm.get_closed_mask().any()
    s, r, res = _run(inp, _bron(R1=1.0), 3000.0,
                     include_valves=True, skip_short_valves=False)
    assert res[-1, s.node_index["J1"], 0] == pytest.approx(1.0, abs=1e-9)
    assert np.all(res[:, s.node_index["J2"], 0] == 0.0)
    assert np.all(res[:, s.node_index["J3"], 0] == 0.0)


# ── V6 ────────────────────────────────────────────────────────────────────────
def test_stroomomkering_volgt_nieuwe_aanvoer(tmp_path):
    inp = _inp(tmp_path, _NET_REV)
    s, r, res = _run(inp, _bron(R1=1.0, R2=3.0), 3000.0, hyd_dt=600.0)
    j = s.node_index["J1"]
    t = r.time_axis("s")
    # t < 1200 s: J1 krijgt water uit R1; t > 1200 s (+ reistijd): uit R2.
    assert np.allclose(res[(t > 60) & (t < 1150), j, 0], 1.0, atol=1e-9)
    assert np.allclose(res[t > 1300, j, 0], 3.0, atol=1e-9)


# ── V7 ────────────────────────────────────────────────────────────────────────
def test_mengpunt_komt_overeen_met_epanet(tmp_path):
    inp = _inp(tmp_path, _NET_MIX)
    knopen = ("J1", "J2")
    s, r, res = _run(inp, _bron(R1=1.0, R2=3.0), 900.0)
    t_ep, q_ep = _epanet_tijdreeks(inp, {"R1": 1.0, "R2": 3.0}, knopen, 900.0)
    t_nf = r.time_axis("s")
    afwijkend = 0
    for kk, knoop in enumerate(knopen):
        nf = np.interp(t_ep, t_nf, res[:, s.node_index[knoop], 0])
        afwijkend += int(np.sum(np.abs(nf - q_ep[:, kk]) > 1e-6))
        # stationaire waarde: precies gelijk aan EPANET
        assert res[-1, s.node_index[knoop], 0] == pytest.approx(q_ep[-1, kk], rel=1e-6)
    # Buiten de frontovergangen is de tijdreeks gelijk aan EPANET: hooguit een
    # paar kwaliteitstijdstappen (gemeten: 2 per front, 6 in totaal van 362).
    assert afwijkend <= 8, f"{afwijkend} afwijkende punten t.o.v. EPANET"
