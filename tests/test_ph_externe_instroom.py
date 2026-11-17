"""
PhreeqSolutionMode en externe instroom (negatieve vraag).

E1  Externe instroom draagt de ingestelde oplossing: pH op de injectieknoop == PHREEQC-mix
    van de twee instromen naar debietsverhouding.
E2  Het vrijgegeven water krijgt de gemengde oplossing: de downstream knoop heeft dezelfde pH.
E3  Zonder set_source_solution() wordt gewaarschuwd en de achtergrondoplossing gebruikt.
"""

from __future__ import annotations

import numpy as np
import pytest

phreeqpython = pytest.importorskip("phreeqpython")

from nzingaflow import NzingaFlowSolver  # noqa: E402
from nzingaflow.geochemistry import PhreeqSolutionMode  # noqa: E402

INP = """[TITLE]
externe instroom test
[JUNCTIONS]
;ID Elev Demand Pattern
J1 0 -10
J2 0 20
[RESERVOIRS]
;ID Head
R1 100
[TANKS]
[PIPES]
;ID N1 N2 Len Dia Rough MinorLoss Status
P1 R1 J1 300 4 100 0 Open
P2 J1 J2 300 4 100 0 Open
[PUMPS]
[VALVES]
[EMITTERS]
[QUALITY]
[SOURCES]
[REACTIONS]
[TIMES]
Duration 0:00
[OPTIONS]
Units GPM
Headloss H-W
Specific Gravity 1.0
Viscosity 1.0
Trials 40
Accuracy 0.001
Quality None
Unbalanced Continue
Pattern 1
Demand Multiplier 1.0
[REPORT]
Status No
[COORDINATES]
J1 0 0
J2 0 0
R1 0 0
[END]
"""

A = {"pH": 7.0, "temp": 15, "units": "mmol/L", "Alkalinity": 3.0, "Na": 2.0, "Cl": 1.0, "Ca": 0.5}
B = {"pH": 8.2, "temp": 15, "units": "mmol/L", "Alkalinity": 1.0, "Na": 1.0, "Cl": 2.0, "Ca": 1.5}
DT, DUUR = 60.0, 3 * 3600


@pytest.fixture
def net_path(tmp_path):
    p = tmp_path / "ext.inp"
    p.write_text(INP, encoding="utf-8")
    return str(p)


def _run(net_path, pp, psm, s, sol_ext=None):
    """Simuleer; geeft de pH-reeks op J1 en J2 (laatste stap) terug."""
    s.set_initial_quality(np.zeros((s.node_count, 1)), simtime=0, dt=DT)
    psm.fill(s.segments, psm.background_number)
    ph = {"J1": None, "J2": None}
    for st in range(int(DUUR / DT)):
        if st % 5 == 0:
            s.update_hydraulics(simtime=int(st * DT))
        flow, _ = s._get_hydraulics()
        outs = s._node_outpipes.get(s.node_index["R1"], [])
        vol = float(sum(flow[p] for p in outs)) * DT
        if vol > 0:
            n0 = s.segments.n
            s.inject("R1", np.zeros(1), vol)
            for idx in range(n0, s.segments.n):
                psm.on_segment_added(idx, psm_src_r1)
        if sol_ext is not None:
            s.set_source_solution("J1", sol_ext)
        s.step(
            dt=DT,
            decay_k=np.zeros(1),
            merge_interval=10**9,
            merge_tol=1e-6,
            merge_rel_tol=0.0,
            check_cfl=False,
            merge_span=1.0,
        )
        sols = psm._node_solutions
        for k in ph:
            if s.node_index[k] in sols:
                ph[k] = pp.get_solution(sols[s.node_index[k]]).pH
    return ph


def _fractie_extern(net_path):
    """Debietsfractie van de externe instroom op J1, uit de conservatieve tracer."""
    s = NzingaFlowSolver(net_path, n_species=1, include_valves=True, auto_reservoir_sources=False)
    s.set_initial_quality(np.zeros((s.node_count, 1)), simtime=0, dt=DT)
    C = None
    for st in range(int(DUUR / DT)):
        if st % 5 == 0:
            s.update_hydraulics(simtime=int(st * DT))
        flow, _ = s._get_hydraulics()
        outs = s._node_outpipes.get(s.node_index["R1"], [])
        vol = float(sum(flow[p] for p in outs)) * DT
        if vol > 0:
            s.inject("R1", np.array([1.0]), vol)  # R1-water = 1, extern = 0
        C = s.step(
            dt=DT,
            decay_k=np.zeros(1),
            merge_interval=10,
            merge_tol=1e-6,
            merge_rel_tol=0.0,
            check_cfl=False,
            merge_span=1.0,
        )
    return float(C[s.node_index["J1"], 0])


psm_src_r1 = None


def test_e1_externe_instroom_mengt_met_zijn_oplossing(net_path):
    global psm_src_r1
    pp = phreeqpython.PhreeqPython()
    sA, sB = pp.add_solution(A), pp.add_solution(B)
    psm_src_r1 = sA.number
    # achtergrond = A (niet B), zodat het verschil tussen de externe oplossing en de
    # achtergrond zichtbaar is: zonder set_source_solution zou J1 dan pH 7,0 geven
    psm = PhreeqSolutionMode(pp, background_number=sA.number)
    s = NzingaFlowSolver(
        net_path, n_species=1, include_valves=True, auto_reservoir_sources=False, geochem=psm
    )
    ph = _run(net_path, pp, psm, s, sol_ext=sB.number)

    f_r1 = _fractie_extern(net_path)  # aandeel R1-water op J1 (tracer)
    ref = pp.mix_solutions({sA: f_r1, sB: 1 - f_r1}).pH
    assert ph["J1"] == pytest.approx(ref, abs=1e-3)


def test_e2_vrijgegeven_water_krijgt_gemengde_oplossing(net_path):
    global psm_src_r1
    pp = phreeqpython.PhreeqPython()
    sA, sB = pp.add_solution(A), pp.add_solution(B)
    psm_src_r1 = sA.number
    psm = PhreeqSolutionMode(pp, background_number=sA.number)
    s = NzingaFlowSolver(
        net_path, n_species=1, include_valves=True, auto_reservoir_sources=False, geochem=psm
    )
    ph = _run(net_path, pp, psm, s, sol_ext=sB.number)
    assert ph["J2"] is not None
    assert ph["J2"] == pytest.approx(ph["J1"], abs=1e-3)


def test_e3_zonder_bronoplossing_waarschuwing_en_achtergrond(net_path):
    global psm_src_r1
    pp = phreeqpython.PhreeqPython()
    sA, sB = pp.add_solution(A), pp.add_solution(B)
    psm_src_r1 = sA.number
    psm = PhreeqSolutionMode(pp, background_number=sB.number)
    s = NzingaFlowSolver(
        net_path, n_species=1, include_valves=True, auto_reservoir_sources=False, geochem=psm
    )
    with pytest.warns(UserWarning, match="set_source_solution"):
        _run(net_path, pp, psm, s, sol_ext=None)
