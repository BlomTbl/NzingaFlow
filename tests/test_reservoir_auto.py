"""
Automatische reservoirbronnen in NzingaFlowSolver (vereist epynet).

A1  Actief reservoir met bron voedt het netwerk; een uitgeschakeld reservoir (head 0) niet
A2  auto_reservoir_sources=False voedt niets automatisch
A3  Automatisch gelijk aan handmatige inject() met hetzelfde volume
A4  MASS-bron op een actief reservoir wordt geweigerd met een duidelijke fout
"""

from __future__ import annotations

import numpy as np
import pytest

from nzingaflow import NzingaFlowSolver

NET = """[TITLE]
reservoir test
[OPTIONS]
Units              LPS
Headloss           H-W
Quality            Chemical mg/L
[JUNCTIONS]
J1  0  1.0
[RESERVOIRS]
R_on   100
R_off  0
[PIPES]
P1  R_on  J1  100  200  100  0  Open
P2  R_off J1  100  200  100  0  Open
[SOURCES]
R_on  CONCEN  1.0
R_off CONCEN  5.0
[COORDINATES]
J1      0 0
R_on    1 0
R_off   -1 0
[TIMES]
Duration           2:00
Hydraulic Timestep 0:05
Quality Timestep   0:05
Pattern Timestep   0:05
[END]
"""


def _inp(tmp_path, text=NET):
    p = tmp_path / "res.inp"
    p.write_text(text, encoding="utf-8")
    return str(p)


def _j1_conc(solver, steps=24):
    out = None
    for k in range(steps):
        t = k * 300.0
        solver.update_hydraulics(simtime=int(t))
        node_C = solver.step(dt=300.0, decay_k=np.array([0.0]))
        out = node_C
    return float(out[solver.node_index["J1"], 0])


def test_a1_active_source_fed_inactive_not(tmp_path):
    s = NzingaFlowSolver(_inp(tmp_path), n_species=1, auto_reservoir_sources=True)
    s.set_initial_quality(np.zeros((s.node_count, 1)))
    c = _j1_conc(s, steps=48)
    assert c > 0.5  # R_on levert 1.0 mg/L
    assert c <= 1.0 + 1e-6  # R_off (5.0) mag nooit meetellen


def test_a2_disabled_feeds_nothing(tmp_path):
    s = NzingaFlowSolver(_inp(tmp_path), n_species=1, auto_reservoir_sources=False)
    s.set_initial_quality(np.zeros((s.node_count, 1)))
    assert _j1_conc(s, steps=24) == pytest.approx(0.0, abs=1e-12)


def test_a3_auto_equals_manual_inject(tmp_path):
    inp = _inp(tmp_path)
    auto = NzingaFlowSolver(inp, n_species=1, auto_reservoir_sources=True)
    auto.set_initial_quality(np.zeros((auto.node_count, 1)))
    manual = NzingaFlowSolver(inp, n_species=1, auto_reservoir_sources=False)
    manual.set_initial_quality(np.zeros((manual.node_count, 1)))

    def feed(sol, t):
        flow, _ = sol._get_hydraulics()
        ri = sol.node_index["R_on"]
        vol = float(flow[sol.pipe_start == ri].sum()) * 300.0
        if vol > 0:
            sol.inject("R_on", np.array([1.0]), vol)

    c_auto = _j1_conc(auto, steps=24)
    c_manual = None
    for k in range(24):
        t = k * 300.0
        manual.update_hydraulics(simtime=int(t))
        feed(manual, t)
        out = manual.step(dt=300.0, decay_k=np.array([0.0]))
        c_manual = float(out[manual.node_index["J1"], 0])
    assert c_auto == pytest.approx(c_manual, rel=1e-9, abs=1e-12)


def test_a4_mass_source_on_active_reservoir_rejected(tmp_path):
    bad = NET.replace("R_on  CONCEN  1.0", "R_on  MASS  1.0")
    with pytest.raises(NotImplementedError, match="MASS"):
        NzingaFlowSolver(_inp(tmp_path, bad), n_species=1, auto_reservoir_sources=True)


def test_a5_manual_inject_on_auto_reservoir_warns(tmp_path):
    s = NzingaFlowSolver(_inp(tmp_path), n_species=1, auto_reservoir_sources=True)
    s.set_initial_quality(np.zeros((s.node_count, 1)))
    s.update_hydraulics(simtime=0)
    with pytest.warns(UserWarning, match="R_on"):
        s.inject("R_on", np.array([1.0]), 0.1)


# ── Regressietests na verificatie tegen EPANET 2.3.3 ──────────────────────────
_NET_REV = """[TITLE]
reservoir keert om
[OPTIONS]
Units              LPS
Headloss           H-W
Quality            Chemical mg/L
[JUNCTIONS]
J1 0 0.0
[RESERVOIRS]
R1 50
R2 30 HP
[PIPES]
P1 R1 J1 100 150 100 0 Open
P2 J1 R2 100 150 100 0 Open
[QUALITY]
R1 2.0
R2 0.0
J1 0.0
[PATTERNS]
HP 1.0 1.0 1.0 1.0 1.0 1.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0 3.0
[COORDINATES]
R1 0 0
J1 1 0
R2 2 0
[TIMES]
Duration           2:00
Hydraulic Timestep 0:05
Quality Timestep   0:05
Pattern Timestep   0:10
[END]
"""

_NET_PSTART = """[TITLE]
patroonstart
[OPTIONS]
Units              LPS
Headloss           H-W
Quality            Chemical mg/L
[JUNCTIONS]
J1 0 5.0
[RESERVOIRS]
R0 50
[PIPES]
P1 R0 J1 100 150 100 0 Open
[SOURCES]
R0 CONCEN 1.0 PAT
[PATTERNS]
PAT 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0
[COORDINATES]
R0 0 0
J1 1 0
[TIMES]
Duration           1:40
Hydraulic Timestep 0:05
Quality Timestep   0:05
Pattern Timestep   0:05
Pattern Start      0:10
[END]
"""


def _series(s, names, steps, start=0):
    out = []
    for k in range(steps):
        s.update_hydraulics(simtime=start + k * 300)
        C = s.step(dt=300.0, decay_k=np.array([0.0]))
        out.append({n: float(C[s.node_index[n], 0]) for n in names})
    return out


def test_a6_reservoir_without_source_keeps_initial_quality_after_reversal(tmp_path):
    """EPANET: R2 (beginkwaliteit 0) levert na de omkering water van 0, ook al
    ontving het eerst water van 2,0. Zelfde uitkomst als EPANET 2.3.3."""
    s = NzingaFlowSolver(_inp(tmp_path, _NET_REV), n_species=1, auto_reservoir_sources=True)
    init = np.zeros((s.node_count, 1))
    init[s.node_index["R1"], 0] = 2.0
    s.set_initial_quality(init)
    rows = _series(s, ["J1", "R2"], 24)
    assert rows[11]["J1"] == pytest.approx(2.0, abs=1e-6)  # t = 3600 s: nog stroming R1 -> R2
    assert rows[14]["J1"] == pytest.approx(0.0, abs=1e-6)  # t = 4500 s: omgekeerd, EPANET: 0
    assert rows[23]["J1"] == pytest.approx(0.0, abs=1e-6)
    assert all(r["R2"] == pytest.approx(0.0, abs=1e-12) for r in rows)  # gerapporteerd als EPANET


def test_a7_pattern_start_matches_epanet(tmp_path):
    """Pattern Start 0:10: het patroon begint bij de derde waarde (EPANET 2.3.3)."""
    s = NzingaFlowSolver(_inp(tmp_path, _NET_PSTART), n_species=1, auto_reservoir_sources=True)
    s.set_initial_quality(np.zeros((s.node_count, 1)))
    rows = _series(s, ["J1"], 6)
    epanet = {1: 2.4657, 2: 3.8219, 3: 4.8219, 4: 5.8219}  # index k = einde stap k+1
    for k, ref in epanet.items():
        assert rows[k]["J1"] == pytest.approx(ref, abs=0.25)


def test_a8_quality_clock_follows_hydraulic_time(tmp_path):
    s = NzingaFlowSolver(_inp(tmp_path, _NET_PSTART), n_species=1, auto_reservoir_sources=True)
    s.set_initial_quality(np.zeros((s.node_count, 1)), simtime=900)
    assert s._qual_clock == pytest.approx(900.0)
    s.update_hydraulics(simtime=1200)
    assert s._qual_clock == pytest.approx(1200.0)
    s.step(dt=300.0, decay_k=np.array([0.0]))
    assert s._qual_clock == pytest.approx(1500.0)
