"""
Bronnen op junctions met negatieve vraag (externe instroom) in de automatische modus.

J1  Junction met negatieve vraag en CONCEN-bron levert bronconcentratie (EPANET-gedrag)
J2  Die instroom bereikt het netwerk stroomafwaarts; concentratie gelijk aan EPANET
J3  Zonder automatische modus blijft de instroom op de standaard (0)
J4  Junction zonder [SOURCES]-regel houdt het oude gedrag (set_source_quality werkt)
"""

from __future__ import annotations

import numpy as np
import pytest

from nzingaflow import NzingaFlowSolver

NET = """[TITLE]
negatieve vraag met bron
[OPTIONS]
Units              LPS
Headloss           H-W
Quality            Chemical mg/L
[JUNCTIONS]
J1  0  -1.0
J2  0   1.0
[RESERVOIRS]
R0  50
[PIPES]
P1  J1  J2  100  150  100  0  Open
P2  R0  J2  100  150  100  0  Open
[SOURCES]
J1  CONCEN  2.0
[COORDINATES]
J1  0 0
J2  1 0
R0  2 0
[TIMES]
Duration           1:30
Hydraulic Timestep 0:05
Quality Timestep   0:05
Pattern Timestep   0:05
[END]
"""


def _solver(tmp_path, text=NET, auto=True):
    p = tmp_path / "junc.inp"
    p.write_text(text, encoding="utf-8")
    s = NzingaFlowSolver(str(p), n_species=1, auto_reservoir_sources=auto)
    init = np.zeros((s.node_count, 1))
    init[s.node_index["J1"], 0] = 5.0  # zelfde beginkwaliteit als in de EPANET-test
    s.set_initial_quality(init)
    return s


def _series(s, steps=18):
    rows = []
    for k in range(steps):
        s.update_hydraulics(simtime=k * 300)
        C = s.step(dt=300.0, decay_k=np.array([0.0]))
        rows.append((float(C[s.node_index["J1"], 0]), float(C[s.node_index["J2"], 0])))
    return rows


def test_j1_source_on_negative_demand_junction(tmp_path):
    rows = _series(_solver(tmp_path))
    j1 = [r[0] for r in rows]
    assert all(v == pytest.approx(2.0) for v in j1)


def test_j2_downstream_reaches_source_concentration(tmp_path):
    rows = _series(_solver(tmp_path))
    assert rows[0][1] == pytest.approx(0.0, abs=1e-9)  # eerst nog geen aankomst
    assert rows[6][1] == pytest.approx(2.0, abs=1e-3)  # t = 2100 s, EPANET: 2.0


def test_j3_disabled_gives_zero_instroom(tmp_path):
    rows = _series(_solver(tmp_path, auto=False))
    assert all(r[0] == pytest.approx(0.0) for r in rows)


def test_j4_without_source_set_source_quality_still_works(tmp_path):
    net = NET.replace("[SOURCES]\nJ1  CONCEN  2.0\n", "")
    s = _solver(tmp_path, text=net)
    s.set_source_quality("J1", np.array([3.0]))
    rows = _series(s)
    assert rows[-1][0] == pytest.approx(3.0)
