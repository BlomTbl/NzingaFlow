"""
EPSRunner.run(record_every=n): alleen elke n-de kwaliteitsstap bewaren (vereist epynet).

E1  Standaard (1) geeft elke stap, tijdas ongewijzigd
E2  record_every=n geeft exact dezelfde rijen als elke n-de rij van de volledige run
E3  time_axis() volgt record_every
E4  Niet-deelbaar aantal stappen: de laatste, onvolledige reeks wordt niet bewaard
E5  Ongeldige waarden geven een ValueError
"""
from __future__ import annotations

import numpy as np
import pytest

from nzingaflow import NzingaFlowSolver
from nzingaflow.eps import EPSRunner

NET = """[TITLE]
record every
[OPTIONS]
Units              LPS
Headloss           H-W
Quality            Chemical mg/L
[JUNCTIONS]
J1  0  1.0
[RESERVOIRS]
R1  100
[PIPES]
P1  R1  J1  100  200  100  0  Open
[SOURCES]
R1  CONCEN  1.0  PAT
[PATTERNS]
PAT 1.0 2.0 3.0 4.0 5.0 6.0
[COORDINATES]
J1  0 0
R1  1 0
[TIMES]
Duration           2:00
Hydraulic Timestep 0:05
Quality Timestep   0:05
Pattern Timestep   0:05
[END]
"""


def _run(tmp_path, **kw):
    p = tmp_path / "rec.inp"
    p.write_text(NET, encoding="utf-8")
    s = NzingaFlowSolver(str(p), n_species=1)
    s.set_initial_quality(np.zeros((s.node_count, 1)))
    r = EPSRunner(s, qual_dt=300.0, hyd_dt=300.0, duration=7200.0)       # 24 stappen
    res = r.run(decay_k=[0.0], check_cfl=False, **kw)
    return r, res


def test_e1_default_records_every_step(tmp_path):
    r, res = _run(tmp_path)
    assert res.shape[0] == 24
    assert r.time_axis("s")[0] == 300.0 and r.time_axis("s")[-1] == 7200.0


def test_e2_rows_equal_every_nth_row(tmp_path):
    _, full = _run(tmp_path)
    _, thin = _run(tmp_path, record_every=6)
    assert thin.shape[0] == 4
    for k in range(4):
        np.testing.assert_array_equal(thin[k], full[6 * (k + 1) - 1])


def test_e3_time_axis_follows_record_every(tmp_path):
    r, thin = _run(tmp_path, record_every=6)
    np.testing.assert_allclose(r.time_axis("s"), [1800.0, 3600.0, 5400.0, 7200.0])
    assert len(r.time_axis("s")) == thin.shape[0]


def test_e4_non_divisible_drops_incomplete_tail(tmp_path):
    r, thin = _run(tmp_path, record_every=5)          # 24 stappen: 4 rijen (na stap 5, 10, 15, 20)
    assert thin.shape[0] == 4
    np.testing.assert_allclose(r.time_axis("s"), [1500.0, 3000.0, 4500.0, 6000.0])


@pytest.mark.parametrize("bad", [0, -1, 2.5])
def test_e5_invalid_record_every(tmp_path, bad):
    with pytest.raises(ValueError, match="record_every"):
        _run(tmp_path, record_every=bad)


def test_e6_merge_span_reaches_solver_step(tmp_path):
    p = tmp_path / "rec.inp"
    p.write_text(NET, encoding="utf-8")
    s = NzingaFlowSolver(str(p), n_species=1)
    s.set_initial_quality(np.zeros((s.node_count, 1)))
    seen = []
    orig = s.step
    s.step = lambda *a, **k: (seen.append(k.get("merge_span", "ontbreekt")), orig(*a, **k))[1]
    r = EPSRunner(s, qual_dt=300.0, hyd_dt=300.0, duration=1200.0)
    r.run(decay_k=[0.0], check_cfl=False)                       # standaard
    r.run(decay_k=[0.0], check_cfl=False, merge_span=None)      # geen beperking
    assert seen[:4] == [1.0] * 4 and seen[4:] == [None] * 4
