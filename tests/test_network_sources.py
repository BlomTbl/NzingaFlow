"""
Bronnen, reservoirs en patronen via epynet (vereist epynet).

N1  Alleen reservoirs met head > 0 komen in de lijst
N2  Bronsoorten: CONCEN geeft sterkte en patroon, geen bron geeft None
N3  Patroonwaarden en patroonfactor (herhalend, met start en stap)
N4  Junctions met bron komen in de junctionlijst, reservoirs niet
"""

from __future__ import annotations

import pytest

epynet_network = pytest.importorskip("epynet.network")

from nzingaflow.network_sources import (  # noqa: E402
    junction_sources,
    pattern_multiplier,
    pattern_timing,
    pattern_values,
    reservoir_sources,
)

INP = """[TITLE]
test
[JUNCTIONS]
J1 0 0.001
J2 0 0.0
[RESERVOIRS]
R_on 100
R_off 0
R_plain 80
[TANKS]
[PIPES]
P1 R_on J1 100 200 100 0 Open
P2 R_plain J2 100 200 100 0 Open
P3 R_off J2 100 200 100 0 Open
[COORDINATES]
[SOURCES]
R_on CONCEN 2.5 PAT_A
R_off CONCEN 9.0
J2 CONCEN 4.0
[PATTERNS]
PAT_A 1.0 2.0
PAT_A 3.0
[TIMES]
Pattern Timestep  0:05
Pattern Start     0:00
[OPTIONS]
Units LPS
Headloss D-W
[END]
"""


@pytest.fixture
def net(tmp_path):
    p = tmp_path / "net.inp"
    p.write_text(INP, encoding="utf-8")
    return epynet_network.Network(
        str(p), rpt_file=str(tmp_path / "r.rpt"), out_file=str(tmp_path / "o.out")
    )


def test_n1_only_active_reservoirs(net):
    names = [r.name for r in reservoir_sources(net)]
    assert names == ["R_on", "R_plain"]  # R_off (head 0) uitgesloten


def test_n2_source_kinds(net):
    by_name = {r.name: r for r in reservoir_sources(net)}
    assert by_name["R_on"].kind == "CONCEN"
    assert by_name["R_on"].strength == pytest.approx(2.5)
    assert by_name["R_on"].pattern == "PAT_A"
    assert by_name["R_plain"].kind is None
    assert by_name["R_plain"].strength is None


def test_n3_pattern_values_and_multiplier(net):
    vals = pattern_values(net, "PAT_A")
    assert vals == pytest.approx([1.0, 2.0, 3.0])
    step, start = pattern_timing(net)
    assert (step, start) == (300, 0)
    assert pattern_multiplier(vals, 0, step, start) == 1.0
    assert pattern_multiplier(vals, 300, step, start) == 2.0
    assert pattern_multiplier(vals, 900, step, start) == 1.0  # herhaalt


def test_n4_junction_sources(net):
    names = [j.name for j in junction_sources(net)]
    assert names == ["J2"]  # reservoirs horen niet in deze lijst
