"""
Foutafhandeling: geen stille terugvalwaarden.

F1  Tankvolume niet uitleesbaar        -> RuntimeError met tanknaam (niet stil 1000 m³)
F2  Afsluiterstatus niet uitleesbaar   -> RuntimeError (niet stil 'open')
F3  Bronkwaliteit niet uitleesbaar     -> RuntimeWarning met reservoir-id
F4  Mislukte hydraulische berekening   -> RuntimeWarning (niet stil debiet 0)
"""
from __future__ import annotations
import warnings

import pytest

from nzingaflow import NzingaFlowSolver
from nzingaflow.hydraulics import HydraulicModel
from nzingaflow.parse_inp import load_from_epynet

from .test_epynet_networks import tank_inp, valve_inp  # noqa: F401  (fixtures)

pytestmark = pytest.mark.requires_epynet


def test_tankvolume_fout_geeft_runtimeerror(tank_inp, monkeypatch):
    solver = NzingaFlowSolver(tank_inp, n_species=1)

    def kapot(*a, **k):
        raise ValueError("kapot")

    monkeypatch.setattr("nzingaflow.units.volume_to_m3", kapot)
    with pytest.raises(RuntimeError, match="T1"):
        solver._get_tank_volumes()


def test_afsluiterstatus_fout_geeft_runtimeerror(valve_inp, monkeypatch):
    hm = HydraulicModel(valve_inp, include_valves=True)
    hm.solve()
    # sanity: zonder fout werkt het
    assert hm.get_closed_mask().dtype == bool

    klep = hm._get_links()[hm.get_valve_mask().nonzero()[0][0]]

    def kapot(self):
        raise ValueError("status onleesbaar")

    monkeypatch.setattr(type(klep), "status", property(kapot))
    with pytest.raises(RuntimeError, match="afsluiter"):
        hm.get_closed_mask()


def test_bronkwaliteit_fout_geeft_waarschuwing(valve_inp, monkeypatch):
    hm = HydraulicModel(valve_inp, include_valves=True)
    hm.solve()
    net = hm.net
    orig = net.EN_getnodevalue


    from nzingaflow import units as u
    prop = u.EN_NodeProperty.EN_SOURCEQUAL

    def stuk2(idx, p):
        if p == prop:
            raise ValueError("bron onleesbaar")
        return orig(idx, p)

    monkeypatch.setattr(net, "EN_getnodevalue", stuk2)
    with pytest.warns(RuntimeWarning, match="Bronkwaliteit"):
        load_from_epynet(net)


def test_mislukte_solve_geeft_waarschuwing(valve_inp, monkeypatch):
    hm = HydraulicModel(valve_inp, include_valves=True)
    hm.solve()

    import nzingaflow.hydraulics as hyd

    class Kapot:
        def __init__(self, net): pass
        def solve_step(self, **k): raise ValueError("solve faalt")
        def close(self): pass

    monkeypatch.setattr(hyd, "_NoSaveHydraulicSolver", Kapot)
    with pytest.warns(RuntimeWarning, match="Hydraulische berekening"):
        load_from_epynet(hm.net)


def test_reservoir_zonder_bron_geeft_geen_waarschuwing(valve_inp):
    """Geen [SOURCES]-regel (EPANET-fout 240) is normaal en blijft stil."""
    hm = HydraulicModel(valve_inp, include_valves=True)
    hm.solve()
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        load_from_epynet(hm.net)
