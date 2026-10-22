"""
Passief overslaan in GeochemSolver (geen epynet en geen PHREEQC vereist).

P1  has_reactions is alleen True als er een kinetiek-, evenwichts- of reactiescript is
P2  Zonder reacties: apply_geochemistry roept PHREEQC niet aan
P3  Met een kinetiek-script: apply_geochemistry verwerkt de segmenten wel
P4  apply_mixing blijft werken (initialiseert PHREEQC) ook zonder reacties
"""
from __future__ import annotations

import numpy as np

from nzingaflow.segments import SegmentStore
from nzingaflow.geochemistry import GeochemSolver, SpeciesMap


def _solver(**kwargs) -> GeochemSolver:
    smap = SpeciesMap(
        species_names=["Cl2", "pH"],
        phreeqc_names=["Cl", None],
        units=["mg/L", ""],
        is_pH=[False, True],
    )
    return GeochemSolver(smap, **kwargs)


def _store_with_one_segment() -> SegmentStore:
    st = SegmentStore(capacity=100, n_species=2)
    st.add(0, 0.0, 1.0, np.array([5.0, 7.0]))
    return st


def _counters(g: GeochemSolver, monkeypatch) -> dict:
    calls = {"init": 0, "batch": 0}

    def _count_init():
        calls["init"] += 1

    def _count_batch(*args, **kwargs):
        calls["batch"] += 1

    monkeypatch.setattr(g, "_init_pp", _count_init)
    monkeypatch.setattr(g, "_process_batch", _count_batch)
    return calls


def test_p1_has_reactions_flags():
    assert _solver().has_reactions is False
    assert _solver(kinetics_script="KINETICS 1\nEND").has_reactions is True
    assert _solver(equilibrium_phases={"Calcite": 0.0}).has_reactions is True
    assert _solver(reaction_script="REACTION 1\nEND").has_reactions is True


def test_p2_passive_skips_phreeqc(monkeypatch):
    g = _solver()
    calls = _counters(g, monkeypatch)
    g.apply_geochemistry(_store_with_one_segment(), dt=300.0)
    assert calls == {"init": 0, "batch": 0}


def test_p3_reactions_still_processed(monkeypatch):
    g = _solver(kinetics_script="KINETICS 1\nEND")
    calls = _counters(g, monkeypatch)
    g.apply_geochemistry(_store_with_one_segment(), dt=300.0)
    assert calls["batch"] == 1


def test_p4_mixing_still_initialises_phreeqc(monkeypatch):
    g = _solver()
    calls = _counters(g, monkeypatch)
    # node_flow = 0 -> geen knopen om te mengen; alleen de initialisatie telt
    g.apply_mixing(np.zeros((1, 2)), np.zeros(1), dt=300.0)
    assert calls["init"] == 1
