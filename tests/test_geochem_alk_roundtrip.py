"""
Alkaliniteit blijft behouden bij een ronde door PHREEQC (vereist phreeqpython).

Aanleiding: het terugschrijven berekende Alk alleen uit vrije soorten (HCO3-, CO3-2,
OH-, H+). Bij 60 mg/L calcium zit ~2 % van de alkaliniteit in CaHCO3+ en CaCO3, dus
elke ronde verloor alkaliniteit (-9 % na vijf stappen). Nu tellen die complexen mee.

G1  _sol_value('Alk') geeft de ingevoerde alkaliniteit terug, met en zonder calcium
G2  Vijf rondes door PHREEQC laten Alk binnen 0,5 % (was -9 %)
"""
from __future__ import annotations

import numpy as np
import pytest

phreeqpython = pytest.importorskip("phreeqpython")

from nzingaflow.geochemistry import GeochemSolver, SpeciesMap, _sol_value
from nzingaflow.segments import SegmentStore


@pytest.mark.parametrize("ca_mgl", [0.0, 60.0, 120.0])
def test_g1_sol_value_alk_includes_ion_pairs(ca_mgl):
    pp = phreeqpython.PhreeqPython()
    ca = f"  Ca {ca_mgl} mg/L\n" if ca_mgl else ""
    pp.ip.run_string(f"SOLUTION 1\n  units mol/L\n  pH 7.6\n  Alkalinity 3.0 meq/L\n{ca}END\n")
    sol = pp.get_solution(1)
    assert _sol_value(sol, "Alk") == pytest.approx(3.0, rel=0.005)
    sol.forget()


def test_g2_alk_stable_over_repeated_phreeqc_rounds(monkeypatch):
    smap = SpeciesMap(species_names=["Ca", "Alk", "pH"], phreeqc_names=["Ca", "Alk", None],
                      units=["mg/L", "meq/L", ""], is_pH=[False, False, True])
    monkeypatch.setattr(GeochemSolver, "has_reactions", property(lambda self: True))
    st = SegmentStore(capacity=20, n_species=3)
    st.add(0, 0.0, 1.0, np.array([60.0, 3.0, 7.6]))
    g = GeochemSolver(smap)
    for _ in range(5):
        g.apply_geochemistry(st, dt=300.0)
    assert st.C[0, 1] == pytest.approx(3.0, rel=0.005)       # Alk
    assert st.C[0, 2] == pytest.approx(7.6, abs=1e-3)        # pH


# ── Per kg water en exacte alkaliniteit (PHREEQC ALK) ─────────────────────────

import nzingaflow.geochemistry as _gc
from nzingaflow.geochemistry import full_water_chemistry


def _force_phreeqc(monkeypatch):
    monkeypatch.setattr(GeochemSolver, "has_reactions", property(lambda self: True))


def test_g3_unit_mapping_is_per_kg_water():
    smap = SpeciesMap(species_names=["Ca"], phreeqc_names=["Ca"], units=["mg/L"], is_pH=[False])
    g = GeochemSolver(smap)
    val, unit = g._kgw_input("Ca", 60.0, "mg/L")
    assert unit == "mmol/kgw"
    assert val == pytest.approx(60.0 / GeochemSolver._MOLAR_MASS["Ca"])     # onze eigen tabel
    assert g._kgw_input("Alk", 3.0, "meq/L") == (3.0, "meq/kgw")
    assert g._kgw_input("Cl", 0.5, "mmol/L") == (0.5, "mmol/kgw")
    assert g.background["units"] == "mol/kgw"                                # was mol/L


def test_g4_mixing_path_roundtrip_exact():
    smap = SpeciesMap(species_names=["Ca", "Alk", "pH", "Cl"], phreeqc_names=["Ca", "Alk", None, "Cl"],
                      units=["mg/L", "meq/L", "", "mg/L"], is_pH=[False, False, True, False])
    g = GeochemSolver(smap); g._init_pp()
    c = np.array([60.0, 3.0, 7.6, 0.3])
    back = g._read_back(c, g._make_solution(c))
    assert back == pytest.approx(c, rel=1e-5)


def test_g5_alk_exact_with_iron_and_manganese():
    """Met Fe en Mn ontstaan hydroxocomplexen die een soortensom mist; PHREEQC's ALK niet."""
    pp = phreeqpython.PhreeqPython()
    pp.ip.run_string("SOLUTION 1\n  units mol/kgw\n  temp 15\n  pH 7.6\n  pe 4\n  Alkalinity 3.0 meq/kgw\n"
                     "  Ca 1.5 mmol/kgw\n  Fe 0.0100 mmol/kgw\n  Mn 0.0005 mmol/kgw\nSAVE SOLUTION 1\nEND\n")
    exact = _gc._alk_batch(pp, [1])[0]
    assert exact == pytest.approx(3.0, rel=1e-6)
    # terugvaloptie (soortensom, coëfficiënten volgens phreeqc.dat): dicht bij, niet exact
    assert _gc._sol_value(pp.get_solution(1), "Alk") == pytest.approx(3.0, rel=0.001)


def test_g6_alk_fallback_when_phreeqc_alk_fails(monkeypatch):
    def boom(pp, nums):
        raise RuntimeError("geen selected output")
    monkeypatch.setattr(_gc, "_alk_batch", boom)
    pp = phreeqpython.PhreeqPython()
    pp.ip.run_string("SOLUTION 1\n  units mol/kgw\n  pH 7.6\n  Alkalinity 3.0 meq/kgw\nSAVE SOLUTION 1\nEND\n")
    out = _gc._alk_exact(pp, [1], [pp.get_solution(1)])
    assert out[0] == pytest.approx(3.0, rel=0.005)


def test_g7_no_drift_full_chemistry_with_fe_mn(monkeypatch):
    """100 rondes door PHREEQC met Fe en Mn: alle soorten blijven binnen 0,001 %.
    Origineel (per liter, soortensom): na 60 rondes Alk +73 %, Ca +1,7 %, Fe +2,0 %."""
    _force_phreeqc(monkeypatch)
    rng = np.random.default_rng(3)
    st = SegmentStore(capacity=100, n_species=6)
    for i in range(20):
        st.add(i % 5, float(i), 1.0, np.array([rng.uniform(.1, .5), rng.uniform(7.2, 8.0), rng.uniform(2.0, 4.0),
                                               rng.uniform(40, 90), rng.uniform(0, .1), rng.uniform(0, .03)]))
    c0 = st.C[:st.n].copy()
    g = full_water_chemistry(equilibrium_minerals=[])
    for _ in range(100):
        g.apply_geochemistry(st, dt=300.0)
    rel = np.abs(st.C[:st.n] - c0) / np.where(c0 == 0, 1, c0) * 100
    for idx, name in [(0, "Cl2"), (2, "Alk"), (3, "Ca"), (4, "Fe"), (5, "Mn")]:
        assert rel[:, idx].max() < 1e-3, f"{name} drift {rel[:, idx].max():.5f} %"
    assert np.abs(st.C[:st.n, 1] - c0[:, 1]).max() < 1e-3          # pH (afronding op 4 decimalen)
