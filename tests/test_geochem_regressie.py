"""
Regressietests voor de PHREEQC-route (vereist phreeqpython, echte PHREEQC).

R1  Oxidatietoestand in de stofnaam ('C(4)') geeft geen nul meer terug bij het lezen.
R2  Geen ionen uit de achtergrond: een carbonaatmengsel geeft dezelfde pH als PHREEQC direct.
R3  Overzadigd calciet slaat neer (desaturate), tot het SI-doel, naar de PHREEQC-waarde.
R4  Kinetiek (RATES/KINETICS) werkt en geeft eerste-orde verval, geen crash.
R5  Ongeldige mappings (Alkalinity, meq voor Ca) geven een duidelijke fout.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

phreeqpython = pytest.importorskip("phreeqpython")

from nzingaflow.geochemistry import (GeochemSolver, SpeciesMap,
                                     DEFAULT_BACKGROUND, chlorine_decay_geochem)
from nzingaflow.segments import SegmentStore


def _direct_ph(alk_meq, ct_mmol, ca_mmol=0.0):
    pp = phreeqpython.PhreeqPython()
    ca = f" Ca {ca_mmol} mmol/kgw\n" if ca_mmol else ""
    pp.ip.run_string(f"SOLUTION 1\n units mmol/kgw\n pH 7.5\n pe 4\n temp 15\n"
                     f" Alkalinity {alk_meq} meq/kgw\n C(4) {ct_mmol} mmol/kgw\n{ca}SAVE SOLUTION 1\nEND\n")
    return pp.get_solution(1)


# R1 ─────────────────────────────────────────────────────────────────────────

def test_r1_readback_c4_keeps_carbon():
    smap = SpeciesMap(species_names=["Alk", "CT", "pH"], phreeqc_names=["Alk", "C(4)", None],
                      units=["meq/L", "mmol/L", ""], is_pH=[False, False, True])
    g = GeochemSolver(smap)
    node = np.array([[2.0, 1.5, 7.0]])
    g.apply_mixing(node, np.array([1.0]), dt=60.0)
    assert node[0, 1] == pytest.approx(1.5, rel=1e-4)      # vroeger 0


# R2 ─────────────────────────────────────────────────────────────────────────

def test_r2_default_background_adds_no_ions():
    assert not any(k in DEFAULT_BACKGROUND for k in ("Ca", "Mg", "Na", "Cl", "S(6)", "Alkalinity"))


def test_r2_mixing_ph_matches_direct_phreeqc():
    smap = SpeciesMap(species_names=["Alk", "CT", "pH"], phreeqc_names=["Alk", "C(4)", None],
                      units=["meq/L", "mmol/L", ""], is_pH=[False, False, True])
    g = GeochemSolver(smap)
    node = np.array([[3.0, 2.75, 7.5]])
    g.apply_mixing(node, np.array([1.0]), dt=60.0)
    ref = _direct_ph(3.0, 2.75)
    assert node[0, 2] == pytest.approx(ref.pH, abs=1e-3)
    assert node[0, 1] == pytest.approx(2.75, rel=1e-4)


# R3 ─────────────────────────────────────────────────────────────────────────

def test_r3_supersaturated_calcite_precipitates_like_phreeqc():
    pp = phreeqpython.PhreeqPython()
    pp.ip.run_string("SOLUTION 1\n units mmol/kgw\n temp 15\n pH 8.5\n pe 4\n"
                     " Alkalinity 3.0 meq/kgw\n C(4) 3.0 mmol/kgw\n Ca 2.0 mmol/kgw\n"
                     "EQUILIBRIUM_PHASES 1\n Calcite 0.0 0.0\nSAVE SOLUTION 1\nEND\n")
    truth = pp.get_solution(1)
    smap = SpeciesMap(species_names=["Ca", "Alk", "CT", "pH"],
                      phreeqc_names=["Ca", "Alk", "C(4)", None],
                      units=["mmol/L", "meq/L", "mmol/L", ""], is_pH=[False, False, False, True])
    st = SegmentStore(capacity=3, n_species=4)
    st.add(0, 0.0, 1.0, np.array([2.0, 3.0, 3.0, 8.5]))
    g = GeochemSolver(smap, equilibrium_phases={"Calcite": (0.0, 0.0)})
    g.apply_geochemistry(st, dt=60.0)
    assert st.C[0, 0] == pytest.approx(truth.total_element("Ca"), abs=0.01)
    assert st.C[0, 3] == pytest.approx(truth.pH, abs=0.01)
    assert st.C[0, 0] < 2.0 - 0.1                      # er is echt neergeslagen


def test_r3_undersaturated_still_dissolves():
    smap = SpeciesMap(species_names=["Ca", "Alk", "CT", "pH"],
                      phreeqc_names=["Ca", "Alk", "C(4)", None],
                      units=["mmol/L", "meq/L", "mmol/L", ""], is_pH=[False, False, False, True])
    st = SegmentStore(capacity=3, n_species=4)
    st.add(0, 0.0, 1.0, np.array([0.1, 3.0, 3.0, 7.0]))   # ver onder verzadiging
    g = GeochemSolver(smap, equilibrium_phases={"Calcite": (0.0, 0.0)})
    g.apply_geochemistry(st, dt=60.0)
    assert st.C[0, 0] > 0.1                               # calciet lost op


# R4 ─────────────────────────────────────────────────────────────────────────

def test_r4_chlorine_decay_is_first_order():
    k_day = 0.5
    geo = chlorine_decay_geochem(k_bulk_per_day=k_day)
    st = SegmentStore(capacity=3, n_species=2)
    st.add(0, 0.0, 1.0, np.array([1.0, 7.5]))
    dt = 3600.0
    geo.apply_geochemistry(st, dt=dt)
    expected = 1.0 * math.exp(-k_day / 86400.0 * dt)
    assert st.C[0, 0] == pytest.approx(expected, rel=2e-3)


def test_r4_rates_without_save_is_a_clear_error():
    smap = SpeciesMap(species_names=["Cl"], phreeqc_names=["Cl"], units=["mg/L"], is_pH=[False])
    g = GeochemSolver(smap, kinetics_script="RATES\nX\n-start\n10 rate = PARM(1)*TOT(\"Cl\")\n-end\nKINETICS 1\nX\n-formula Cl -1\n-m 1.0\n-parms 1e-5\n-steps {dt_s} seconds\nEND\n")
    st = SegmentStore(capacity=3, n_species=1)
    st.add(0, 0.0, 1.0, np.array([35.45]))
    with pytest.raises(ValueError, match="SAVE"):
        g.apply_geochemistry(st, dt=60.0)


def test_r4_kinetics_crash_is_gone_for_mol_placeholders():
    smap = SpeciesMap(species_names=["Cl"], phreeqc_names=["Cl"], units=["mg/L"], is_pH=[False])
    g = GeochemSolver(smap, kinetics_script="RATES\nX\n-start\n10 rate = PARM(1)*TOT(\"Cl\")\n"
                      "20 SAVE rate * TIME\n-end\nKINETICS 1\nX\n-formula Cl -1\n-m 1.0\n-parms {dt_d}\n"
                      "-steps {dt_s} seconds\n-step_divide 10\nEND\n")
    st = SegmentStore(capacity=3, n_species=1)
    st.add(0, 0.0, 1.0, np.array([35.45]))
    g.apply_geochemistry(st, dt=60.0)     # mag niet crashen
    assert st.C[0, 0] >= 0.0


# R5 ─────────────────────────────────────────────────────────────────────────

def test_r5_alkalinity_name_is_rejected():
    with pytest.raises(ValueError, match="Alk"):
        SpeciesMap(species_names=["A"], phreeqc_names=["Alkalinity"], units=["meq/L"])


def test_r5_meq_for_non_alk_is_rejected():
    with pytest.raises(ValueError, match="meq"):
        SpeciesMap(species_names=["Ca"], phreeqc_names=["Ca"], units=["meq/L"])
