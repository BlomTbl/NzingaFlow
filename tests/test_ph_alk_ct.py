"""
pH uit (ALK, CT) met PHREEQC: ph_from_alk_ct() (vereist phreeqpython).

P1  Waarde gelijk aan directe PHREEQC-aanroep, met achtergrondionen.
P2  Ongeldige invoer geeft NaN; vorm blijft behouden.
P3  Dubbele paren worden één keer berekend (zelfde uitkomst).
"""

from __future__ import annotations

import numpy as np
import pytest

phreeqpython = pytest.importorskip("phreeqpython")

from nzingaflow.geochemistry import ph_from_alk_ct

BG = {"Na": 1.74, "Ca": 1.01, "Mg": 0.20, "Cl": 0.98, "S(6)": 0.11, "N(5)": 0.111}


def _direct(alk, ct):
    pp = phreeqpython.PhreeqPython()
    d = {
        "temp": "15",
        "pe": "4",
        "units": "mmol/kgw",
        "Alkalinity": f"{alk} meq/kgw",
        "C(4)": f"{ct} mmol/kgw",
    }
    d.update({k: f"{v} mmol/kgw" for k, v in BG.items()})
    s = pp.add_solution(d)
    return s.pH


@pytest.mark.parametrize("alk, ct", [(3.06, 3.13), (2.0, 2.5), (0.05, 1.0), (4.0, 2.0)])
def test_p1_matches_direct_phreeqc(alk, ct):
    assert float(ph_from_alk_ct(alk, ct, background=BG)) == pytest.approx(
        _direct(alk, ct), abs=1e-3
    )


def test_p2_invalid_input_is_nan_and_shape_kept():
    alk = np.array([[3.0, -1.0], [2.0, 3.0]])
    ct = np.array([[3.0, 3.0], [0.0, 3.0]])
    out = ph_from_alk_ct(alk, ct, background=BG)
    assert out.shape == (2, 2)
    assert np.isnan(out[0, 1]) and np.isnan(out[1, 0])
    assert np.isfinite(out[0, 0]) and np.isfinite(out[1, 1])


def test_p3_duplicate_pairs_same_result():
    alk = np.array([3.06, 3.06, 2.0])
    ct = np.array([3.13, 3.13, 2.5])
    out = ph_from_alk_ct(alk, ct, background=BG)
    assert out[0] == out[1]
    assert out[0] == pytest.approx(_direct(3.06, 3.13), abs=1e-3)
