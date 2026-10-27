"""
Optionele pakketten (scipy, pandas) geven een duidelijke foutmelding met de juiste extra.

Zonder deze tests zou een gewone installatie pas bij gebruik falen met een kale
ModuleNotFoundError, zonder aanwijzing welke extra nodig is.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from nzingaflow.msx import MsxReactionSystem
from nzingaflow.msxlibrary import MsxSimulationResult, MsxSpecies


def test_rk45_without_scipy_names_the_msx_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.integrate", None)
    rxn = MsxReactionSystem(["A"], pipe_rates={"A": "-0.1 * A"}, solver="rk45")

    with pytest.raises(ImportError, match=r"nzingaflow\[msx\]"):
        rxn._integrate(np.array([1.0]), 1.0, lambda t, y: -0.1 * y)


@pytest.mark.parametrize("solver", ["euler", "rk4", "ros2"])
def test_solvers_without_scipy_dependency_keep_working(monkeypatch, solver):
    monkeypatch.setitem(sys.modules, "scipy", None)
    monkeypatch.setitem(sys.modules, "scipy.integrate", None)
    rxn = MsxReactionSystem(["A"], pipe_rates={"A": "-0.1 * A"}, solver=solver)

    y = rxn._integrate(np.array([1.0]), 1.0, lambda t, y: -0.1 * y)

    assert y[0] == pytest.approx(np.exp(-0.1), rel=0.06)


def test_to_dataframe_without_pandas_names_the_pandas_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "pandas", None)
    species = MsxSpecies(index=1, name="CL2", kind=0, units="mg/L", atol=1e-8, rtol=1e-4)
    result = MsxSimulationResult(
        time_s=np.array([0.0, 3600.0]),
        node_quality=np.zeros((2, 1, 1)),
        link_quality=np.zeros((2, 1, 1)),
        species=[species],
        node_names=["N1"],
        link_names=["L1"],
    )

    with pytest.raises(ImportError, match=r"nzingaflow\[pandas\]"):
        result.to_dataframe("CL2")
