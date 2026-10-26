"""
Fouten in de geochemie worden zichtbaar gemaakt, niet stilzwijgend weggegooid.

F1  _sol_value('Alk') gooit een fout als de speciatie niet beschikbaar is (was: stil 0)
F2  Een mineraal dat niet berekend kan worden geeft NaN en een waarschuwing (was: weggelaten)
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pytest

from nzingaflow.geochemistry import GeochemSolver, SpeciesMap, _sol_value


class _GeenSpeciatie:
    @property
    def species(self):
        raise ValueError("speciatie weg")


def test_f1_alk_raises_when_species_unavailable():
    with pytest.raises(RuntimeError, match="Alkaliniteit niet te bepalen"):
        _sol_value(_GeenSpeciatie(), "Alk")


class _Speciatie:
    def __init__(self, species):
        self.species = species


def test_f1_alk_sums_species_and_missing_species_count_as_zero():
    sol = _Speciatie({"HCO3-": 0.002})  # mol/L, andere soorten ontbreken
    assert _sol_value(sol, "Alk") == pytest.approx(2.0)


class _SolMetFouten:
    def si(self, mineral):
        if mineral == "Aragonite":
            raise RuntimeError("geen data voor Aragonite")
        return 0.5

    def forget(self):
        pass


def test_f2_failing_mineral_gives_nan_and_warning(monkeypatch, caplog):
    smap = SpeciesMap(species_names=["Ca"], phreeqc_names=["Ca"], units=["mg/L"], is_pH=[False])
    g = GeochemSolver(smap)
    monkeypatch.setattr(GeochemSolver, "_init_pp", lambda self: None)
    monkeypatch.setattr(GeochemSolver, "_make_solution", lambda self, C: _SolMetFouten())

    with caplog.at_level(logging.WARNING, logger="nzingaflow.geochemistry"):
        si = g.saturation_indices(np.array([60.0]))

    assert si["Aragonite"] is not None and math.isnan(si["Aragonite"])
    assert si["Calcite"] == pytest.approx(0.5)
    assert "Aragonite" in caplog.text
