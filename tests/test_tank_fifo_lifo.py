"""
FIFO- en LIFO-tanks (EPANET [MIXING]).

T1  FIFO: het oudste water gaat eerst uit. Het beginwater (C = 0) verlaat de tank eerst;
    later komt het water dat na t = 0 is ingestroomd (C = 1) naar buiten.
T2  LIFO: het nieuwste water gaat eerst uit. Na t = 1 h (instroom C = 0) gaat er direct C = 0 uit.
T3  MIXED (CSTR) geeft daartussen een geleidelijke overgang.
T4  Volume-balans: parcels_eind = V0 + instroom − uitstroom.
T5  PhreeqSolutionMode: uitgaand water heeft de oplossing van het parcel dat vertrekt
    (FIFO: beginoplossing eerst; LIFO: nieuwste oplossing eerst).
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from nzingaflow import NzingaFlowSolver

INP = """[TITLE]
FIFO/LIFO test
[JUNCTIONS]
;ID Elev Demand Pattern
J1 0 0
J2 0 5
[RESERVOIRS]
;ID Head
R1 30
[TANKS]
;ID Elev InitLevel MinLevel MaxLevel Diameter MinVol VolCurve
T1 0 1 0 10 5 0
[PIPES]
;ID N1 N2 Len Dia Rough MinorLoss Status
P1 R1 T1 300 300 100 0 Open
P2 T1 J1 300 300 100 0 Open
P3 J1 J2 300 300 100 0 Open
[MIXING]
;Tank Model
T1 {model}
[PUMPS]
[VALVES]
[EMITTERS]
[QUALITY]
[SOURCES]
[REACTIONS]
[TIMES]
Duration 4:00
Hydraulic Timestep 0:05
Quality Timestep 0:01
[OPTIONS]
Units LPS
Headloss H-W
Specific Gravity 1.0
Viscosity 1.0
Trials 40
Accuracy 0.001
Quality None
Unbalanced Continue
Pattern 1
Demand Multiplier 1.0
[REPORT]
Status No
[COORDINATES]
J1 0 0
J2 0 0
R1 0 0
T1 0 0
[END]
"""

DT = 60.0
DUUR = 4 * 3600
# Voor FIFO staat er vóór het ingestroomde water (C = 1) eerst beginwater (C = 0):
# het tankvolume (18 m³) plus het water in de leiding P1 (ca. 21 m³) bij 5 L/s uitstroom,
# dus het C = 1-water verlaat de tank pas rond 2,2 h en is er tot ca. 3,2 h.
V_INIT = 18.0  # beginvolume tank [m³]
SWITCH = 3600.0  # instroom C = 1 (of sol A) vóór dit tijdstip, daarna C = 0 (sol B)


class Recorder(NzingaFlowSolver):
    """Legt per stap vast wat de tank afgeeft (volume, concentratie, oplossing)."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.t_now = 0.0
        self.log: list[tuple[float, float, float, float]] = []  # (t, vol, C, pH)

    def _release_tanks(self, flow, dt):
        pieces = self._tank_release
        vol = sum(item[1] for item in pieces)
        if vol > 0.0:
            c = sum(item[1] * float(item[2][0]) for item in pieces) / vol
            ph = float("nan")
            if self._geochem is not None and hasattr(self._geochem, "sol_ids") and pieces:
                ph = _pp_ph(self._geochem, pieces)
            self.log.append((self.t_now, vol, c, ph))
        super()._release_tanks(flow, dt)


_PP = None


def _pp_ph(geo, pieces) -> float:
    total = sum(item[1] for item in pieces)
    return sum(item[1] * geo.pp.get_solution(item[3]).pH for item in pieces) / total


def _model_file(tmp_path, model):
    p = tmp_path / f"tank_{model}.inp"
    p.write_text(INP.replace("{model}", model), encoding="utf-8")
    return str(p)


def _run(net_path, *, geochem=None, sol_in=None, sol_bg=None, extra_hook=None):
    """Simuleer 3 h. Instroom uit R1: C = 1 (of sol_in[0]) vóór SWITCH, daarna C = 0 (of sol_in[1])."""
    s = Recorder(
        net_path, n_species=1, include_valves=True, auto_reservoir_sources=False, geochem=geochem
    )
    s.set_initial_quality(np.zeros((s.node_count, 1)), simtime=0, dt=DT)
    if geochem is not None:
        geochem.fill(s.segments, sol_bg)
    # Beginvolume van de tank: 18 m³ beginwater (C = 0), direct gezet. Het tankvolume dat
    # EPANET bij init rapporteert hangt af van de eerste hydraulische stap; dat is hier niet
    # het onderwerp. Bij 5 L/s uitstroom verlaat dit volume de tank in precies 1 uur.
    if s._tank_models[0] in ("FIFO", "LIFO"):
        sol0 = sol_bg if geochem is not None else None
        s._tank_parcels[0] = deque([[V_INIT, np.zeros(1), sol0]])
        s._tank_V[0] = V_INIT
    v0 = V_INIT
    in_vol_total = 0.0
    for st in range(int(DUUR / DT)):
        t = st * DT
        s.t_now = t
        if st % 5 == 0:
            s.update_hydraulics(simtime=int(t))
        flow, _ = s._get_hydraulics()
        # instroom = uitstroom van de tank (vaste demand), zodat het tankvolume constant blijft
        vol = float(flow[s.pipe_index["P2"]]) * DT
        if vol > 0.0:
            n0 = s.segments.n
            c = np.array([1.0 if t < SWITCH else 0.0])
            s.inject("R1", c, vol)
            in_vol_total += vol
            if geochem is not None:
                num = sol_in[0] if t < SWITCH else sol_in[1]
                for idx in range(n0, s.segments.n):
                    geochem.on_segment_added(idx, num)
        s.step(
            dt=DT,
            decay_k=np.zeros(1),
            merge_interval=10**9,
            merge_tol=1e-6,
            merge_rel_tol=0.0,
            check_cfl=False,
            merge_span=1.0,
        )
    s.v0 = v0
    return s, in_vol_total


def _c_at(log, t_target, half_width=600.0):
    """Gemiddelde uitgaande concentratie rond t_target (± half_width s)."""
    vals = [c for t, v, c, _ in log if abs(t - t_target) <= half_width and v > 0]
    return float(np.mean(vals)) if vals else float("nan")


def test_t1_fifo_oudste_water_eerst(tmp_path):
    s, _ = _run(_model_file(tmp_path, "FIFO"))
    assert _c_at(s.log, 0.5 * 3600) < 0.1  # beginwater en leidingwater (C = 0) gaan eerst eruit
    assert _c_at(s.log, 2.7 * 3600) > 0.9  # daarna het water van t < 1 h (C = 1)
    assert _c_at(s.log, 3.7 * 3600) < 0.1  # daarna weer C = 0


def test_t2_lifo_nieuwste_water_eerst(tmp_path):
    s, _ = _run(_model_file(tmp_path, "LIFO"))
    assert _c_at(s.log, 0.5 * 3600) > 0.9  # nieuwste water (C = 1) gaat eerst eruit
    assert _c_at(s.log, 1.5 * 3600) < 0.1  # na de omschakeling: nieuwste is C = 0
    assert _c_at(s.log, 3.5 * 3600) < 0.1


def test_t3_mixed_geleidelijk(tmp_path):
    s, _ = _run(_model_file(tmp_path, "MIXED"))
    c = _c_at(s.log, 1.5 * 3600)
    assert 0.05 < c < 0.95


def test_t4_volume_balans(tmp_path):
    s, _ = _run(_model_file(tmp_path, "FIFO"))
    v_end = float(sum(p[0] for p in s._tank_parcels[0]))
    # V0 + in − uit = V_eind (uit de tellers van de solver, niet uit de tankafgifte)
    assert s.v0 > 0.0
    assert np.isclose(v_end, s.v0 + s._parcel_in[0] - s._parcel_out[0], rtol=1e-9)
    # de afgifte in het log is precies wat uit de parcels is gehaald
    assert np.isclose(sum(v for _, v, _, _ in s.log), s._parcel_out[0], rtol=1e-9)


def test_t5_psm_oplossing_per_parcel(tmp_path):
    pytest.importorskip("phreeqpython")
    import phreeqpython

    from nzingaflow.geochemistry import PhreeqSolutionMode

    pp = phreeqpython.PhreeqPython()
    A = {
        "pH": 7.0,
        "temp": 15,
        "units": "mmol/L",
        "Alkalinity": 3.0,
        "Na": 2.0,
        "Cl": 1.0,
        "Ca": 0.5,
    }
    B = {
        "pH": 8.2,
        "temp": 15,
        "units": "mmol/L",
        "Alkalinity": 1.0,
        "Na": 1.0,
        "Cl": 2.0,
        "Ca": 1.5,
    }
    sA, sB = pp.add_solution(A).number, pp.add_solution(B).number
    psm = PhreeqSolutionMode(pp, auto_gc_interval=0, background_number=sB)
    # Beginwater = B, instroom eerst A, daarna B
    s, _ = _run(_model_file(tmp_path, "FIFO"), geochem=psm, sol_in=(sA, sB), sol_bg=sB)
    assert _pH_at(s.log, 1.0 * 3600) == pytest.approx(8.2, abs=0.05)  # FIFO: beginwater (B) eerst
    assert _pH_at(s.log, 2.7 * 3600) == pytest.approx(7.0, abs=0.05)  # daarna het water van A
    assert _pH_at(s.log, 3.7 * 3600) == pytest.approx(8.2, abs=0.05)  # en weer B

    psm2 = PhreeqSolutionMode(pp, auto_gc_interval=0, background_number=sB)
    s2, _ = _run(_model_file(tmp_path, "LIFO"), geochem=psm2, sol_in=(sA, sB), sol_bg=sB)
    assert _pH_at(s2.log, 0.5 * 3600) == pytest.approx(7.0, abs=0.05)  # LIFO: nieuwste (A) eerst
    assert _pH_at(s2.log, 1.5 * 3600) == pytest.approx(8.2, abs=0.05)  # daarna B


def _pH_at(log, t_target, half_width=600.0):
    vals = [ph for t, v, c, ph in log if abs(t - t_target) <= half_width and v > 0 and ph == ph]
    return float(np.mean(vals)) if vals else float("nan")
