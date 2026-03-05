# tests/test_inzingaflow.py
"""
Unit tests voor InzingaFlow — geen epynet/EPANET vereist.

Alle tests draaien op interne numpy-logica zodat CI werkt
zonder EPANET-installatie.
"""

import numpy as np
import pytest

from inzingaflow import (
    SegmentStore,
    bulk_first_order_multi,
    wall_first_order_multi,
    advect,
    node_mixing_multi,
    tank_step_implicit,
    merge_segments,
    recommended_dt,
    check_dt,
    MassBalanceTracker,
)
from inzingaflow.geochemistry import SpeciesMap, GeochemSolver


# ── SegmentStore ──────────────────────────────────────────────────────────────

class TestSegmentStore:

    def test_add_and_retrieve(self):
        s = SegmentStore(capacity=10, n_species=2)
        s.add(pipe=0, x=1.0, volume=0.5, C_vector=np.array([1.0, 2.0]))
        assert s.n == 1
        assert s.pipe[0] == 0
        assert s.x[0] == 1.0
        np.testing.assert_array_equal(s.C[0], [1.0, 2.0])

    def test_resize_no_wrap(self):
        """Resize moet data correct kopiëren zonder np.resize-wrapping."""
        s = SegmentStore(capacity=4, n_species=2)
        for i in range(4):
            s.add(pipe=i, x=float(i), volume=1.0, C_vector=np.array([float(i), float(i) * 2]))
        # Trigger resize
        s.add(pipe=99, x=99.0, volume=1.0, C_vector=np.array([99.0, 198.0]))
        assert s._capacity == 8
        assert s.n == 5
        for i in range(4):
            assert s.pipe[i] == i, f"pipe[{i}] corrupt na resize"
            assert s.x[i] == float(i)
            assert s.C[i, 0] == float(i)
        assert s.pipe[4] == 99

    def test_remove_swap_with_last(self):
        s = SegmentStore(capacity=5, n_species=1)
        for i in range(4):
            s.add(pipe=i, x=float(i), volume=1.0, C_vector=np.array([float(i)]))
        s.remove([1])   # verwijder index 1 → last (3) wordt naar 1 gekopieerd
        assert s.n == 3
        pipes = sorted(s.pipe[:s.n].tolist())
        assert pipes == [0, 2, 3]

    def test_active_properties(self):
        s = SegmentStore(capacity=5, n_species=2)
        s.add(pipe=0, x=1.0, volume=2.0, C_vector=np.array([3.0, 4.0]))
        np.testing.assert_array_equal(s.pipe_a, [0])
        np.testing.assert_array_equal(s.x_a, [1.0])
        np.testing.assert_array_equal(s.volume_a, [2.0])

    def test_len(self):
        s = SegmentStore(capacity=10, n_species=1)
        assert len(s) == 0
        s.add(pipe=0, x=0.0, volume=1.0, C_vector=np.array([1.0]))
        assert len(s) == 1


# ── LTA kernfuncties ─────────────────────────────────────────────────────────

class TestBulkDecay:

    def test_no_decay(self):
        C = np.array([[1.0, 2.0], [3.0, 4.0]])
        C0 = C.copy()
        bulk_first_order_multi(C, np.array([0.0, 0.0]), dt=60.0)
        np.testing.assert_array_almost_equal(C, C0)

    def test_first_order(self):
        C = np.array([[1.0]])
        k = np.array([0.001])
        dt = 100.0
        bulk_first_order_multi(C, k, dt)
        expected = np.exp(-0.001 * 100.0)
        assert abs(C[0, 0] - expected) < 1e-10

    def test_multi_species_independent(self):
        """Elke stof vervalt onafhankelijk met eigen k."""
        C = np.ones((3, 2))
        k = np.array([0.01, 0.0])
        dt = 50.0
        bulk_first_order_multi(C, k, dt)
        np.testing.assert_array_almost_equal(C[:, 0], np.full(3, np.exp(-0.01 * 50)))
        np.testing.assert_array_almost_equal(C[:, 1], np.ones(3))

    def test_empty_array(self):
        C = np.empty((0, 2))
        bulk_first_order_multi(C, np.array([0.01, 0.0]), 60.0)  # mag niet crashen


class TestAdvect:

    def test_basic_movement(self):
        x = np.array([0.0, 5.0])
        pipe = np.array([0, 0], dtype=np.int32)
        velocity = np.array([1.0])  # 1 m/s
        advect(x, pipe, velocity, dt=5.0)
        np.testing.assert_array_almost_equal(x, [5.0, 10.0])

    def test_zero_velocity(self):
        x = np.array([3.0, 7.0])
        x0 = x.copy()
        pipe = np.array([0, 0], dtype=np.int32)
        advect(x, pipe, np.array([0.0]), dt=100.0)
        np.testing.assert_array_equal(x, x0)

    def test_multi_pipe(self):
        x = np.array([0.0, 0.0])
        pipe = np.array([0, 1], dtype=np.int32)
        velocity = np.array([1.0, 2.0])
        advect(x, pipe, velocity, dt=3.0)
        np.testing.assert_array_almost_equal(x, [3.0, 6.0])


class TestNodeMixing:

    def test_single_segment_exits(self):
        exit_mask = np.array([True])
        pipe = np.array([0], dtype=np.int32)
        C = np.array([[2.0, 4.0]])
        flow = np.array([0.01])
        pipe_end = np.array([1], dtype=np.int32)
        node_C = node_mixing_multi(exit_mask, pipe, C, flow, pipe_end, node_count=3)
        np.testing.assert_array_almost_equal(node_C[1], [2.0, 4.0])
        np.testing.assert_array_equal(node_C[0], [0.0, 0.0])

    def test_weighted_mixing(self):
        """Twee segmenten met verschillende concentraties → gewogen gemiddelde."""
        exit_mask = np.array([True, True])
        pipe = np.array([0, 1], dtype=np.int32)
        C = np.array([[1.0], [3.0]])
        flow = np.array([1.0, 1.0])
        pipe_end = np.array([2, 2], dtype=np.int32)   # beide naar knoop 2
        node_C = node_mixing_multi(exit_mask, pipe, C, flow, pipe_end, node_count=3)
        # Gewogen gemiddelde: (1*1 + 1*3) / 2 = 2.0
        assert abs(node_C[2, 0] - 2.0) < 1e-10

    def test_no_exits(self):
        exit_mask = np.zeros(5, dtype=bool)
        pipe = np.zeros(5, dtype=np.int32)
        C = np.ones((5, 2))
        flow = np.ones(3)
        pipe_end = np.array([1, 2, 0], dtype=np.int32)
        node_C = node_mixing_multi(exit_mask, pipe, C, flow, pipe_end, node_count=3)
        np.testing.assert_array_equal(node_C, np.zeros((3, 2)))


class TestTankStep:

    def test_steady_state(self):
        """Als Q_in=Q_out en C_in=C_tank, moet C onveranderd blijven."""
        C_tank = np.array([[1.0]])
        tank_step_implicit(
            C_tank,
            Q_in=np.array([0.01]),
            C_in=np.array([[1.0]]),
            Q_out=np.array([0.01]),
            V_tank=np.array([10.0]),
            k_b=np.array([0.0]),
            dt=60.0,
        )
        np.testing.assert_array_almost_equal(C_tank, [[1.0]])

    def test_decay_only(self):
        """Zonder in/uitflow: C(t) = C(0) * exp(-k*dt) bij kleine dt."""
        C_tank = np.array([[1.0]])
        k = 0.001
        dt = 10.0
        tank_step_implicit(
            C_tank,
            Q_in=np.array([0.0]),
            C_in=np.array([[0.0]]),
            Q_out=np.array([0.0]),
            V_tank=np.array([1000.0]),
            k_b=np.array([k]),
            dt=dt,
        )
        expected = 1.0 / (1 + k * dt)   # impliciet Euler
        assert abs(C_tank[0, 0] - expected) < 1e-6


# ── Merging ───────────────────────────────────────────────────────────────────

class TestMerging:

    def test_identical_segments_collapse(self):
        s = SegmentStore(capacity=20, n_species=1)
        for i in range(5):
            s.add(pipe=0, x=float(i), volume=1.0, C_vector=np.array([1.0]))
        n_removed = merge_segments(s, tol=1e-6)
        assert n_removed > 0
        assert s.n < 5

    def test_different_segments_not_merged(self):
        s = SegmentStore(capacity=10, n_species=1)
        s.add(pipe=0, x=0.0, volume=1.0, C_vector=np.array([1.0]))
        s.add(pipe=0, x=1.0, volume=1.0, C_vector=np.array([2.0]))
        n_before = s.n
        merge_segments(s, tol=1e-6)
        assert s.n == n_before

    def test_different_pipes_not_merged(self):
        """Segmenten in verschillende leidingen mogen nooit samengevoegd worden."""
        s = SegmentStore(capacity=10, n_species=1)
        s.add(pipe=0, x=0.0, volume=1.0, C_vector=np.array([1.0]))
        s.add(pipe=1, x=0.0, volume=1.0, C_vector=np.array([1.0]))
        merge_segments(s, tol=1e-6)
        assert s.n == 2

    def test_volume_weighted_average(self):
        """Na merging: volume-gewogen concentratie moet correct zijn."""
        s = SegmentStore(capacity=10, n_species=1)
        s.add(pipe=0, x=0.0, volume=2.0, C_vector=np.array([1.0]))
        s.add(pipe=0, x=1.0, volume=2.0, C_vector=np.array([1.0]))  # identiek → merge
        merge_segments(s, tol=1e-6)
        assert s.n == 1
        assert abs(s.volume[0] - 4.0) < 1e-10


# ── Stabiliteit ───────────────────────────────────────────────────────────────

class TestStability:

    def test_recommended_dt_cfl(self):
        lengths = np.array([100.0, 50.0])
        vels    = np.array([1.0, 2.0])
        k_bulk  = np.array([0.0])
        dt = recommended_dt(lengths, vels, k_bulk)
        dt_cfl = min(100 / 1, 50 / 2)   # 25 s
        assert dt <= dt_cfl

    def test_recommended_dt_never_above_cfl(self):
        """Zeer korte leiding: min_dt mag nooit boven CFL klampen."""
        lengths = np.array([0.05])
        vels    = np.array([2.0])
        k_bulk  = np.array([0.0])
        dt = recommended_dt(lengths, vels, k_bulk, min_dt=1.0)
        dt_cfl = 0.05 / 2.0
        assert dt <= dt_cfl, f"dt={dt:.4f} > dt_CFL={dt_cfl:.4f}: CFL geschonden"

    def test_check_dt_reports_violation(self):
        lengths = np.array([10.0])
        vels    = np.array([1.0])
        k_bulk  = np.array([0.0])
        report = check_dt(dt=20.0, pipe_length=lengths, velocity=vels,
                          k_bulk=k_bulk, warn=False)
        assert not report['cfl_ok']
        assert len(report['violations']) > 0

    def test_check_dt_ok(self):
        lengths = np.array([100.0])
        vels    = np.array([0.5])
        k_bulk  = np.array([0.0])
        report = check_dt(dt=10.0, pipe_length=lengths, velocity=vels,
                          k_bulk=k_bulk, warn=False)
        assert report['cfl_ok']
        assert report['stable']


# ── MassBalanceTracker ────────────────────────────────────────────────────────

class TestMassBalance:

    def test_injection_tracked(self):
        tracker = MassBalanceTracker(n_species=1)
        tracker.record_injection(np.array([2.0]), volume=0.5)
        assert abs(tracker.injected[0] - 1.0) < 1e-10

    def test_report_no_segments(self):
        tracker = MassBalanceTracker(n_species=1)
        tracker.record_injection(np.array([1.0]), volume=1.0)
        s = SegmentStore(capacity=10, n_species=1)
        report = tracker.report(s)
        assert 'balance_error' in report
        assert 'injected' in report


# ── SpeciesMap ────────────────────────────────────────────────────────────────

class TestSpeciesMap:

    def test_basic_construction(self):
        smap = SpeciesMap(
            species_names=['Cl2', 'pH'],
            phreeqc_names=['Cl',  None],
            units        =['mg/L', ''],
            is_pH        =[False,  True],
        )
        assert smap.n_species == 2
        assert smap.ph_index == 1
        assert smap.active_indices == [0]

    def test_default_is_pH(self):
        """is_pH mag weggelaten worden — default is allemaal False."""
        smap = SpeciesMap(
            species_names=['Ca', 'Mg'],
            phreeqc_names=['Ca', 'Mg'],
            units        =['mg/L', 'mg/L'],
        )
        assert smap.ph_index is None
        assert smap.active_indices == [0, 1]

    def test_length_mismatch_raises(self):
        with pytest.raises(AssertionError):
            SpeciesMap(
                species_names=['A', 'B'],
                phreeqc_names=['a'],        # te kort
                units        =['mg/L', 'mg/L'],
            )
