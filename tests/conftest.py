# tests/conftest.py
"""
Gedeelde pytest fixtures voor NzingaFlow validatietests.

Fixtures
--------
simple_pipe_network
    Miniatuurnetwerk in geheugen: 1 leiding, 2 knopen.
    Bruikbaar voor unit-tests zonder epynet.

segment_store
    Lege SegmentStore met standaard capaciteit en 2 stoffen.
"""
from __future__ import annotations
import numpy as np
import pytest


@pytest.fixture
def segment_store():
    """Lege SegmentStore (capacity=1000, n_species=2)."""
    from nzingaflow import SegmentStore
    return SegmentStore(capacity=1000, n_species=2)


@pytest.fixture
def simple_network_arrays():
    """
    Minimaal netwerk als numpy-arrays (geen epynet vereist).

    Topologie: R1 --[pipe 0]--> J1 --[pipe 1]--> J2
    """
    pipe_start  = np.array([0, 1], dtype=np.int32)
    pipe_end    = np.array([1, 2], dtype=np.int32)
    pipe_length = np.array([500.0, 500.0])
    pipe_area   = np.array([np.pi * 0.05**2, np.pi * 0.05**2])
    velocity    = np.array([0.5, 0.5])
    flow        = velocity * pipe_area
    node_count  = 3
    return {
        "pipe_start":  pipe_start,
        "pipe_end":    pipe_end,
        "pipe_length": pipe_length,
        "pipe_area":   pipe_area,
        "velocity":    velocity,
        "flow":        flow,
        "node_count":  node_count,
        "n_pipes":     2,
    }
