# nzingaflow/__init__.py
"""
NzingaFlow — Lagrangian Transport kwaliteitssimulator voor EPANET-netwerken.

Kernklassen
-----------
NzingaFlowSolver
    Vectorized LTA-solver; koppeling met EPANET via epynet.
    Ondersteunt multi-species, wandreacties, tanks en (optioneel) volledige
    geochemie via PhreeqPython (GeochemSolver).

EPSRunner
    Voert een Extended Period Simulation uit: hydraulische updates,
    tijdvariabele injectie en voortgangsrapportage.

GeochemSolver / SpeciesMap
    PhreeqPython-integratie; vervangt lineair eerste-orde verval door
    volledige PHREEQC-geochemie (kinetiek + evenwichtsfasen + pH-buffering).

HydraulicModel
    Lage-level koppeling met epynet; levert topologie en hydraulica als
    numpy-arrays.

Snelstart
---------
    import numpy as np
    from nzingaflow import NzingaFlowSolver, EPSRunner

    solver = NzingaFlowSolver("netwerk.inp", n_species=1)
    runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)

    results = runner.run(
        decay_k=np.array([0.0]),
        inject_schedule={"R1": [(0, 86400, np.array([1.0]))]},
        verbose=True,
    )
    # results.shape == (n_stappen, node_count, 1)

Met geochemie (PhreeqPython):
    from nzingaflow import NzingaFlowSolver, EPSRunner
    from nzingaflow.geochemistry import GeochemSolver, SpeciesMap, full_water_chemistry

    geo = full_water_chemistry()
    solver = NzingaFlowSolver("netwerk.inp", n_species=6, geochem=geo)
    runner = EPSRunner(solver, qual_dt=10.0, hyd_dt=300.0, duration=86400.0)
    results = runner.run(decay_k=np.zeros(6), ...)
"""

from .eps import EPSRunner
from .geochemistry import (
    GeochemSolver,
    PhreeqSolutionMode,
    SpeciesMap,
    chlorine_decay_geochem,
    full_water_chemistry,
)
from .hydraulics import HydraulicModel
from .lta import (
    USE_NUMBA,
    advect,
    build_combined_exp,
    bulk_first_order_multi,
    combined_decay_multi,
    compute_wall_k,
    exit_detect,
    node_mixing_multi,
    tank_step_implicit,
    wall_first_order_multi,
    warmup_numba,
)
from .merging import merge_segments, warmup_numba_merging
from .msx import (
    MsxReactionSystem,
    arsenic_oxidation_msx,
    chloramine_decay_msx,
    chlorine_nom_msx,
)
from .msxlibrary import (
    MsxError,
    MsxNativeLib,
    MsxNetworkState,
    MsxSimulation,
    MsxSimulationResult,
    MsxSourceRecord,
    MsxSpecies,
    run_msx,
)
from .segments import SegmentStore
from .solver import NzingaFlowSolver
from .stability import MassBalanceTracker, check_dt, recommended_dt

__version__ = "1.3.1"
__author__ = "NzingaFlow"

__all__ = [
    # Hoofd-API
    "NzingaFlowSolver",
    "warmup_numba",
    "USE_NUMBA",
    "warmup_numba_merging",
    "combined_decay_multi",
    "build_combined_exp",
    "EPSRunner",
    # Geochemie (PhreeqPython)
    "GeochemSolver",
    "SpeciesMap",
    "chlorine_decay_geochem",
    "full_water_chemistry",
    "PhreeqSolutionMode",
    # MSX multi-species reactielaag (pure Python)
    "MsxReactionSystem",
    "chloramine_decay_msx",
    "chlorine_nom_msx",
    "arsenic_oxidation_msx",
    # MSX native library bridge (ctypes → libepanetmsx)
    "MsxNativeLib",
    "MsxNetworkState",
    "MsxSimulation",
    "MsxSimulationResult",
    "MsxSpecies",
    "MsxSourceRecord",
    "MsxError",
    "run_msx",
    # Bouwstenen
    "HydraulicModel",
    "SegmentStore",
    # LTA-kernfuncties
    "bulk_first_order_multi",
    "wall_first_order_multi",
    "compute_wall_k",
    "advect",
    "exit_detect",
    "node_mixing_multi",
    "tank_step_implicit",
    "merge_segments",
    # Stabiliteit
    "recommended_dt",
    "check_dt",
    "MassBalanceTracker",
]

# Vaste PHREEQC-invoer (decimaalteken), zie locale_guard.py
from . import locale_guard  # noqa: F401
