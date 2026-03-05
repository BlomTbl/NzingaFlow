# inzingaflow/__init__.py
"""
InzingaFlow — Lagrangian Transport kwaliteitssimulator voor EPANET-netwerken.

Kernklassen
-----------
InzingaFlowSolver
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
    from inzingaflow import InzingaFlowSolver, EPSRunner

    solver = InzingaFlowSolver("netwerk.inp", n_species=1)
    runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)

    results = runner.run(
        decay_k=np.array([0.0]),
        inject_schedule={"R1": [(0, 86400, np.array([1.0]))]},
        verbose=True,
    )
    # results.shape == (n_stappen, node_count, 1)

Met geochemie (PhreeqPython):
    from inzingaflow import InzingaFlowSolver, EPSRunner
    from inzingaflow.geochemistry import GeochemSolver, SpeciesMap, full_water_chemistry

    geo = full_water_chemistry()
    solver = InzingaFlowSolver("netwerk.inp", n_species=6, geochem=geo)
    runner = EPSRunner(solver, qual_dt=10.0, hyd_dt=300.0, duration=86400.0)
    results = runner.run(decay_k=np.zeros(6), ...)
"""

from .solver      import InzingaFlowSolver
from .eps         import EPSRunner
from .hydraulics  import HydraulicModel
from .segments    import SegmentStore
from .stability   import recommended_dt, check_dt, MassBalanceTracker
from .lta         import (
    bulk_first_order_multi,
    wall_first_order_multi,
    compute_wall_k,
    advect,
    node_mixing_multi,
    tank_step_implicit,
)
from .merging     import merge_segments
from .geochemistry import GeochemSolver, SpeciesMap, chlorine_decay_geochem, full_water_chemistry

__version__ = "1.0.0"
__author__  = "InzingaFlow"

__all__ = [
    # Hoofd-API
    "InzingaFlowSolver",
    "EPSRunner",
    # Geochemie
    "GeochemSolver",
    "SpeciesMap",
    "chlorine_decay_geochem",
    "full_water_chemistry",
    # Bouwstenen
    "HydraulicModel",
    "SegmentStore",
    # LTA-kernfuncties
    "bulk_first_order_multi",
    "wall_first_order_multi",
    "compute_wall_k",
    "advect",
    "node_mixing_multi",
    "tank_step_implicit",
    "merge_segments",
    # Stabiliteit
    "recommended_dt",
    "check_dt",
    "MassBalanceTracker",
]
