"""
Vaste PHREEQC-invoer, ongeacht de systeemlocale.

PHREEQC leest getallen met het decimaalteken van de actieve locale. Op
Nederlandse Windows-installaties is dat een komma, waardoor pH 7.91 als 7.0
wordt ingelezen. Deze module zet de locale voor elke PHREEQC-aanroep tijdelijk
op 'C' en herstelt die daarna. Importeer deze module eenmaal; daarna geldt het
overal, ook voor pp.add_solution() rechtstreeks in een notebook.
"""

import contextlib
import locale

from phreeqpython import viphreeqc as _vip


@contextlib.contextmanager
def _c_numeric():
    oud = locale.setlocale(locale.LC_NUMERIC)
    try:
        locale.setlocale(locale.LC_NUMERIC, "C")
        yield
    finally:
        locale.setlocale(locale.LC_NUMERIC, oud)


if not getattr(_vip.VIPhreeqc, "_locale_guard", False):
    _oorspronkelijk = _vip.VIPhreeqc.run_string

    def _run_string_c(self, cmd_string):
        with _c_numeric():
            return _oorspronkelijk(self, cmd_string)

    _vip.VIPhreeqc.run_string = _run_string_c
    _vip.VIPhreeqc._locale_guard = True
