# nzingaflow/units.py
"""
Enige bron voor EPANET-eenheidsconversies en -property-codes.

Vóór dit bestand bestonden er twee, licht verschillende conversietabellen —
één in `hydraulics.py` (gebruikt door `HydraulicModel`, dat via epynet's
live `.flow`/`.velocity`/`.diameter`-properties werkt) en één in
`parse_inp.py` (gebruikt door `load_from_epynet()`, dat rechtstreeks
`EN_getlinkvalue(...)` aanroept). Omdat het twee losse tabellen waren,
liepen ze uiteen zonder dat één van beide "fout leek": beide gaven immers
gewoon een getal terug.

Gevonden verschillen (nu hier gecorrigeerd, één keer):
- **AFD** (acre-feet/dag) stond in **beide** tabellen 1000× te klein
  (`1.42764e-5` i.p.v. `1.42764e-2`). 1 acre-foot = 1233,4818 m³;
  1233,4818 / 86400 s ≈ 0,01427641 m³/s — dus de derde decimaal-orde
  klopte, alleen de macht van 10 niet.
- **MLD** (miljoen liter/dag) stond alleen in `parse_inp.py` 1000× te
  klein (`1/86_400` i.p.v. `1/86.4`) — die tabel vergat de "mega"-factor
  (1 MLD = 1000 m³/dag, niet 1 m³/dag).
- **Snelheid**: `parse_inp.py` paste **nooit** een ft/s→m/s-conversie toe
  op `EN_VELOCITY` voor US-eenheidsnetwerken (CFS/GPM/MGD/IMGD/AFD) —
  `hydraulics.py` deed dat wel, via `get_hydraulic_state()`.
- **Diameter**: `parse_inp.py` vermenigvuldigde altijd met `1e-3` (mm→m),
  ook voor US-eenheden waarin EPANET diameter in **inch** teruggeeft —
  `hydraulics.py` had hier al de juiste SI/US-vertakking.

Beide modules gebruiken nu uitsluitend de functies/tabellen hieronder, dus
een toekomstige correctie hoeft maar op één plek.
"""

from __future__ import annotations
import numpy as np

# Native EPYnetDTD-enums worden hier herexporteerd zodat andere modules
# (`hydraulics.py`, `parse_inp.py`) niet zelf magische ints hoeven te
# spiegelen (zoals `parse_inp.py` vóór deze consolidatie deed — foutgevoelig
# bij een toekomstige EPANET-toolkitwijziging).
from epynet.enum import (
    EN_CountType,
    EN_InitHydOption,
    EN_LinkProperty,
    EN_NodeProperty,
    EN_TimeParameter,
)

__all__ = [
    "EN_CountType", "EN_InitHydOption", "EN_LinkProperty",
    "EN_NodeProperty", "EN_TimeParameter",
    "FLOW_CODE_TO_LABEL", "FLOW_TO_M3S", "US_UNITS",
    "INCH_TO_M", "FT_S_TO_M_S",
    "flow_units_label", "diameter_to_m", "velocity_to_ms", "flow_to_m3s",
]

# ── Flow-eenheden ────────────────────────────────────────────────────────────

# Codes zoals geretourneerd door EN_getflowunits() — vaste EPANET-volgorde.
FLOW_CODE_TO_LABEL: dict[int, str] = {
    0: 'CFS', 1: 'GPM', 2: 'MGD', 3: 'IMGD', 4: 'AFD',
    5: 'LPS', 6: 'LPM', 7: 'MLD', 8: 'CMH', 9: 'CMD',
}

# Conversiefactor flow-eenheid → m³/s. Herleid uit de exacte SI-definities
# (foot = 0.3048 m, US-gallon = 3.785411784 L, imperial gallon = 4.54609 L,
# acre = 43560 ft² = 4046.8564224 m² — alle drie internationaal exact
# vastgelegd, dus geen afgeronde tussenstappen):
#   CFS  : 1 ft³ = 0.3048³ m³                                → 0.0283168466
#   GPM  : 1 US gal/min = 3.785411784 L / 60 s                → 6.30901964e-5
#   MGD  : 1e6 US gal/dag                                     → 0.0438126364
#   IMGD  : 1e6 imperial gal/dag                              → 0.0526167824
#   AFD  : 1 acre-foot/dag = (4046.8564224·0.3048) m³ / 86400 → 0.0142764102
#   LPS  : 1 L/s = 1e-3 m³/s                                  → 0.001
#   LPM  : 1 L/min                                            → 1.66667e-5
#   MLD  : 1e6 L/dag = 1000 m³/dag                            → 0.0115740741
#   CMH  : 1 m³/uur                                           → 2.77778e-4
#   CMD  : 1 m³/dag                                           → 1.15740741e-5
FLOW_TO_M3S: dict[str, float] = {
    'CFS':  0.0283168466,
    'GPM':  6.30901964e-5,
    'MGD':  0.0438126364,
    'IMGD': 0.0526167824,
    'AFD':  0.0142764102,
    'LPS':  0.001,
    'LPM':  1.0 / 60_000.0,
    'MLD':  1.0 / 86.4,
    'CMH':  1.0 / 3_600.0,
    'CMD':  1.0 / 86_400.0,
}

# US-eenheden: diameter in inch, snelheid in ft/s (i.p.v. mm / m/s bij SI).
US_UNITS: frozenset[str] = frozenset({'CFS', 'GPM', 'MGD', 'IMGD', 'AFD'})

INCH_TO_M: float = 0.0254
FT_S_TO_M_S: float = 0.3048


def flow_units_label(net) -> str:
    """
    Lees de flow-eenheid van een (kale EPYnetDTD-)`Network` op.

    Retourneert een label uit FLOW_CODE_TO_LABEL ('CMH', 'LPS', 'GPM', ...).
    Fallback: 'CMH' (meest voorkomend in Nederlandse drinkwaternetwerken),
    zowel bij een onbekende code als bij een falende toolkitaanroep.
    """
    try:
        code = int(net.EN_getflowunits())
    except Exception:
        return 'CMH'
    return FLOW_CODE_TO_LABEL.get(code, 'CMH')


def flow_to_m3s(flow, units: str):
    """Converteer flow (scalar of ndarray) in `units` naar m³/s."""
    return flow * FLOW_TO_M3S.get(units, 1.0 / 3_600.0)


def diameter_to_m(diameter, units: str):
    """
    Converteer diameter (scalar of ndarray) van EPANET-eenheden naar meter.

    EPANET-conventie (EN_getlinkvalue(..., EN_DIAMETER)):
        SI  (CMH, LPS, LPM, MLD, CMD) : diameter in mm
        US  (CFS, GPM, MGD, IMGD, AFD): diameter in inch
    """
    if units in US_UNITS:
        return diameter * INCH_TO_M
    return diameter / 1000.0


def velocity_to_ms(velocity, units: str):
    """
    Converteer snelheid (scalar of ndarray) van EPANET-eenheden naar m/s.

    EPANET-conventie (EN_getlinkvalue(..., EN_VELOCITY)):
        SI (CMH, LPS, ...) : m/s  (geen conversie nodig)
        US (CFS, GPM, ...) : ft/s → m/s via FT_S_TO_M_S
    """
    if units in US_UNITS:
        return velocity * FT_S_TO_M_S
    return velocity
