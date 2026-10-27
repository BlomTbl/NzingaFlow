"""
Bronnen, reservoirs en patronen uit een geladen epynet-netwerk.

Vervangt het vroegere tekstparser-module ``inp_sources``: epynet leest het
.inp-bestand met de EPANET-kern zelf, zodat de bronnen dezelfde betekenis
hebben als in EPANET (inclusief patroonstart en -stap).

Gebruikt door NzingaFlowSolver voor de automatische reservoir- en junctionbronnen
(``auto_reservoir_sources=True``).
"""

from __future__ import annotations

from dataclasses import dataclass

from epynet.enum import EN_SourceType, EN_TimeParameter


@dataclass(frozen=True)
class ReservoirSource:
    """Een ingeschakeld reservoir (head > 0) met optioneel een bron."""

    name: str
    head: float
    kind: str | None  # 'CONCEN', 'MASS', 'SETPOINT', 'FLOWPACED' of None (geen bron)
    strength: float | None  # bronsterkte [mg/L] voor CONCEN, anders None
    pattern: str | None  # patroonnaam of None (constant)


def _source_of(node):
    """(kind, sterkte, patroonnaam) van een knoop, of (None, None, None) zonder bron."""
    try:
        kind = EN_SourceType(int(node.source_type)).name.replace("EN_", "")
    except Exception:  # EPANET fout 240: knoop heeft geen bron
        return None, None, None
    strength = float(node.source_quality)
    pat = node.source_pattern
    return kind, strength, (pat.uid if pat is not None else None)


def pattern_multiplier(values: list[float], t: float, step: int, start: int) -> float:
    """Patroonfactor op tijd t [s], volgens EPANET: index = ((t + start) // stap) mod lengte."""
    if not values:
        return 1.0
    return values[int((t + start) // step) % len(values)]


def pattern_timing(net) -> tuple[int, int]:
    """(patroonstap [s], patroonstart [s]) uit de tijdparameters van het netwerk."""
    step = net.EN_gettimeparam(EN_TimeParameter.EN_PATTERNSTEP)
    start = net.EN_gettimeparam(EN_TimeParameter.EN_PATTERNSTART)
    return int(step), int(start)


def reservoir_sources(net) -> list[ReservoirSource]:
    """Alle reservoirs met head > 0, met hun eventuele bron. Head 0 = uitgeschakeld."""
    out: list[ReservoirSource] = []
    for r in net.reservoirs:
        # In EPANET is de vaste opvoerhoogte van een reservoir opgeslagen als elevation
        # (r.head is de berekende waarde en is vóór een solve 0).
        head = float(r.elevation)
        if head <= 0.0:
            continue
        kind, strength, pat = _source_of(r)
        out.append(
            ReservoirSource(
                name=r.uid,
                head=head,
                kind=kind,
                strength=strength if kind == "CONCEN" else None,
                pattern=pat,
            )
        )
    return out


def junction_sources(net) -> list[ReservoirSource]:
    """Bronnen op junctions (geen reservoir, geen tank). Gelden op stappen met negatieve vraag."""
    out: list[ReservoirSource] = []
    for j in net.junctions:
        kind, strength, pat = _source_of(j)
        if kind is None:
            continue
        out.append(
            ReservoirSource(
                name=j.uid,
                head=0.0,
                kind=kind,
                strength=strength if kind == "CONCEN" else None,
                pattern=pat,
            )
        )
    return out


def pattern_values(net, name: str) -> list[float]:
    """Patroonwaarden van patroon ``name``; ValueError als het niet bestaat."""
    for p in net.patterns:
        if p.uid == name:
            return list(p.values)
    raise ValueError(f"patroon '{name}' niet gevonden in het netwerk.")
