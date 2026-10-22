"""
Lees reservoirs, bronnen en patronen uit een EPANET .inp-bestand.

Geen epynet nodig: dit module werkt puur op de tekst van het bestand, zodat
de logica los getest kan worden.

Gebruikt door NzingaFlowSolver voor de automatische reservoirbronnen
(``auto_reservoir_sources=True``).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReservoirSource:
    """Een ingeschakeld reservoir (head > 0) met optioneel een bron."""
    name:     str
    head:     float
    kind:     str | None      # 'CONCEN', 'MASS', 'SETPOINT', 'FLOWPACED' of None (geen bron)
    strength: float | None    # bronsterkte [mg/L] voor CONCEN, anders None
    pattern:  str | None      # patroonnaam of None (constant)


def _sections(path: str) -> dict[str, list[list[str]]]:
    """Alle niet-lege, niet-commentaar regels per sectie (sectienamen in hoofdletters)."""
    out: dict[str, list[list[str]]] = {}
    current = None
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("["):
                current = line.strip("[]").strip().upper()
                out.setdefault(current, [])
                continue
            if current is not None:
                out[current].append(line.split())
    return out


def parse_time_s(text: str) -> int:
    """EPANET-tijd ('H:MM', 'H:MM:SS' of 'H', in uren) naar seconden."""
    parts = [int(p) for p in text.split(":")]
    if len(parts) == 1:
        return parts[0] * 3600
    h, m = parts[0], parts[1]
    s = parts[2] if len(parts) > 2 else 0
    return h * 3600 + m * 60 + s


def read_patterns(path: str) -> tuple[dict[str, list[float]], int, int]:
    """Geeft (patronen, patroonstap [s], patroonstart [s]) uit het bestand."""
    secs = _sections(path)
    pats: dict[str, list[float]] = {}
    for row in secs.get("PATTERNS", []):
        if len(row) < 2:
            continue
        pats.setdefault(row[0], []).extend(float(v) for v in row[1:])
    step, start = 3600, 0
    for row in secs.get("TIMES", []):
        key = " ".join(row[:2]).upper()
        if key.startswith("PATTERN TIMESTEP") and len(row) >= 3:
            step = parse_time_s(row[-1])
        elif key.startswith("PATTERN START") and len(row) >= 3:
            start = parse_time_s(row[-1])
    return pats, step, start


def pattern_multiplier(values: list[float], t: float, step: int, start: int) -> float:
    """Patroonfactor op tijd t [s], volgens EPANET: index = ((t + start) // stap) mod lengte.

    `start` (Pattern Start) schuift het patroon op: met start = 2 stappen begint de
    simulatie bij de derde patroonwaarde. Het patroon herhaalt zich.
    """
    if not values:
        return 1.0
    return values[int((t + start) // step) % len(values)]


def read_reservoir_sources(path: str) -> list[ReservoirSource]:
    """Alle reservoirs met head > 0, met hun eventuele bron.

    Reservoirs met head 0 (uitgeschakeld) worden niet opgenomen.
    """
    secs = _sections(path)
    sources: dict[str, tuple[str, float, str | None]] = {}
    for row in secs.get("SOURCES", []):
        if len(row) >= 3:
            pat = row[3] if len(row) >= 4 else None
            sources[row[0]] = (row[1].upper(), float(row[2]), pat)

    out: list[ReservoirSource] = []
    for row in secs.get("RESERVOIRS", []):
        if len(row) < 2:
            continue
        name, head = row[0], float(row[1])
        if head <= 0.0:          # uitgeschakeld reservoir: niet meenemen
            continue
        if name in sources:
            kind, strength, pat = sources[name]
            out.append(ReservoirSource(
                name=name, head=head, kind=kind,
                strength=strength if kind == "CONCEN" else None,
                pattern=pat))
        else:
            out.append(ReservoirSource(name=name, head=head, kind=None,
                                       strength=None, pattern=None))
    return out


def read_junction_sources(path: str) -> list[ReservoirSource]:
    """Bronnen op knopen die geen reservoir en geen tank zijn (meestal junctions).

    Deze worden alleen gebruikt op stappen waarin de knoop een negatieve vraag
    heeft (= externe instroom): de instroom krijgt dan de bronconcentratie.
    """
    secs = _sections(path)
    reservoirs = {row[0] for row in secs.get("RESERVOIRS", []) if row}
    tanks = {row[0] for row in secs.get("TANKS", []) if row}
    out: list[ReservoirSource] = []
    for row in secs.get("SOURCES", []):
        if len(row) < 3 or row[0] in reservoirs or row[0] in tanks:
            continue
        kind = row[1].upper()
        out.append(ReservoirSource(
            name=row[0], head=0.0, kind=kind,
            strength=float(row[2]) if kind == "CONCEN" else None,
            pattern=row[3] if len(row) >= 4 else None))
    return out
