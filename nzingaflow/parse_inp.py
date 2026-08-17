# nzingaflow/parse_inp.py
"""
Twee ingangen voor netwerktopologie:

  load_from_epynet(net, simtime=0)
      Leest topologie, hydraulica, coördinaten en patronen rechtstreeks
      uit een geladen epynet.Network object (kale EPYnetDTD-`Network`,
      geen `epynet.compat`-laag nodig). Geeft de meest nauwkeurige
      flow/velocity omdat EPANET het netwerk zelf oplost.

  parse_inp(path)
      Minimale .inp-parser zonder epynet. Gebruikt als fallback of voor
      eenvoudige tests. Hydraulica via estimate_hydraulics().

Beide functies geven dezelfde dict-structuur terug:

    nodes          : {naam: {'type': 'junction'|'reservoir'|'tank', ...}}
    node_names     : list[str]                    volgorde = index
    node_index     : {naam: int}
    node_count     : int
    pipe_ids       : list[str]
    pipe_start     : (n,) int32    0-based knoopindex
    pipe_end       : (n,) int32    0-based knoopindex
    pipe_length    : (n,) float64  [m]
    pipe_diam      : (n,) float64  [m]
    pipe_area      : (n,) float64  [m²]
    pipe_raw       : list[dict]    ruwe gegevens per leiding
    node_coords    : {naam: (x, y)}   leeg dict als niet beschikbaar
    flow           : (n,) float64  [m³/s]  None bij parse_inp (gebruik estimate_hydraulics)
    velocity       : (n,) float64  [m/s]   None bij parse_inp
    patterns       : {pat_id: list[float]}  multipliers per patroon-stap
    pat_step_s     : int    patroon-tijdstap [s]
    hyd_step_s     : int    hydraulische tijdstap [s]
    duration_s     : int    simulatieduur [s]
    source_patterns: {reservoir_uid: {'base_q': float, 'pattern': list[float]}}
"""

from __future__ import annotations
import re
import numpy as np
from pathlib import Path
from math import pi

from epynet.node import Reservoir, Tank
from epynet.link import Pipe, Pump

from . import units as u


# ═════════════════════════════════════════════════════════════════════════════
# Primaire ingang: epynet
# ═════════════════════════════════════════════════════════════════════════════

def load_from_epynet(net, simtime: int = 0) -> dict:
    """
    Laad netwerktopologie, hydraulica, coördinaten en patronen uit epynet.

    Parameters
    ----------
    net      : epynet.Network  (kale EPYnetDTD-Network, reeds geladen,
               hoeft niet opgelost te zijn)
    simtime  : simulatietijdstip in seconden waarop hydraulica wordt opgelost.
               Standaard 0 (steady-state beginconditie).

    Returns
    -------
    dict  — zelfde structuur als parse_inp() plus:
        flow            : (n,) float64  m³/s  (werkelijke EPANET-waarden)
        velocity        : (n,) float64  m/s
        patterns        : {pat_id: list[float]}
        pat_step_s      : int
        hyd_step_s      : int
        duration_s      : int
        source_patterns : {res_uid: {'base_q': float, 'pattern': list[float]}}

    Let op — eenmalig gebruik
    ──────────────────────────
    Deze functie opent en sluit haar eigen `_NoSaveHydraulicSolver`-sessie
    (via een with-blok). Voor herhaalde solves op hetzelfde netwerk (een
    EPS-loop) is `HydraulicModel` (hydraulics.py) efficiënter: die
    hergebruikt de EN_openH()-sessie over meerdere solve()-aanroepen heen
    i.p.v.'m hier bij elke load_from_epynet()-aanroep opnieuw te openen.
    """
    # Geen `net.ep`-shim meer — EPYnetDTD's ENxxx-toolkitfuncties heten
    # rechtstreeks EN_xxx op het Network-object zelf. `_NoSaveHydraulicSolver`
    # hergebruikt uit hydraulics.py i.p.v. hier een eigen kopie te onderhouden.
    from .hydraulics import _NoSaveHydraulicSolver

    # ── 1. Tijdparameters ─────────────────────────────────────────────────────
    hyd_step_s  = int(net.EN_gettimeparam(u.EN_TimeParameter.EN_HYDSTEP))
    pat_step_s  = int(net.EN_gettimeparam(u.EN_TimeParameter.EN_PATTERNSTEP))
    duration_s  = int(net.EN_gettimeparam(u.EN_TimeParameter.EN_DURATION))
    # Nul-waarden zijn geldig maar onbruikbaar — gebruik veilige defaults
    if hyd_step_s  <= 0: hyd_step_s  = 3600
    if pat_step_s  <= 0: pat_step_s  = 3600
    if duration_s  <= 0: duration_s  = 24 * 3600

    # ── 2. Flow-eenheden ─────────────────────────────────────────────────────
    flow_units = u.flow_units_label(net)

    # ── 3. Knopen ─────────────────────────────────────────────────────────────
    nodes      = {}
    node_names = []
    node_coords = {}

    for node in net.nodes:
        uid = node.uid
        # Type bepalen via isinstance — net.nodes is al getypeerd door
        # EPYnetDTD's NodeFactory (Junction/Reservoir/Tank), dus geen
        # membership-check tegen een losse uid-verzameling meer nodig.
        if isinstance(node, Reservoir):
            ntype = 'reservoir'
        elif isinstance(node, Tank):
            ntype = 'tank'
        else:
            ntype = 'junction'

        nodes[uid] = {'type': ntype}
        node_names.append(uid)

        # Coördinaten via EN_getcoord
        try:
            x, y = net.EN_getcoord(node.index)
            node_coords[uid] = (float(x), float(y))
        except Exception:
            pass   # knoop zonder coördinaten: overgeslagen

    node_index = {n: i for i, n in enumerate(node_names)}

    # ── 4. Hydraulica oplossen op gevraagd tijdstip ───────────────────────────
    try:
        solver = _NoSaveHydraulicSolver(net)
        solver.solve_step(pattern_start_time=simtime)
        solver.close()
    except Exception:
        pass   # als solve faalt: flow/velocity worden 0

    # ── 5. Leidingen + pompen + kleppen ──────────────────────────────────────
    pipe_ids    = []
    pipe_raw    = []
    pipe_starts = []
    pipe_ends   = []
    pipe_lengths = []
    pipe_diams  = []
    flow_list   = []
    vel_list    = []

    for link in net.links:
        uid = link.uid
        idx = link.index

        fn_uid = link.from_node.uid
        tn_uid = link.to_node.uid

        if fn_uid not in node_index or tn_uid not in node_index:
            continue

        # Geometrie — EN_DIAMETER/EN_LENGTH: mm/inch resp. altijd in de
        # invoereenheid van het .inp (zie units.diameter_to_m()).
        try:
            diam_raw = float(net.EN_getlinkvalue(idx, u.EN_LinkProperty.EN_DIAMETER))
            length_m = float(net.EN_getlinkvalue(idx, u.EN_LinkProperty.EN_LENGTH))
        except Exception:
            diam_raw = 100.0
            length_m = 1.0

        diam_m = u.diameter_to_m(diam_raw, flow_units)
        # Kleppen en pompen krijgen een symbolische minimale lengte
        if length_m <= 0:
            length_m = 0.1

        # Hydraulica — EN_VELOCITY: m/s (SI) of ft/s (US), zie
        # units.velocity_to_ms(). Vóór deze consolidatie ontbrak deze
        # conversie hier volledig (velocity bleef ft/s op een US-netwerk).
        try:
            raw_flow = float(net.EN_getlinkvalue(idx, u.EN_LinkProperty.EN_FLOW))
            raw_vel  = float(net.EN_getlinkvalue(idx, u.EN_LinkProperty.EN_VELOCITY))
        except Exception:
            raw_flow = 0.0
            raw_vel  = 0.0

        # Stromingsrichting: negatief = omgekeerd t.o.v. from→to
        # NzingaFlow werkt altijd met positieve flow; richting zit in pipe_start/end
        if raw_flow < 0:
            fn_uid, tn_uid = tn_uid, fn_uid   # draai richting om

        flow_m3s = abs(u.flow_to_m3s(raw_flow, flow_units))
        vel_ms   = abs(u.velocity_to_ms(raw_vel, flow_units))

        # Link-type via isinstance — net.links is al getypeerd door
        # EPYnetDTD's LinkFactory (Pipe/Pump/Valve).
        if isinstance(link, Pipe):
            ltype = 'pipe'
        elif isinstance(link, Pump):
            ltype = 'pump'
        else:
            ltype = 'valve'

        pipe_ids.append(uid)
        pipe_starts.append(node_index[fn_uid])
        pipe_ends.append(node_index[tn_uid])
        pipe_lengths.append(length_m)
        pipe_diams.append(diam_m)
        flow_list.append(flow_m3s)
        vel_list.append(vel_ms)
        pipe_raw.append({
            'id': uid, 'from': fn_uid, 'to': tn_uid,
            'length': length_m, 'diameter': diam_m,
            'type': ltype,
        })

    pipe_start  = np.array(pipe_starts,  dtype=np.int32)
    pipe_end    = np.array(pipe_ends,    dtype=np.int32)
    pipe_length = np.array(pipe_lengths, dtype=np.float64)
    pipe_diam   = np.array(pipe_diams,   dtype=np.float64)
    pipe_area   = pi * (pipe_diam / 2.0) ** 2
    flow        = np.array(flow_list,    dtype=np.float64)
    velocity    = np.array(vel_list,     dtype=np.float64)

    # Waar velocity == 0 maar diameter > 0: bereken uit flow
    zero_vel = (velocity == 0) & (pipe_area > 0)
    velocity[zero_vel] = flow[zero_vel] / pipe_area[zero_vel]

    # ── 6. Patronen ───────────────────────────────────────────────────────────
    patterns = {}
    n_pat = net.EN_getcount(u.EN_CountType.EN_PATCOUNT)
    for pi_idx in range(1, n_pat + 1):
        try:
            pat_uid  = net.EN_getpatternid(pi_idx)
            pat_len  = net.EN_getpatternlen(pi_idx)
            mults    = [float(net.EN_getpatternvalue(pi_idx, k + 1))
                        for k in range(pat_len)]
            patterns[pat_uid] = mults
        except Exception:
            pass

    # ── 7. Bronpatronen per reservoir ─────────────────────────────────────────
    source_patterns = {}
    for res in net.reservoirs:
        uid  = res.uid
        nidx = res.index
        try:
            base_q    = float(net.EN_getnodevalue(nidx, u.EN_NodeProperty.EN_SOURCEQUAL))
            pat_idx   = int(net.EN_getnodevalue(nidx, u.EN_NodeProperty.EN_SOURCEPAT))
        except Exception:
            base_q, pat_idx = 1.0, 0

        if pat_idx > 0:
            try:
                pat_uid = net.EN_getpatternid(pat_idx)
                mults   = patterns.get(pat_uid, [1.0])
            except Exception:
                mults = [1.0]
        else:
            mults = [1.0]

        source_patterns[uid] = {
            'base_q':  base_q,
            'pattern': [base_q * m for m in mults],
        }

    return {
        # ── topologie ──────────────────────────────────────────────────
        'nodes':           nodes,
        'node_names':      node_names,
        'node_index':      node_index,
        'node_count':      len(node_names),
        # ── geometrie ──────────────────────────────────────────────────
        'pipe_ids':        pipe_ids,
        'pipe_start':      pipe_start,
        'pipe_end':        pipe_end,
        'pipe_length':     pipe_length,
        'pipe_diam':       pipe_diam,
        'pipe_area':       pipe_area,
        'pipe_raw':        pipe_raw,
        # ── ruimtelijk ─────────────────────────────────────────────────
        'node_coords':     node_coords,
        # ── hydraulica ─────────────────────────────────────────────────
        'flow':            flow,
        'velocity':        velocity,
        # ── tijdparameters ─────────────────────────────────────────────
        'hyd_step_s':      hyd_step_s,
        'pat_step_s':      pat_step_s,
        'duration_s':      duration_s,
        # ── patronen & bronnen ─────────────────────────────────────────
        'patterns':        patterns,
        'source_patterns': source_patterns,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Fallback ingang: .inp parser (geen epynet vereist)
# ═════════════════════════════════════════════════════════════════════════════

def parse_inp(path) -> dict:
    """
    Lees een EPANET .inp bestand zonder epynet.

    Leest: [JUNCTIONS], [RESERVOIRS], [TANKS], [PIPES], [VALVES],
           [COORDINATES], [PATTERNS], [SOURCES], [TIMES].

    Returns dezelfde dict als load_from_epynet(), maar:
        flow     = None  (gebruik estimate_hydraulics() na aanroep)
        velocity = None
    """
    path = Path(path)
    text = path.read_text(encoding='utf-8', errors='replace')
    sections = _split_sections(text)

    # ── Knopen ────────────────────────────────────────────────────────────────
    nodes = {}
    for line in sections.get('JUNCTIONS', []):
        parts = line.split()
        if parts:
            nodes[parts[0]] = {'type': 'junction'}

    for line in sections.get('RESERVOIRS', []):
        parts = line.split()
        if parts:
            nodes[parts[0]] = {'type': 'reservoir'}

    for line in sections.get('TANKS', []):
        parts = line.split()
        if parts:
            nodes[parts[0]] = {'type': 'tank',
                               'diameter': float(parts[5]) if len(parts) > 5 else 10.0}

    node_names = list(nodes.keys())
    node_index = {n: i for i, n in enumerate(node_names)}

    # ── Leidingen ─────────────────────────────────────────────────────────────
    pipe_ids, pipe_raw = [], []
    pipe_starts, pipe_ends, pipe_lengths, pipe_diams = [], [], [], []

    for line in sections.get('PIPES', []):
        parts = line.split()
        if len(parts) < 5:
            continue
        pid, frm, to = parts[0], parts[1], parts[2]
        if frm not in node_index or to not in node_index:
            continue
        leng  = float(parts[3])
        diam_m = float(parts[4]) * 1e-3
        pipe_ids.append(pid);     pipe_starts.append(node_index[frm])
        pipe_ends.append(node_index[to]); pipe_lengths.append(leng)
        pipe_diams.append(diam_m)
        pipe_raw.append({'id': pid, 'from': frm, 'to': to,
                         'length': leng, 'diameter': diam_m, 'type': 'pipe'})

    # Kleppen als leidingen met symbolische lengte
    for line in sections.get('VALVES', []):
        parts = line.split()
        if len(parts) < 4:
            continue
        pid, frm, to = parts[0], parts[1], parts[2]
        if frm not in node_index or to not in node_index:
            continue
        diam_m = float(parts[3]) * 1e-3
        pipe_ids.append(pid);     pipe_starts.append(node_index[frm])
        pipe_ends.append(node_index[to]); pipe_lengths.append(0.1)
        pipe_diams.append(diam_m)
        pipe_raw.append({'id': pid, 'from': frm, 'to': to,
                         'length': 0.1, 'diameter': diam_m, 'type': 'valve'})

    pipe_start  = np.array(pipe_starts,  dtype=np.int32)
    pipe_end    = np.array(pipe_ends,    dtype=np.int32)
    pipe_length = np.array(pipe_lengths, dtype=np.float64)
    pipe_diam   = np.array(pipe_diams,   dtype=np.float64)
    pipe_area   = pi * (pipe_diam / 2.0) ** 2

    # ── Coördinaten ───────────────────────────────────────────────────────────
    node_coords = {}
    for line in sections.get('COORDINATES', []):
        parts = line.split()
        if len(parts) >= 3:
            try:
                node_coords[parts[0]] = (float(parts[1]), float(parts[2]))
            except ValueError:
                pass

    # ── Patronen ──────────────────────────────────────────────────────────────
    patterns: dict[str, list[float]] = {}
    for line in sections.get('PATTERNS', []):
        parts = line.split()
        if len(parts) >= 2:
            pid   = parts[0]
            mults = [float(v) for v in parts[1:]]
            patterns.setdefault(pid, []).extend(mults)

    # ── Tijdparameters uit [TIMES] ─────────────────────────────────────────────
    hyd_step_s = 3600; pat_step_s = 3600; duration_s = 24 * 3600
    for line in sections.get('TIMES', []):
        lo = line.lower()
        val = _parse_time_value(line)
        if val is None:
            continue
        if   'hydraulic timestep' in lo: hyd_step_s = val
        elif 'pattern timestep'   in lo: pat_step_s = val
        elif 'duration'           in lo: duration_s = val

    # ── Bronpatronen uit [SOURCES] ────────────────────────────────────────────
    source_patterns = {}
    for line in sections.get('SOURCES', []):
        parts = line.split()
        if len(parts) < 3:
            continue
        res_uid  = parts[0]
        # src_type = parts[1]  # CONCEN / MASS / FLOW / SETPOINT
        try:
            base_q = float(parts[2])
        except ValueError:
            continue
        pat_id   = parts[3] if len(parts) > 3 else None
        mults    = patterns.get(pat_id, [1.0]) if pat_id else [1.0]
        source_patterns[res_uid] = {
            'base_q':  base_q,
            'pattern': [base_q * m for m in mults],
        }

    return {
        'nodes':           nodes,
        'node_names':      node_names,
        'node_index':      node_index,
        'node_count':      len(node_names),
        'pipe_ids':        pipe_ids,
        'pipe_start':      pipe_start,
        'pipe_end':        pipe_end,
        'pipe_length':     pipe_length,
        'pipe_diam':       pipe_diam,
        'pipe_area':       pipe_area,
        'pipe_raw':        pipe_raw,
        'node_coords':     node_coords,
        'flow':            None,   # gebruik estimate_hydraulics()
        'velocity':        None,
        'hyd_step_s':      hyd_step_s,
        'pat_step_s':      pat_step_s,
        'duration_s':      duration_s,
        'patterns':        patterns,
        'source_patterns': source_patterns,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Hydraulica-schatter (fallback zonder epynet)
# ═════════════════════════════════════════════════════════════════════════════

def estimate_hydraulics(topo: dict, base_flow: float = 1e-3) -> tuple:
    """
    Schat flow [m³/s] en velocity [m/s] per leiding.

    Methode: flow ∝ doorsnede-oppervlak  (gelijke snelheid in alle leidingen).
    base_flow = referentiedebiet voor de leiding met de grootste doorsnede.

    Returns
    -------
    flow     : (n,) float64  m³/s
    velocity : (n,) float64  m/s
    """
    area = topo['pipe_area']
    if area.max() > 0:
        flow = base_flow * area / area.max()
    else:
        flow = np.full_like(area, base_flow)
    velocity = np.where(area > 0, flow / area, 0.0)
    return flow, velocity


# ═════════════════════════════════════════════════════════════════════════════
# Hulpfuncties
# ═════════════════════════════════════════════════════════════════════════════

def _split_sections(text: str) -> dict[str, list[str]]:
    """Splits .inp tekst op secties; geeft {SECTIENAAM: [regels]} terug."""
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        line = re.sub(r';.*', '', line).strip()
        if not line:
            continue
        if line.startswith('['):
            current = line.strip('[]').upper()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return sections


def _parse_time_value(line: str) -> int | None:
    """
    Leest tijdwaarde uit een [TIMES]-regel.
    Ondersteunt formaten: '24:00', '3600', '1.5' (uren als float).
    Geeft seconden terug als int, of None bij fout.
    """
    parts = line.split()
    # Tijdwaarde staat na de label (laatste token)
    val_str = parts[-1] if parts else ''
    try:
        if ':' in val_str:
            h, m = val_str.split(':', 1)
            return int(h) * 3600 + int(m) * 60
        else:
            return int(float(val_str) * 3600)
    except (ValueError, IndexError):
        return None
