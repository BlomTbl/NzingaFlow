# nzingaflow/msxlibrary.py
"""
MsxLibraryBridge — directe koppeling met de EPANET-MSX gedeelde bibliotheek.

Gebruik de C API (epanetmsx.dll / libepanetmsx.so) om waternetwerken met
meerdere chemische stoffen te simuleren. De brug biedt drie lagen:

  1. ``MsxNativeLib``     — dunne ctypes-wrapper rond alle MSX_* C-functies.
  2. ``MsxNetworkState``  — leesobject: concentraties, bronnen, parameters.
  3. ``MsxSimulation``    — hoog-niveau orchestrator: laden, uitvoeren, uitlezen.

Compatibiliteit
---------------
EPANET-MSX 2.0 (epanetmsx.h uit github.com/USEPA/EPANETMSX).
Werkt met Windows (DLL) en Linux (shared object); macOS (.dylib) nog niet
meegebouwd — zie nzingaflow/lib/README voor bouwinstructies.

Let op: de MSX_* C-API is tussen 1.1 en 2.0 op punten gewijzigd (o.a. is
MSXstep's signatuur van long* naar double* gegaan, en zijn node/link-
tellingen/-namen verplaatst naar de EPANET-laag i.p.v. MSXgetcount/
MSXgetID). Deze wrapper is tegen 2.0 geverifieerd; gebruik hem niet
tegen een MSX 1.1-bibliotheek zonder de signaturen opnieuw te controleren.

Voorbeeld
---------
>>> bridge = MsxSimulation("net2-cl2.inp", "net2-cl2.msx")
>>> result = bridge.run()
>>> print(result.node_concentrations("CL2"))
"""

from __future__ import annotations

import ctypes
import platform
from ctypes import c_char_p, c_int, c_double, POINTER, byref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  Constanten (identiek aan epanetmsx.h, maar benoemd naar NzingaFlow-stijl)
# ══════════════════════════════════════════════════════════════════════════════

class _ObjectType:
    NODE      = 0
    LINK      = 1
    TANK      = 2
    SPECIES   = 3
    TERM      = 4
    PARAMETER = 5
    CONSTANT  = 6
    PATTERN   = 7

class _SpeciesKind:
    BULK = 0
    WALL = 1

class _SourceKind:
    NONE       = -1
    FIXED_CONC = 0   # CONCEN in MSX-dialect
    MASS_FLUX  = 1   # MASS
    SETPOINT   = 2
    FLOW_PACED = 3   # FLOWPACED

class _EnCountType:
    """EN_* objecttypes voor ENgetcount() — komen uit de EPANET (niet-MSX)
    toolkit. MSXgetcount()/MSXgetID() ondersteunen alléén SPECIES/CONSTANT/
    PARAMETER/PATTERN; node-, link- en tank-aantallen en -namen moeten via
    de onderliggende epanet2-bibliotheek worden opgevraagd."""
    NODECOUNT = 0
    TANKCOUNT = 1
    LINKCOUNT = 2

EN_MAXID = 31  # max. aantal tekens in een EPANET ID-naam (epanet2_enums.h)


# ══════════════════════════════════════════════════════════════════════════════
#  Hulpfuncties
# ══════════════════════════════════════════════════════════════════════════════

def _encode(s: str) -> bytes:
    return s.encode("utf-8")


def _default_lib_path() -> str:
    """Zoek de MSX-bibliotheek in de gebruikelijke installatiemappen."""
    system = platform.system()
    candidates: List[str] = []

    if system == "Windows":
        candidates = [
            "epanetmsx.dll",
            r"C:\Program Files\EPANET-MSX\epanetmsx.dll",
        ]
    elif system == "Darwin":
        candidates = [
            "libepanetmsx.dylib",
            "/usr/local/lib/libepanetmsx.dylib",
        ]
    else:  # Linux
        candidates = [
            "libepanetmsx.so",
            "/usr/local/lib/libepanetmsx.so",
            "/usr/lib/libepanetmsx.so",
        ]

    if system == "Windows":
        names = ["epanetmsx.dll"]
    elif system == "Darwin":
        names = ["libepanetmsx.dylib"]
    else:  # Linux
        names = ["libepanetmsx.so"]

    # Zoek in de map van dit bestand, en in een eventuele lib/-submap
    # daarvan (bundeling van meegeleverde binaries, analoog aan epynet).
    # Platform-specifiek, want sinds we Windows- én Linux-binaries in
    # dezelfde lib/-map bundelen staan ze naast elkaar.
    here = Path(__file__).parent
    for base in (here, here / "lib"):
        for name in names:
            p = base / name
            if p.exists():
                return str(p)

    for candidate in candidates:
        if Path(candidate).exists():
            return candidate

    raise FileNotFoundError(
        "EPANET-MSX bibliotheek niet gevonden. Geef het pad expliciet mee via "
        "MsxNativeLib(lib_path='...')."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Laag 1: MsxNativeLib — ctypes-wrapper
# ══════════════════════════════════════════════════════════════════════════════

class MsxNativeLib:
    """
    Dunne wrapper rond de EPANET-MSX gedeelde bibliotheek.

    Alle MSX_* functies zijn direct beschikbaar als Python-methoden.
    Foutcodes worden automatisch vertaald naar ``MsxError`` uitzonderingen
    wanneer ``strict=True`` (standaard).

    Parameters
    ----------
    lib_path : str, optional
        Expliciet pad naar de gedeelde bibliotheek. Wordt automatisch
        gezocht als weggelaten.
    strict : bool
        Als ``True``, gooit niet-nul retourwaarden een ``MsxError``.
    """

    def __init__(self, lib_path: Optional[str] = None, strict: bool = True):
        path = lib_path or _default_lib_path()
        self._lib = ctypes.cdll.LoadLibrary(path)
        self._strict = strict
        self._bind_signatures()

        # libepanetmsx is dynamisch gelinkt tegen een epanet2-bibliotheek
        # (zie NEEDED-entry). MSXopen() leest netwerkgegevens via diezelfde
        # gedeelde globale EPANET-state, dus die moet via ENopen() gevuld
        # zijn vóórdat MSXopen() wordt aangeroepen — zie msxmain.c (de
        # officiële CLI-referentie), die exact deze volgorde aanhoudt:
        # ENopen(inp, rpt, out) → MSXopen(msx) → MSXsolveH() → MSXinit().
        #
        # We laden hier bewust de epanet2-bibliotheek die *naast*
        # libepanetmsx.so/dll staat (dezelfde die libepanetmsx zelf als
        # afhankelijkheid gebruikt), niet de losstaande epynet-kopie —
        # dat zijn twee onafhankelijke geheugeninstanties met elk hun eigen
        # globale state, en alleen de eerste is zichtbaar voor MSXopen().
        self._en = self._load_en_lib(path)
        self._bind_en_signatures()

    @staticmethod
    def _load_en_lib(msx_lib_path: str):
        """Laad de epanet2-bibliotheek die naast de MSX-bibliotheek staat.

        Leidt de verwachte bestandsnaam af van de MSX-bibliotheek die we
        al gekozen hebben (i.p.v. een platformblinde zoeklijst) — zo blijft
        dit correct zelfs als lib_path handmatig is meegegeven, en ongeacht
        of er binaries van meerdere platforms naast elkaar in dezelfde map
        staan (zoals in nzingaflow/lib/, dat Linux+Windows bundelt).
        """
        msx_path = Path(msx_lib_path)
        folder = msx_path.parent
        suffix = msx_path.suffix.lower()
        if suffix == ".dll":
            # Let op: bewust *niet* "epanet2.dll" — Windows' loader
            # resolveert impliciete DLL-afhankelijkheden op kale
            # modulenaam, ongeacht map, zodra een gelijknamige DLL al
            # ergens in het proces geladen is (zie Microsoft's "Dynamic-
            # link library search order"-documentatie). Zou epynet's
            # eigen epanet2.dll al geladen zijn (andere map, zelfde naam),
            # dan zou epanetmsx.dll's impliciete import stilzwijgend naar
            # dát exemplaar resolven — dezelfde klasse bug als de
            # SONAME-botsing op Linux. Vandaar de unieke naam.
            name = "epanet2_msx.dll"
        elif suffix == ".dylib":
            name = "libepanet2_msx.dylib"
        else:
            # Let op: bewust *niet* "libepanet2.so" — die naam botst met de
            # SONAME die epynet's eigen bundled epanet2-bibliotheek gebruikt
            # (epynet/lib/libepanet.so heeft intern SONAME "libepanet2.so").
            # Draaien beide in hetzelfde proces (zoals in NzingaFlow, dat
            # epynet gebruikt), dan retourneert de dynamic linker bij de
            # tweede dlopen() stilzwijgend de AL GELADEN epynet-copy i.p.v.
            # onze eigen bibliotheek — met verwarrende foutcodes als gevolg
            # zodra beide netwerken tegelijk open staan. Vandaar de unieke
            # bestands- én SONAME "libepanet2_msx.so".
            name = "libepanet2_msx.so"
        candidate = folder / name
        if candidate.exists():
            return ctypes.cdll.LoadLibrary(str(candidate))
        raise FileNotFoundError(
            f"Kon '{name}' niet vinden naast '{msx_lib_path}'. "
            "libepanetmsx heeft een epanet2.dll/libepanet2.so/.dylib nodig "
            "in dezelfde map (dit moet de versie zijn waartegen "
            "libepanetmsx zelf is gelinkt, niet een losse epynet-kopie)."
        )

    def _bind_en_signatures(self):
        EN = self._en
        EN.ENopen.restype   = c_int
        EN.ENopen.argtypes  = [c_char_p, c_char_p, c_char_p]
        EN.ENclose.restype  = c_int
        EN.ENclose.argtypes = []
        EN.ENgetcount.restype   = c_int
        EN.ENgetcount.argtypes  = [c_int, POINTER(c_int)]
        EN.ENgetnodeid.restype  = c_int
        EN.ENgetnodeid.argtypes = [c_int, c_char_p]
        EN.ENgetlinkid.restype  = c_int
        EN.ENgetlinkid.argtypes = [c_int, c_char_p]
        EN.ENgeterror.restype   = c_int
        EN.ENgeterror.argtypes  = [c_int, c_char_p, c_int]
        EN.ENgetnodeindex.restype  = c_int
        EN.ENgetnodeindex.argtypes = [c_char_p, POINTER(c_int)]
        EN.ENgetlinkindex.restype  = c_int
        EN.ENgetlinkindex.argtypes = [c_char_p, POINTER(c_int)]

    def _check_en(self, code: int, context: str = "") -> int:
        """Als _check(), maar vertaalt EN-foutcodes (aparte nummering t.o.v.
        MSX-foutcodes) via ENgeterror in plaats van MSXgeterror."""
        if self._strict and code != 0:
            buf = ctypes.create_string_buffer(256)
            try:
                self._en.ENgeterror(code, buf, 256)
                msg = buf.value.decode(errors="replace")
            except Exception:
                msg = f"EPANET fout {code}"
            raise MsxError(code, msg, context)
        return code

    def en_open(self, inp_path: str, rpt_path: str = "", out_path: str = "") -> None:
        """Open het onderliggende EPANET-netwerk (vereist vóór ``open()``)."""
        self._check_en(
            self._en.ENopen(_encode(inp_path), _encode(rpt_path), _encode(out_path)),
            "en_open",
        )

    def en_close(self) -> None:
        """Sluit het onderliggende EPANET-netwerk."""
        try:
            self._en.ENclose()
        except Exception:
            pass

    def en_node_count(self) -> int:
        n = c_int(0)
        self._check_en(self._en.ENgetcount(_EnCountType.NODECOUNT, byref(n)), "en_node_count")
        return n.value

    def en_link_count(self) -> int:
        n = c_int(0)
        self._check_en(self._en.ENgetcount(_EnCountType.LINKCOUNT, byref(n)), "en_link_count")
        return n.value

    def en_node_id(self, index: int) -> str:
        buf = ctypes.create_string_buffer(EN_MAXID + 1)
        self._check_en(self._en.ENgetnodeid(index, buf), "en_node_id")
        return buf.value.decode()

    def en_link_id(self, index: int) -> str:
        buf = ctypes.create_string_buffer(EN_MAXID + 1)
        self._check_en(self._en.ENgetlinkid(index, buf), "en_link_id")
        return buf.value.decode()

    def en_node_index(self, name: str) -> int:
        idx = c_int(0)
        self._check_en(self._en.ENgetnodeindex(_encode(name), byref(idx)), "en_node_index")
        return idx.value

    def en_link_index(self, name: str) -> int:
        idx = c_int(0)
        self._check_en(self._en.ENgetlinkindex(_encode(name), byref(idx)), "en_link_index")
        return idx.value

    # ── Signaturen ────────────────────────────────────────────────────────────

    def _bind_signatures(self):
        L = self._lib

        def _fn(name, restype, *argtypes):
            fn = getattr(L, name)
            fn.restype  = restype
            fn.argtypes = list(argtypes)
            return fn

        self._open        = _fn("MSXopen",        c_int, c_char_p)
        self._solve_hyd   = _fn("MSXsolveH",       c_int)
        self._use_hyd     = _fn("MSXusehydfile",   c_int, c_char_p)
        self._solve_qual  = _fn("MSXsolveQ",        c_int)
        self._init_run    = _fn("MSXinit",          c_int, c_int)
        # MSXstep gebruikt double* voor t/tleft sinds EPANET-MSX 2.0
        # (was long* in de 1.1-header waar deze wrapper oorspronkelijk op
        # was gebaseerd — c_long op de verkeerde grootte zou op Windows
        # geheugencorruptie geven en op Linux/macOS stille foutieve
        # tijdwaarden opleveren).
        self._step        = _fn("MSXstep",          c_int, POINTER(c_double), POINTER(c_double))
        self._save_out    = _fn("MSXsaveoutfile",   c_int, c_char_p)
        self._save_msx    = _fn("MSXsavemsxfile",   c_int, c_char_p)
        self._report      = _fn("MSXreport",        c_int)
        self._close       = _fn("MSXclose",         c_int)

        self._get_index   = _fn("MSXgetindex",      c_int, c_int, c_char_p, POINTER(c_int))
        self._get_idlen   = _fn("MSXgetIDlen",      c_int, c_int, c_int, POINTER(c_int))
        self._get_id      = _fn("MSXgetID",         c_int, c_int, c_int, c_char_p, c_int)
        self._get_count   = _fn("MSXgetcount",      c_int, c_int, POINTER(c_int))

        self._get_species = _fn("MSXgetspecies",    c_int, c_int,
                                POINTER(c_int), c_char_p, POINTER(c_double), POINTER(c_double))
        self._get_const   = _fn("MSXgetconstant",   c_int, c_int, POINTER(c_double))
        self._get_param   = _fn("MSXgetparameter",  c_int, c_int, c_int, c_int, POINTER(c_double))
        self._get_source  = _fn("MSXgetsource",     c_int, c_int, c_int,
                                POINTER(c_int), POINTER(c_double), POINTER(c_int))
        self._get_patlen  = _fn("MSXgetpatternlen", c_int, c_int, POINTER(c_int))
        self._get_patval  = _fn("MSXgetpatternvalue", c_int, c_int, c_int, POINTER(c_double))
        self._get_initq   = _fn("MSXgetinitqual",   c_int, c_int, c_int, c_int, POINTER(c_double))
        self._get_qual    = _fn("MSXgetqual",       c_int, c_int, c_int, c_int, POINTER(c_double))
        self._get_error   = _fn("MSXgeterror",      c_int, c_int, c_char_p, c_int)

        self._set_const   = _fn("MSXsetconstant",   c_int, c_int, c_double)
        self._set_param   = _fn("MSXsetparameter",  c_int, c_int, c_int, c_int, c_double)
        self._set_initq   = _fn("MSXsetinitqual",   c_int, c_int, c_int, c_int, c_double)
        self._set_source  = _fn("MSXsetsource",     c_int, c_int, c_int, c_int, c_double, c_int)
        self._set_patval  = _fn("MSXsetpatternvalue", c_int, c_int, c_int, c_double)
        self._set_pat     = _fn("MSXsetpattern",    c_int, c_int, POINTER(c_double), c_int)
        self._add_pat     = _fn("MSXaddpattern",    c_int, c_char_p)

    # ── Controle-hulper ───────────────────────────────────────────────────────

    def _check(self, code: int, context: str = "") -> int:
        if self._strict and code != 0:
            msg = self.error_message(code)
            raise MsxError(code, msg, context)
        return code

    # ── Publieke methoden — simulatie ─────────────────────────────────────────

    def open(self, msx_path: str) -> None:
        """Laad MSX-invoerbestand en initialiseer het systeem."""
        self._check(self._open(_encode(msx_path)), "open")

    def solve_hydraulics(self) -> None:
        """Voer een volledige hydraulische simulatie uit."""
        self._check(self._solve_hyd(), "solve_hydraulics")

    def use_hydraulic_file(self, hyd_path: str) -> None:
        """Gebruik een eerder opgeslagen hydraulicabestand."""
        self._check(self._use_hyd(_encode(hyd_path)), "use_hydraulic_file")

    def solve_quality(self) -> None:
        """Voer een volledige kwaliteitssimulatie uit."""
        self._check(self._solve_qual(), "solve_quality")

    def initialize_run(self, save_to_file: bool = False) -> None:
        """Bereid stapsgewijze kwaliteitsberekening voor."""
        self._check(self._init_run(int(save_to_file)), "initialize_run")

    def advance(self) -> Tuple[int, int]:
        """
        Zet de kwaliteitsoplossing één tijdstap vooruit.

        Geeft terug
        -----------
        t      : verstreken simulatietijd (s)
        tleft  : resterende tijd (s); 0 betekent klaar
        """
        t     = c_double(0)
        tleft = c_double(0)
        self._check(self._step(byref(t), byref(tleft)), "advance")
        return int(t.value), int(tleft.value)

    def save_results(self, out_path: str) -> None:
        self._check(self._save_out(_encode(out_path)), "save_results")

    def save_msx_file(self, msx_path: str) -> None:
        self._check(self._save_msx(_encode(msx_path)), "save_msx_file")

    def write_report(self) -> None:
        self._check(self._report(), "write_report")

    def close(self) -> None:
        self._check(self._close(), "close")

    # ── Publieke methoden — opvragen ──────────────────────────────────────────

    def object_index(self, obj_type: int, name: str) -> int:
        """Geef de 1-gebaseerde index van een benoemd object."""
        idx = c_int(0)
        self._check(self._get_index(obj_type, _encode(name), byref(idx)), "object_index")
        return idx.value

    def object_count(self, obj_type: int) -> int:
        n = c_int(0)
        self._check(self._get_count(obj_type, byref(n)), "object_count")
        return n.value

    def object_id(self, obj_type: int, index: int) -> str:
        """Geef de naam van een object op basis van zijn index."""
        buf_len = c_int(0)
        self._get_idlen(obj_type, index, byref(buf_len))
        buf = ctypes.create_string_buffer(buf_len.value + 1)
        self._check(self._get_id(obj_type, index, buf, buf_len.value), "object_id")
        return buf.value.decode("utf-8")

    def species_info(self, index: int) -> dict:
        """
        Geef type, eenheden en toleranties voor stof op index.

        Geeft terug: {'kind': int, 'units': str, 'atol': float, 'rtol': float}
        """
        kind  = c_int(0)
        units = ctypes.create_string_buffer(16)
        atol  = c_double(0)
        rtol  = c_double(0)
        self._check(self._get_species(index, byref(kind), units, byref(atol), byref(rtol)), "species_info")
        return {
            "kind":  kind.value,
            "units": units.value.decode("utf-8").strip(),
            "atol":  atol.value,
            "rtol":  rtol.value,
        }

    def constant_value(self, index: int) -> float:
        v = c_double(0)
        self._check(self._get_const(index, byref(v)), "constant_value")
        return v.value

    def parameter_value(self, obj_type: int, obj_index: int, param: int) -> float:
        v = c_double(0)
        self._check(self._get_param(obj_type, obj_index, param, byref(v)), "parameter_value")
        return v.value

    def source_info(self, node: int, species: int) -> dict:
        """
        Geeft bron-type, niveau en patroon-index voor (knoop, stof)-paar.
        """
        src_type = c_int(0)
        level    = c_double(0)
        pat      = c_int(0)
        self._check(self._get_source(node, species, byref(src_type), byref(level), byref(pat)), "source_info")
        return {"type": src_type.value, "level": level.value, "pattern": pat.value}

    def pattern_length(self, pat: int) -> int:
        n = c_int(0)
        self._check(self._get_patlen(pat, byref(n)), "pattern_length")
        return n.value

    def pattern_value(self, pat: int, period: int) -> float:
        v = c_double(0)
        self._check(self._get_patval(pat, period, byref(v)), "pattern_value")
        return v.value

    def initial_quality(self, obj_type: int, index: int, species: int) -> float:
        v = c_double(0)
        self._check(self._get_initq(obj_type, index, species, byref(v)), "initial_quality")
        return v.value

    def concentration(self, obj_type: int, index: int, species: int) -> float:
        """Huidige concentratie op knoop (0) of leiding (1)."""
        v = c_double(0)
        self._check(self._get_qual(obj_type, index, species, byref(v)), "concentration")
        return v.value

    def error_message(self, code: int) -> str:
        buf = ctypes.create_string_buffer(256)
        self._get_error(code, buf, 256)
        return buf.value.decode("utf-8").strip()

    # ── Publieke methoden — instellen ─────────────────────────────────────────

    def set_constant(self, index: int, value: float) -> None:
        self._check(self._set_const(index, c_double(value)), "set_constant")

    def set_parameter(self, obj_type: int, obj_index: int, param: int, value: float) -> None:
        self._check(self._set_param(obj_type, obj_index, param, c_double(value)), "set_parameter")

    def set_initial_quality(self, obj_type: int, index: int, species: int, value: float) -> None:
        self._check(self._set_initq(obj_type, index, species, c_double(value)), "set_initial_quality")

    def set_source(self, node: int, species: int, src_type: int, level: float, pat: int) -> None:
        self._check(self._set_source(node, species, src_type, c_double(level), pat), "set_source")

    def set_pattern_value(self, pat: int, period: int, value: float) -> None:
        self._check(self._set_patval(pat, period, c_double(value)), "set_pattern_value")

    def set_pattern(self, pat: int, multipliers: List[float]) -> None:
        arr = (c_double * len(multipliers))(*multipliers)
        self._check(self._set_pat(pat, arr, len(multipliers)), "set_pattern")

    def add_pattern(self, name: str) -> None:
        self._check(self._add_pat(_encode(name)), "add_pattern")


# ══════════════════════════════════════════════════════════════════════════════
#  Foutklasse
# ══════════════════════════════════════════════════════════════════════════════

class MsxError(RuntimeError):
    """Fout afkomstig van de EPANET-MSX C-bibliotheek."""

    def __init__(self, code: int, message: str, context: str = ""):
        self.code    = code
        self.context = context
        super().__init__(f"MSX fout {code} ({context}): {message}")


# ══════════════════════════════════════════════════════════════════════════════
#  Laag 2: MsxNetworkState — leesobject na simulatie
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MsxSpecies:
    """Metadata van één chemische stof."""
    index : int
    name  : str
    kind  : int    # 0 = bulk, 1 = wand
    units : str
    atol  : float
    rtol  : float

    @property
    def is_bulk(self) -> bool:
        return self.kind == _SpeciesKind.BULK

    @property
    def is_wall(self) -> bool:
        return self.kind == _SpeciesKind.WALL


@dataclass
class MsxSourceRecord:
    """Bron-definitie op één knoop voor één stof."""
    node_index    : int
    node_name     : str
    species_index : int
    species_name  : str
    kind          : int    # _SourceKind.*
    level         : float
    pattern_index : int


@dataclass
class MsxNetworkState:
    """
    Snapshot van alle MSX-netwerkinformatie na het laden van het bestand.

    Bevat stoffen, constanten, bronnen, initiële kwaliteiten en patronen.
    Wordt aangemaakt door ``MsxSimulation.load()``.
    """
    species      : List[MsxSpecies]         = field(default_factory=list)
    constants    : Dict[str, float]         = field(default_factory=dict)
    sources      : List[MsxSourceRecord]    = field(default_factory=list)
    node_initq   : np.ndarray              = field(default_factory=lambda: np.empty(0))
    link_initq   : np.ndarray              = field(default_factory=lambda: np.empty(0))
    equations    : Dict[str, Dict]          = field(default_factory=dict)
    patterns     : Dict[int, List[float]]   = field(default_factory=dict)

    # Naam → index opzoektabellen
    _species_by_name : Dict[str, MsxSpecies] = field(default_factory=dict, repr=False)

    def species_by_name(self, name: str) -> MsxSpecies:
        if name not in self._species_by_name:
            raise KeyError(f"Stof '{name}' niet gevonden. Beschikbaar: {list(self._species_by_name)}")
        return self._species_by_name[name]

    def bulk_species(self) -> List[MsxSpecies]:
        return [s for s in self.species if s.is_bulk]

    def wall_species(self) -> List[MsxSpecies]:
        return [s for s in self.species if s.is_wall]


# ══════════════════════════════════════════════════════════════════════════════
#  Laag 2b: simulatieresultaat
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class MsxSimulationResult:
    """
    Volledige tijdreeksen van een MSX-simulatie.

    Attributen
    ----------
    time_s          : 1-D array met tijdstempels in seconden
    node_quality    : array met vorm (tijdstappen, knopen, stoffen)
    link_quality    : array met vorm (tijdstappen, leidingen, stoffen)
    species         : lijst van ``MsxSpecies`` in de juiste volgorde
    node_names      : lijst van knoopnamen
    link_names      : lijst van leidingnamen
    """
    time_s       : np.ndarray
    node_quality : np.ndarray
    link_quality : np.ndarray
    species      : List[MsxSpecies]
    node_names   : List[str]
    link_names   : List[str]

    def time_hours(self) -> np.ndarray:
        return self.time_s / 3600.0

    def node_concentrations(self, species_name: str) -> np.ndarray:
        """
        Geef alle knoopconcentraties voor één stof als (tijd × knopen) array.
        """
        idx = next((i for i, s in enumerate(self.species) if s.name == species_name), None)
        if idx is None:
            raise KeyError(f"Stof '{species_name}' niet gevonden in resultaten.")
        return self.node_quality[:, :, idx]

    def link_concentrations(self, species_name: str) -> np.ndarray:
        idx = next((i for i, s in enumerate(self.species) if s.name == species_name), None)
        if idx is None:
            raise KeyError(f"Stof '{species_name}' niet gevonden in resultaten.")
        return self.link_quality[:, :, idx]

    def to_dataframe(self, species_name: str, element: str = "node"):
        """
        Zet tijdreeks om naar een pandas DataFrame (rijen = tijd, kolommen = namen).

        Parameters
        ----------
        species_name : naam van de stof
        element      : ``'node'`` of ``'link'``
        """
        import pandas as pd
        if element == "node":
            data  = self.node_concentrations(species_name)
            names = self.node_names
        else:
            data  = self.link_concentrations(species_name)
            names = self.link_names
        df = pd.DataFrame(data, index=self.time_hours(), columns=names)
        df.index.name = "tijd_h"
        return df


# ══════════════════════════════════════════════════════════════════════════════
#  Laag 3: MsxSimulation — hoog-niveau orchestrator
# ══════════════════════════════════════════════════════════════════════════════

class MsxSimulation:
    """
    Hoog-niveau interface voor een EPANET-MSX simulatie.

    Verzorgt het laden van bestanden, het ophalen van netwerkinformatie,
    het uitvoeren van de simulatie en het teruggeven van gestructureerde
    resultaten — zonder dat de gebruiker ctypes of lage-niveau API-aanroepen
    hoeft te kennen.

    Parameters
    ----------
    inp_path    : pad naar het EPANET .inp netwerk
    msx_path    : pad naar het EPANET-MSX .msx reactiebestand
    lib_path    : optioneel expliciet pad naar de gedeelde bibliotheek
    strict      : gooi een uitzondering bij niet-nul foutcodes (standaard True)

    Voorbeeld
    ---------
    >>> sim = MsxSimulation("net2-cl2.inp", "net2-cl2.msx")
    >>> state = sim.load()
    >>> print([s.name for s in state.bulk_species()])
    >>> result = sim.run()
    >>> df = result.to_dataframe("CL2", element="node")
    """

    _SOURCE_KIND_MAP = {
        "CONCEN":    _SourceKind.FIXED_CONC,
        "CONC":      _SourceKind.FIXED_CONC,
        "MASS":      _SourceKind.MASS_FLUX,
        "SETPOINT":  _SourceKind.SETPOINT,
        "FLOWPACED": _SourceKind.FLOW_PACED,
        "FLOW":      _SourceKind.FLOW_PACED,
        "NOSOURCE":  _SourceKind.NONE,
    }

    def __init__(
        self,
        inp_path : str,
        msx_path : str,
        lib_path : Optional[str] = None,
        strict   : bool = True,
    ):
        self.inp_path = str(inp_path)
        self.msx_path = str(msx_path)
        self._lib     = MsxNativeLib(lib_path, strict=strict)
        self._loaded  = False
        self._en_opened = False
        self._state   : Optional[MsxNetworkState] = None

        # Laad EPANET netwerk (hydraulica); MSX vereist dat dit eerst gebeurt
        self._epanet_open()

    # ── Initialisatie ─────────────────────────────────────────────────────────

    def _epanet_open(self):
        """Open het EPANET-netwerk via de standaard epanet2 API.

        MSXopen() verwacht dat het EPANET-netwerk al via ENopen() is
        geladen (zie msxmain.c, de officiële CLI-referentie: ENopen()
        wordt altijd vóór MSXopen() aangeroepen). Zonder deze stap blijft
        de netwerk-state binnen de gedeelde epanet2-bibliotheek leeg en
        faalt (of negeert) MSXopen() feitelijk stil.
        """
        base = self.inp_path[:-4] if self.inp_path.lower().endswith(".inp") else self.inp_path
        rpt_path = base + ".rpt"
        out_path = base + ".bin"
        self._lib.en_open(self.inp_path, rpt_path, out_path)
        self._en_opened = True

    def load(self) -> MsxNetworkState:
        """
        Open het MSX-bestand en bouw een ``MsxNetworkState`` op.

        Geeft terug
        -----------
        MsxNetworkState
            Bevat alle stoffen, constanten, bronnen en initiële kwaliteiten.
        """
        self._lib.open(self.msx_path)
        self._loaded = True
        self._state  = self._build_state()
        return self._state

    def _build_state(self) -> MsxNetworkState:
        lib   = self._lib
        state = MsxNetworkState()

        # ── Stoffen ───────────────────────────────────────────────────────────
        n_species = lib.object_count(_ObjectType.SPECIES)
        for i in range(1, n_species + 1):
            name = lib.object_id(_ObjectType.SPECIES, i)
            info = lib.species_info(i)
            sp   = MsxSpecies(
                index = i,
                name  = name,
                kind  = info["kind"],
                units = info["units"],
                atol  = info["atol"],
                rtol  = info["rtol"],
            )
            state.species.append(sp)
            state._species_by_name[name] = sp

        # ── Constanten ────────────────────────────────────────────────────────
        n_const = lib.object_count(_ObjectType.CONSTANT)
        for i in range(1, n_const + 1):
            name  = lib.object_id(_ObjectType.CONSTANT, i)
            value = lib.constant_value(i)
            state.constants[name] = value

        # ── Initiële kwaliteit: knopen ────────────────────────────────────────
        n_nodes = lib.en_node_count()
        n_sp    = len(state.species)
        state.node_initq = np.zeros((n_nodes, n_sp))
        for ni in range(1, n_nodes + 1):
            for si, sp in enumerate(state.species, start=1):
                state.node_initq[ni - 1, si - 1] = lib.initial_quality(
                    _ObjectType.NODE, ni, si
                )

        # ── Initiële kwaliteit: leidingen ─────────────────────────────────────
        n_links = lib.en_link_count()
        state.link_initq = np.zeros((n_links, n_sp))
        for li in range(1, n_links + 1):
            for si, sp in enumerate(state.species, start=1):
                state.link_initq[li - 1, si - 1] = lib.initial_quality(
                    _ObjectType.LINK, li, si
                )

        # ── Bronnen ───────────────────────────────────────────────────────────
        for ni in range(1, n_nodes + 1):
            node_name = lib.en_node_id(ni)
            for si, sp in enumerate(state.species, start=1):
                info = lib.source_info(ni, si)
                if info["type"] != _SourceKind.NONE:
                    state.sources.append(MsxSourceRecord(
                        node_index    = ni,
                        node_name     = node_name,
                        species_index = si,
                        species_name  = sp.name,
                        kind          = info["type"],
                        level         = info["level"],
                        pattern_index = info["pattern"],
                    ))

        # ── Patronen ──────────────────────────────────────────────────────────
        n_pat = lib.object_count(_ObjectType.PATTERN)
        for pi in range(1, n_pat + 1):
            plen = lib.pattern_length(pi)
            vals = [lib.pattern_value(pi, period) for period in range(1, plen + 1)]
            state.patterns[pi] = vals

        return state

    # ── Simulatie-controle ────────────────────────────────────────────────────

    def run(
        self,
        save_to_file : bool = False,
        hyd_file     : Optional[str] = None,
    ) -> MsxSimulationResult:
        """
        Voer de volledige simulatie uit en geef gestructureerde resultaten terug.

        Parameters
        ----------
        save_to_file : sla kwaliteitsresultaten op in een tijdelijk binair bestand
        hyd_file     : gebruik een eerder opgeslagen hydraulicabestand in plaats van
                       een nieuwe hydraulische berekening

        Geeft terug
        -----------
        MsxSimulationResult
        """
        if not self._loaded:
            self.load()

        lib = self._lib

        if hyd_file:
            lib.use_hydraulic_file(hyd_file)
        else:
            lib.solve_hydraulics()

        lib.initialize_run(save_to_file)

        state    = self._state
        n_nodes  = lib.en_node_count()
        n_links  = lib.en_link_count()
        n_sp     = len(state.species)

        node_names = [lib.en_node_id(i) for i in range(1, n_nodes + 1)]
        link_names = [lib.en_link_id(i) for i in range(1, n_links + 1)]

        # Pre-alloceer opslag (grote stap): schat tijdstappen
        time_list  : List[int]       = [0]
        node_frames: List[np.ndarray] = [state.node_initq.copy()]
        link_frames: List[np.ndarray] = [state.link_initq.copy()]

        # Stapsgewijze simulatielus
        t, tleft = 0, 1
        while tleft > 0:
            t, tleft = lib.advance()

            node_snap = np.zeros((n_nodes, n_sp))
            link_snap = np.zeros((n_links, n_sp))

            for ni in range(1, n_nodes + 1):
                for si in range(1, n_sp + 1):
                    node_snap[ni - 1, si - 1] = lib.concentration(
                        _ObjectType.NODE, ni, si
                    )

            for li in range(1, n_links + 1):
                for si in range(1, n_sp + 1):
                    link_snap[li - 1, si - 1] = lib.concentration(
                        _ObjectType.LINK, li, si
                    )

            time_list.append(t)
            node_frames.append(node_snap)
            link_frames.append(link_snap)

        return MsxSimulationResult(
            time_s       = np.array(time_list, dtype=float),
            node_quality = np.stack(node_frames, axis=0),   # (T, N, S)
            link_quality = np.stack(link_frames, axis=0),   # (T, L, S)
            species      = state.species,
            node_names   = node_names,
            link_names   = link_names,
        )

    # ── Schrijfhulpers ────────────────────────────────────────────────────────

    def update_initial_quality(
        self,
        node_values : Optional[Dict[str, Dict[str, float]]] = None,
        link_values : Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """
        Pas initiële concentraties aan voor knopen en/of leidingen.

        Parameters
        ----------
        node_values : {knoopnaam: {stofnaam: waarde}}
        link_values : {leidingnaam: {stofnaam: waarde}}
        """
        if not self._loaded:
            raise RuntimeError("Laad eerst het MSX-bestand via .load().")
        lib   = self._lib

        def _set(obj_type, items):
            if not items:
                return
            for name, sp_dict in items.items():
                idx = (
                    lib.en_node_index(name) if obj_type == _ObjectType.NODE
                    else lib.en_link_index(name) if obj_type == _ObjectType.LINK
                    else lib.object_index(obj_type, name)
                )
                for sp_name, value in sp_dict.items():
                    sp_idx = lib.object_index(_ObjectType.SPECIES, sp_name)
                    lib.set_initial_quality(obj_type, idx, sp_idx, value)

        _set(_ObjectType.NODE, node_values)
        _set(_ObjectType.LINK, link_values)

    def configure_source(
        self,
        node_name    : str,
        species_name : str,
        kind         : str,
        level        : float,
        pattern_name : Optional[str] = None,
    ) -> None:
        """
        Configureer of overschrijf een externe bron op een knoop.

        Parameters
        ----------
        node_name    : ID van de knoop
        species_name : naam van de stof
        kind         : ``'CONCEN'``, ``'MASS'``, ``'SETPOINT'``, ``'FLOWPACED'``
                       of ``'NOSOURCE'``
        level        : bronsterkte
        pattern_name : naam van het tijdpatroon (leeg = geen patroon)
        """
        if not self._loaded:
            raise RuntimeError("Laad eerst het MSX-bestand via .load().")
        lib      = self._lib
        kind_map = self._SOURCE_KIND_MAP
        src_code = kind_map.get(kind.upper(), _SourceKind.NONE)

        node_idx = lib.en_node_index(node_name)
        sp_idx   = lib.object_index(_ObjectType.SPECIES, species_name)
        pat_idx  = 0
        if pattern_name:
            pat_idx = lib.object_index(_ObjectType.PATTERN, pattern_name)

        lib.set_source(node_idx, sp_idx, src_code, level, pat_idx)

    def update_constant(self, name: str, value: float) -> None:
        """Wijzig de waarde van een named constant."""
        if not self._loaded:
            raise RuntimeError("Laad eerst het MSX-bestand via .load().")
        idx = self._lib.object_index(_ObjectType.CONSTANT, name)
        self._lib.set_constant(idx, value)

    def add_time_pattern(self, name: str, multipliers: List[float]) -> None:
        """Voeg een nieuw tijdpatroon toe en stel de vermenigvuldigers in."""
        if not self._loaded:
            raise RuntimeError("Laad eerst het MSX-bestand via .load().")
        self._lib.add_pattern(name)
        idx = self._lib.object_index(_ObjectType.PATTERN, name)
        self._lib.set_pattern(idx, multipliers)

    # ── Context-manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.load()
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        """Sluit de MSX-bibliotheek en het onderliggende EPANET-netwerk."""
        if self._loaded:
            try:
                self._lib.close()
            except MsxError:
                pass
            self._loaded = False
        if self._en_opened:
            self._lib.en_close()
            self._en_opened = False

    def __repr__(self) -> str:
        status = "geladen" if self._loaded else "niet geladen"
        return (
            f"MsxSimulation(inp={Path(self.inp_path).name!r}, "
            f"msx={Path(self.msx_path).name!r}, status={status!r})"
        )


# ══════════════════════════════════════════════════════════════════════════════
#  Gemaksfunctie
# ══════════════════════════════════════════════════════════════════════════════

def run_msx(
    inp_path    : str,
    msx_path    : str,
    lib_path    : Optional[str] = None,
    hyd_file    : Optional[str] = None,
) -> MsxSimulationResult:
    """
    Voer een MSX-simulatie in één aanroep uit.

    Voorbeeld
    ---------
    >>> result = run_msx("net2-cl2.inp", "net2-cl2.msx")
    >>> df = result.to_dataframe("CL2")
    """
    with MsxSimulation(inp_path, msx_path, lib_path) as sim:
        return sim.run(hyd_file=hyd_file)
