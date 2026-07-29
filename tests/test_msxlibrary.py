# tests/test_msxlibrary.py
"""
Integratietests voor de MSX-native-library-brug (nzingaflow.msxlibrary).

Deze tests draaien tegen de echte epanetmsx/epanet2 gedeelde bibliotheken
(meegeleverd in nzingaflow/lib/) via het officiële EPANET-MSX
voorbeeldnetwerk (arseenoxidatie/-adsorptie, USEPA EPANETMSX Examples/).

In tegenstelling tot msx.py's MsxReactionSystem (een pure-Python
herimplementatie, gedekt door andere tests) hangt dit bestand af van de
gecompileerde native bibliotheek — vandaar de aparte marker.

Structuur
---------
M1  MsxNativeLib laadt en vindt de epanet2-companion-bibliotheek
M2  MsxSimulation.open() opent het EPANET-netwerk vóór MSXopen()
    (regressietest voor de bug waarbij ENopen() nooit werd aangeroepen)
M3  Species worden correct uitgelezen uit het .msx-bestand
M4  run() levert concentraties op voor alle gerapporteerde knopen
M5  configure_source() met een geldige node-ID werkt end-to-end
M6  configure_source() met een onbestaande node-ID geeft een nette MsxError
    (regressietest voor de bug waarbij node/link-opzoeking via de
    verkeerde — MSX-laag i.p.v. EPANET-laag — API liep)
M7  Context-manager sluit zowel de MSX- als de EN-bibliotheek af

Alle tests zijn gemarkeerd met @pytest.mark.requires_msx en slaan zichzelf
over (skip) als de native bibliotheken niet gevonden kunnen worden, zodat
de rest van de testsuite op een machine zonder deze binaries blijft draaien.
"""
from __future__ import annotations

import textwrap

import pytest

msxlibrary = pytest.importorskip("nzingaflow.msxlibrary")

pytestmark = pytest.mark.requires_msx


def _lib_available() -> bool:
    try:
        msxlibrary._default_lib_path()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.requires_msx,
    pytest.mark.skipif(
        not _lib_available(),
        reason="epanetmsx/epanet2 gedeelde bibliotheken niet gevonden voor dit platform",
    ),
]


# ── Voorbeeldnetwerk: arseenoxidatie/-adsorptie (officiële EPANET-MSX Examples/) ──
_EXAMPLE_INP = textwrap.dedent("""\
    [TITLE]
    EPANET-MSX Example Network

    [JUNCTIONS]
    ;ID    Elev  Demand  Pattern
     A     0     4.1
     B     0     3.4
     C     0     5.5
     D     0     2.3

    [RESERVOIRS]
    ;ID      Head
     Source   100

    [PIPES]
    ;ID  Node1   Node2  Length  Diameter  Roughness
     1   Source  A      1000    12        100
     2   A       B      1000    10        100
     3   A       C      1000    10        100
     4   B       D      1000    8         100
     5   C       D      1000    8         100

    [TIMES]
     DURATION      48:00
     HYDRAULIC TIMESTEP  1:00
     QUALITY TIMESTEP    0:05

    [REPORT]
     STATUS  NO
     SUMMARY NO

    [END]
""")

_EXAMPLE_MSX = textwrap.dedent("""\
    [TITLE]
    Arsenic Oxidation/Adsorption Example

    [OPTIONS]
      AREA_UNITS M2
      RATE_UNITS HR
      SOLVER     RK5
      TIMESTEP   360
      RTOL       0.001
      ATOL       0.0001

    [SPECIES]
      BULK AS3   UG
      BULK AS5   UG
      BULK AStot UG
      WALL AS5s  UG
      BULK NH2CL MG

    [COEFFICIENTS]
      CONSTANT Ka   10.0
      CONSTANT Kb   0.1
      CONSTANT K1   5.0
      CONSTANT K2   1.0
      CONSTANT Smax 50

    [TERMS]
      Ks       K1/K2

    [PIPES]
      RATE    AS3    -Ka*AS3*NH2CL
      RATE    AS5    Ka*AS3*NH2CL - Av*(K1*(Smax-AS5s)*AS5 - K2*AS5s)
      RATE    NH2CL  -Kb*NH2CL
      EQUIL   AS5s   Ks*Smax*AS5/(1+Ks*AS5) - AS5s
      FORMULA AStot  AS3 + AS5

    [TANKS]
      RATE    AS3    -Ka*AS3*NH2CL
      RATE    AS5    Ka*AS3*NH2CL
      RATE    NH2CL  -Kb*NH2CL
      FORMULA AStot  AS3 + AS5

    [QUALITY]
      NODE    Source AS3   10.0
      NODE    Source NH2CL 2.5

    [REPORT]
      NODES   C   D
      LINKS   5
      SPECIES  AStot YES
      SPECIES  AS5   YES
      SPECIES  AS5s  YES
      SPECIES  NH2CL YES
""")


@pytest.fixture
def example_network(tmp_path):
    """Schrijft het voorbeeldnetwerk naar tijdelijke .inp/.msx-bestanden."""
    inp_path = tmp_path / "example.inp"
    msx_path = tmp_path / "example.msx"
    inp_path.write_text(_EXAMPLE_INP)
    msx_path.write_text(_EXAMPLE_MSX)
    return str(inp_path), str(msx_path)


# ── M1 ───────────────────────────────────────────────────────────────────────
def test_native_lib_loads_and_finds_companion():
    """MsxNativeLib laadt zonder handmatig lib_path en vindt epanet2 ernaast."""
    lib = msxlibrary.MsxNativeLib()
    assert lib._en is not None


# ── M2 ───────────────────────────────────────────────────────────────────────
def test_simulation_opens_epanet_network_before_msx(example_network):
    """
    Regressietest: _epanet_open() moet ENopen() daadwerkelijk aanroepen,
    en dat gebeurt al bij constructie — vóór het (aparte) laden van het
    MSX-bestand zelf via .load()/__enter__.

    Vóór de fix was _epanet_open() een no-op — MSXopen() faalde dan stil
    of gaf onzinresultaten omdat de gedeelde EPANET-netwerkstate nooit
    gevuld werd.
    """
    inp_path, msx_path = example_network
    sim = msxlibrary.MsxSimulation(inp_path, msx_path)
    try:
        assert sim._en_opened is True
        assert sim._loaded is False  # .load() is nog niet aangeroepen
        sim.load()
        assert sim._loaded is True
    finally:
        sim.close()


# ── M3 ───────────────────────────────────────────────────────────────────────
def test_species_parsed_from_msx_file(example_network):
    inp_path, msx_path = example_network
    with msxlibrary.MsxSimulation(inp_path, msx_path) as sim:
        names = {sp.name for sp in sim._state.species}
        assert names == {"AS3", "AS5", "AStot", "AS5s", "NH2CL"}


# ── M4 ───────────────────────────────────────────────────────────────────────
def test_run_returns_concentrations_for_all_nodes(example_network):
    inp_path, msx_path = example_network
    with msxlibrary.MsxSimulation(inp_path, msx_path) as sim:
        result = sim.run()
        conc = result.node_concentrations("AS3")
        # 5 knopen: junctions A, B, C, D + het reservoir Source.
        assert conc.shape[0] > 0
        assert conc.shape[1] == 5
        assert (conc >= 0).all()


# ── M5 / M6 ──────────────────────────────────────────────────────────────────
def test_configure_source_valid_node(example_network):
    inp_path, msx_path = example_network
    with msxlibrary.MsxSimulation(inp_path, msx_path) as sim:
        sim.configure_source("A", "AS3", "CONCEN", 10.0)  # mag niet raisen
        result = sim.run()
        assert result.node_concentrations("AS3").shape[0] > 0


def test_configure_source_unknown_node_raises_clean_error(example_network):
    """
    Regressietest: node-opzoeking liep voorheen via MSXgetindex(NODE, ...),
    wat MSX niet ondersteunt (alleen SPECIES/CONSTANT/PARAMETER/PATTERN) en
    dus altijd faalde — ook voor geldige node-namen. Na de fix (via
    ENgetnodeindex) geeft een écht onbestaande node een correcte,
    door ENgeterror vertaalde foutmelding.
    """
    inp_path, msx_path = example_network
    with msxlibrary.MsxSimulation(inp_path, msx_path) as sim:
        with pytest.raises(msxlibrary.MsxError) as excinfo:
            sim.configure_source("DOES_NOT_EXIST", "AS3", "CONCEN", 10.0)
        assert "node" in str(excinfo.value).lower()


# ── M7 ───────────────────────────────────────────────────────────────────────
def test_context_manager_closes_both_libraries(example_network):
    inp_path, msx_path = example_network
    sim = msxlibrary.MsxSimulation(inp_path, msx_path)
    with sim:
        pass
    assert sim._loaded is False
    assert sim._en_opened is False
