Meegeleverde binaries — herkomst en bouwinstructies
=====================================================

Deze map bevat de gedeelde bibliotheken voor `msxlibrary.py`
(`MsxSimulation`/`MsxNativeLib`), gebouwd vanuit de officiële
EPANET-MSX 2.0 broncode: https://github.com/USEPA/EPANETMSX

Aanwezig
--------
- `libepanetmsx.so` / `libepanet2_msx.so`   — Linux x86-64
- `epanetmsx.dll`   / `epanet2_msx.dll`     — Windows x86-64
                                               (statisch gelinkt tegen de
                                               mingw-runtime; geen losse
                                               libgcc/libgomp-dll's nodig)

Ontbrekend
----------
- macOS (`.dylib`) — niet meegebouwd. EPANETMSX's `CMakeLists.txt`
  ondersteunt dit wel (zie hun eigen CI-matrix op `macos-latest`/
  `macos-13`); bouw lokaal op een Mac met:

    git clone https://github.com/USEPA/EPANETMSX.git
    cd EPANETMSX
    git submodule update --init          # haalt EPANET2.2 op
    brew install libomp                  # OpenMP-runtime
    cmake -B build .
    cmake --build build --config Release --target package

  Kopieer daarna `libepanetmsx.dylib` en `libepanet2_msx.dylib` (zie
  naamgeving hieronder) naar deze map.

Waarom de "_msx"-naamgeving op de epanet2-companion?
------------------------------------------------------
`libepanetmsx`/`epanetmsx.dll` zijn dynamisch gelinkt tegen een eigen
epanet2-bibliotheek (niet statisch ingebakken). Deze heet bewust
`libepanet2_msx.so` / `epanet2_msx.dll` — NIET `libepanet2.so` /
`epanet2.dll` — omdat die laatste namen zouden botsen met epynet's
eigen bundled epanet2-bibliotheek (`epynet/lib/libepanet.so`, die
intern dezelfde SONAME `libepanet2.so` gebruikt) zodra beide packages
in hetzelfde proces geladen zijn, zoals in NzingaFlow. Zowel op Linux
(via SONAME) als op Windows (via de "reeds geladen module met deze
naam wordt hergebruikt, ongeacht map"-regel uit Microsoft's Dynamic-
Link Library Search Order) resolvt een botsende naam anders stilzwijgend
naar de verkeerde, al-geladen bibliotheek. Bouw je zelf een macOS-versie,
geef de companion-dylib dezelfde `_msx`-suffix om dit te vermijden.

Versie-compatibiliteit
-----------------------
Deze wrapper is geschreven en getest tegen EPANET-MSX **2.0**, niet
1.1. De MSX_* C-API is tussen die versies op punten gewijzigd (zie de
docstring bovenaan `msxlibrary.py`). Vervang de binaries hier niet
door een 1.1-build zonder de ctypes-signaturen opnieuw te controleren.
