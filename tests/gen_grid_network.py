"""Genereer een synthetisch grid-netwerk (EPANET .inp) voor profilering.

WAARSCHUWING (zie migratieopdracht): een dicht grid is geen representatieve
topologie voor het _handle_exits()-splitsingspad in solver.py — gebruik dit
alleen voor hydraulica-schaal-metingen (EN_openH, property-read-kosten).
"""
import sys

def generate_grid_inp(n_side: int, path: str) -> None:
    """n_side x n_side grid van junctions, plus 1 reservoir aan de rand."""
    lines = []
    lines.append("[TITLE]")
    lines.append(f"Synthetisch {n_side}x{n_side} grid-netwerk")
    lines.append("")
    lines.append("[JUNCTIONS]")
    for i in range(n_side):
        for j in range(n_side):
            uid = f"J{i}_{j}"
            lines.append(f" {uid}    0    1.0")
    lines.append("")
    lines.append("[RESERVOIRS]")
    lines.append(" R1    100")
    lines.append("")
    lines.append("[PIPES]")
    pipe_id = 1
    # Verbind reservoir met hoekpunt (0,0)
    lines.append(f" PIPE{pipe_id}    R1    J0_0    100    300    100    0    Open")
    pipe_id += 1
    for i in range(n_side):
        for j in range(n_side):
            uid = f"J{i}_{j}"
            if j + 1 < n_side:
                uid2 = f"J{i}_{j+1}"
                lines.append(f" PIPE{pipe_id}    {uid}    {uid2}    100    150    100    0    Open")
                pipe_id += 1
            if i + 1 < n_side:
                uid2 = f"J{i+1}_{j}"
                lines.append(f" PIPE{pipe_id}    {uid}    {uid2}    100    150    100    0    Open")
                pipe_id += 1
    lines.append("")
    lines.append("[TIMES]")
    lines.append(" Duration 0")
    lines.append("")
    lines.append("[OPTIONS]")
    lines.append(" Units LPS")
    lines.append("")
    lines.append("[COORDINATES]")
    lines.append(" R1  -100  0")
    for i in range(n_side):
        for j in range(n_side):
            uid = f"J{i}_{j}"
            lines.append(f" {uid}  {i*10}  {j*10}")
    lines.append("")
    lines.append("[REPORT]")
    lines.append(" Status  No")
    lines.append(" Summary No")
    lines.append("")
    lines.append("[END]")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    n_nodes = n_side * n_side + 1
    n_pipes = pipe_id - 1
    print(f"gegenereerd: {n_nodes} knopen, {n_pipes} leidingen -> {path}")


if __name__ == "__main__":
    n_side = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    out = sys.argv[2] if len(sys.argv) > 2 else "/tmp/grid_network.inp"
    generate_grid_inp(n_side, out)
