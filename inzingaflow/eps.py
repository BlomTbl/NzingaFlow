# inzingaflow/eps.py
from __future__ import annotations
import numpy as np
from typing import Callable


class EPSRunner:
    """
    Extended Period Simulation (EPS) runner voor inzingaflow.

    Kenmerken
    ---------
    - Hydraulische updates elke hyd_dt seconden
    - Automatische CFL-check voor de eerste tijdstap
    - inject_schedule voor tijdvariabele injectie (per knoop, per interval)
    - inject_fn callback voor dynamische injectie
    - Voortgangsmelding met massabalans (verbose=True)
    - time_axis() voor directe plot-as

    Gebruik
    -------
        runner = EPSRunner(solver, qual_dt=5.0, hyd_dt=300.0, duration=86400.0)
        results = runner.run(
            decay_k=np.array([0.001, 0.0]),
            inject_schedule={"R1": [(0, 3600, np.array([1.0, 1.0]))]},
            verbose=True,
        )
        # results.shape == (n_stappen, node_count, n_species)
    """

    def __init__(
        self,
        solver,
        qual_dt:  float,
        hyd_dt:   float,
        duration: float,
    ):
        self.solver   = solver
        self.qual_dt  = float(qual_dt)
        self.hyd_dt   = float(hyd_dt)
        self.duration = float(duration)

    def run(
        self,
        decay_k,
        inject_schedule:  dict | None = None,
        inject_fn:        Callable | None = None,
        booster_schedule: dict | None = None,
        merge_interval:   int   = 10,
        merge_tol:        float = 1e-6,
        check_cfl:        bool  = True,
        verbose:          bool  = False,
    ) -> np.ndarray:
        """
        Voer de volledige EPS-simulatie uit.

        Parameters
        ----------
        decay_k          : (n_species,) bulkvervalconstanten [1/s]
        inject_schedule  : {knoopnaam: [(t_start, t_end, C_vector), ...]}
                           Injectievolume = debiet × qual_dt per tijdstap.
        inject_fn        : callable(t: float, solver) — dynamische injectie.
                           Heeft voorrang boven inject_schedule.
        booster_schedule : {knoopnaam: [(t_start, t_end, C_set, flow_frac), ...]}
                           Booster-injectie: stel concentratie in op vaste waarde
                           op de opgegeven knoop. C_set[s] < 0 = stof s ongewijzigd.
                           flow_frac is optioneel (default 1.0).
        merge_interval   : segmentmerging elke N stappen
        merge_tol        : concentratietolerantie voor merging [mg/L]
        check_cfl        : CFL-check op eerste tijdstap
        verbose          : print voortgang + massabalans elke 10%

        Returns
        -------
        results : (n_stappen, node_count, n_species)
        """
        decay_k         = np.asarray(decay_k, dtype=np.float64)
        inject_schedule  = inject_schedule  or {}
        booster_schedule = booster_schedule or {}
        n_steps = int(self.duration / self.qual_dt)
        results = np.zeros(
            (n_steps, self.solver.node_count, self.solver.n_species),
            dtype=np.float64,
        )

        t        = 0.0
        next_hyd = 0.0
        verbose_every = max(1, n_steps // 10)

        for step in range(n_steps):

            # Hydraulica bijwerken
            if t >= next_hyd:
                self.solver.update_hydraulics(simtime=int(t))
                next_hyd += self.hyd_dt

            # Injectie
            if inject_fn is not None:
                inject_fn(t, self.solver)
            elif inject_schedule:
                self._apply_schedule(t, inject_schedule)

            # Booster-injectie
            if booster_schedule:
                self._apply_booster(t, booster_schedule)

            # Kwaliteitstijdstap
            node_C = self.solver.step(
                dt=self.qual_dt,
                decay_k=decay_k,
                merge_interval=merge_interval,
                merge_tol=merge_tol,
                check_cfl=(check_cfl and step == 0),
            )
            results[step] = node_C
            t += self.qual_dt

            # Voortgang
            if verbose and (step + 1) % verbose_every == 0:
                mb = self.solver.mass_balance()
                mb_str = ""
                if mb:
                    err = mb['balance_error']
                    mb_str = f"  massafout={err.max()*100:.3f}%"
                print(
                    f"  EPS {100*(step+1)/n_steps:5.1f}%  "
                    f"t={t/3600:.2f}h  segs={self.solver.segments.n}"
                    f"{mb_str}"
                )

        return results

    def _apply_booster(self, t: float, booster_schedule: dict) -> None:
        """
        Verwerk booster-injectieschema voor tijdstip t.

        Schema-formaat:
            {knoopnaam: [(t_start, t_end, C_set),               ...]}
          of
            {knoopnaam: [(t_start, t_end, C_set, flow_frac),    ...]}

        C_set[s] < 0 laat stof s ongewijzigd.
        """
        for node_uid, intervals in booster_schedule.items():
            for interval in intervals:
                t_start, t_end = interval[0], interval[1]
                if not (t_start <= t < t_end):
                    continue
                C_set = np.asarray(interval[2], dtype=np.float64)
                flow_frac = float(interval[3]) if len(interval) > 3 else 1.0
                self.solver.booster_inject(node_uid, C_set, flow_frac=flow_frac)

    def _apply_schedule(self, t: float, schedule: dict) -> None:
        """
        Verwerk injectieschema voor tijdstip t.

        Maakt gebruik van solver.inject() zodat de injectie proportioneel
        wordt verdeeld over alle uitgaande leidingen (bug-fix t.o.v. eerdere
        versie die alleen de leiding met het hoogste debiet bediende).
        """
        flow, _ = self.solver._get_hydraulics()
        for node_uid, intervals in schedule.items():
            for t_start, t_end, C_vec in intervals:
                if not (t_start <= t < t_end):
                    continue
                node_idx = self.solver.node_index.get(node_uid)
                if node_idx is None:
                    raise KeyError(f"Onbekende knoopnaam: '{node_uid}'")
                out = self.solver._node_outpipes.get(node_idx, [])
                if not out:
                    continue
                # Totaalvolume = som van (debiet × qual_dt) voor alle uitgaande leidingen
                tot_flow = sum(flow[p] for p in out)
                vol_total = max(tot_flow * self.qual_dt, 0.0)
                if vol_total > 0:
                    C_arr = np.asarray(C_vec, dtype=np.float64)
                    self.solver.inject(node_uid, C_arr, vol_total)

    def time_axis(self, unit: str = 's') -> np.ndarray:
        """Tijdas van simulatieresultaten. unit: 's', 'min' of 'h'."""
        t = np.arange(int(self.duration / self.qual_dt)) * self.qual_dt
        return t / {'s': 1, 'min': 60, 'h': 3600}.get(unit, 1)

    def __repr__(self) -> str:
        return (
            f"<EPSRunner qual_dt={self.qual_dt}s hyd_dt={self.hyd_dt}s "
            f"duration={self.duration/3600:.1f}h>"
        )
