"""Ustvari eksperimentalni načrt configs/plan_v1.json (vhod za vse RL eksperimente).

Koraki:
  1. za vse konfiguracije (kotanja x kappa) izračuna sile F_E, F_krit, F_max, k, u,
  2. izmeri idealni čas črpanja BREZ palice (diagnostika, oba začetna potiska),
  3. določi začetni skupni T_max = zaokroži_navzgor(faktor * max t_pump, korak),
  4. za vsako konfiguracijo izmeri razpone stanj po virih (random / pumping_pole / pumping_nopole)
     in iz njih meje za tile coding (+ rezerva),
  5. zapiše načrt v configs/plan_v1.json (commitaj ga!) in kopijo + dnevnik v runs/.

Uporaba:  python scripts/make_plan.py [--episodes 100] [--margin 0.25] [--tmax-factor 2.0] [--allow-dirty]
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

from hillcart.diagnostics import bounds_from_ranges, pumping_reference, state_ranges
from hillcart.dynamics import PoleCartParams
from hillcart.experiment_grid import BASE, GEOMETRY_SERIES, KAPPA_VALUES, all_configs, series_of
from hillcart.initial_states import TRAIN_NOISE, eval_states
from hillcart.simulator import SimConfig
from hillcart.tracking import Run, git_info

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=100, help="epizod na vir (random, pumping_pole) na konfiguracijo")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--tmax-factor", type=float, default=2.0)
    ap.add_argument("--tmax-round", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "configs" / "plan_v1.json"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    prm = PoleCartParams()
    cfgs = all_configs()
    run = Run("make_plan", vars(args), base_dir=ROOT / "runs", seed=args.seed,
              require_clean=not args.allow_dirty, repo_dir=ROOT)

    # 1-2: sile in idealni čas črpanja brez palice
    rows = []
    for i, c in enumerate(cfgs):
        f = c.forces(prm)
        pr = pumping_reference(c.valley, f["F_max"], prm)
        rows.append({"id": c.id, "series": series_of(c), "valley": c.valley.describe(), "forces": f,
                     "pumping_reference_no_pole": pr})
        run.scalar("plan/t_pump_no_pole", pr["t_pump"] if math.isfinite(pr["t_pump"]) else -1, i)

    # 3: skupni T_max
    finite = [r["pumping_reference_no_pole"]["t_pump"] for r in rows
              if math.isfinite(r["pumping_reference_no_pole"]["t_pump"])]
    t_pump_max = max(finite)
    t_max = math.ceil(args.tmax_factor * t_pump_max / args.tmax_round) * args.tmax_round
    sim_cfg = SimConfig(t_max=t_max)

    # 4: meje po konfiguraciji
    for i, (c, r) in enumerate(zip(cfgs, rows)):
        rg = state_ranges(c.valley, r["forces"]["F_max"], sim_cfg, args.episodes, args.seed + i, prm)
        r["state_ranges"] = rg
        r["tile_bounds"] = {"x": [-c.valley.W, c.valley.W], "theta": [-sim_cfg.theta_max, sim_cfg.theta_max],
                            **bounds_from_ranges(rg, args.margin)}
        run.scalar("plan/x_dot_bound", r["tile_bounds"]["x_dot"][1], i)
        run.scalar("plan/theta_dot_bound", r["tile_bounds"]["theta_dot"][1], i)

    plan = {
        "version": "plan_v1",
        "generated_by": {"script": "scripts/make_plan.py", "git": git_info(ROOT), "args": vars(args),
                         "run_dir": str(run.dir.name)},
        "params": prm.describe(),
        "sim": {**asdict(sim_cfg), "max_steps": sim_cfg.max_steps,
                "t_max_rule": f"ceil({args.tmax_factor} * max t_pump_no_pole / {args.tmax_round}) * {args.tmax_round}",
                "t_pump_no_pole_max": t_pump_max, "t_max_status": "ZAČETNA VREDNOST - preveri v pilotu"},
        "actions": "{-F_max, 0, +F_max}",
        "design": {"base": BASE, "kappa_values": KAPPA_VALUES, "geometry_series": GEOMETRY_SERIES},
        "initial_states": {"train_noise_uniform": TRAIN_NOISE, "eval_states": eval_states(),
                           "note": "trening: enakomeren šum; vrednotenje: fiksna mreža 3^4 = 81 stanj, požrešna politika"},
        "seeds": {"tuning": [100, 101, 102, 103, 104], "main": list(range(10)),
                  "note": "semena za nastavljanje hiperparametrov se v glavnih zagonih ne uporabijo"},
        "phases": {
            "P_pilot": "osnovna kotanja, kappa 0.9 in 0.3, Sarsa(lambda)+TC, semena 100-101: preveri T_max, "
                       "proračun korakov in delež stanj zunaj tile_bounds; nato T_max in proračun ZAMRZNI",
            "T_tuning": "osnovna kotanja, kappa 0.5, semena 100-104, enak proračun iskanja za vse metode",
            "A_methods": "S0_kappa (osnovna kotanja x 5 kappa) x vse metode x semena 0-9",
            "B_geometry": "S1-S3 (6 dodatnih kotanj x 5 kappa) x referenčna metoda x semena 0-9; referenčna "
                          "metoda = najvišji povprečni delež uspeha v fazi A (pravilo določeno vnaprej)",
        },
        "metrics": {
            "primary": "delež uspeha na 81 fiksnih začetnih stanjih, požrešna politika, po koncu treninga",
            "secondary": ["mediana časa do cilja (uspešne epizode)", "razčlenitev izidov: fail_pole / fail_left / timeout",
                          "učna krivulja: delež uspeha na vmesnih vrednotenjih", "koraki okolja do prvega uspeha "
                          "in do >= 50 % uspeha", "število neuspelih učnih epizod", "delež stanj zunaj tile_bounds"],
            "aggregation": "povprečje čez semena + 95 % interval zaupanja (bootstrap)",
            "feasibility_boundary": "najmanjši kappa z mediano deleža uspeha >= 0.5 (in najmanjši kappa z vsaj enim uspehom)",
        },
        "configs": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    run.save_json("plan_v1.json", plan)
    run.close({"t_max": t_max, "t_pump_no_pole_max": t_pump_max, "n_configs": len(rows)})

    print(f"T_max = {t_max:g} s  (max t_pump brez palice = {t_pump_max:.1f} s, faktor {args.tmax_factor})")
    print(f"{'id':<28}{'W':>5}{'F_max':>7}{'k':>6}{'u':>6}{'t_pump':>8}{'obr.':>5}{'xd_b':>7}{'thd_b':>7}")
    for r in rows:
        f, pr, b = r["forces"], r["pumping_reference_no_pole"], r["tile_bounds"]
        print(f"{r['id']:<28}{r['valley']['W']:5.1f}{f['F_max']:7.2f}{f['k']:6.2f}{f['u']:6.3f}"
              f"{pr['t_pump']:8.1f}{str(pr['reversals']):>5}{b['x_dot'][1]:7.2f}{b['theta_dot'][1]:7.2f}")
    print("načrt:", out)
    print("dnevnik:", run.dir)


if __name__ == "__main__":
    main()
