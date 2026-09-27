"""Faza A: glavni eksperiment - serija S0_kappa na osnovni kotanji.

5 vrednosti kappa x 10 semen = 50 zagonov. To ni več pilot: rezultat je meja izvedljivosti,
osrednja ugotovitev diplome.

Zamrznjena konfiguracija:
  metoda        true online Sarsa(lambda) + tile coding (16 plositev x 9^4 plosic na akcijo)
  nagrada       R-C + potencialno oblikovanje Phi_E (c = 1), Ng, Harada & Russell (1999)
  ohlajanje     alpha0 0.5 -> 0.05, epsilon 0.05 -> 0 (linearno cez proracun)   [P3c]
  inicializacija optimisticna, w = 1/16 (q0 = +1)                               [P2]
  lambda, gamma 0.9, 0.999                                                      [P2]
  proracun      3e6 korakov, vrednotenje vsakih 1e5                             [P3]
  semena        0-9 (glavna; semena 100-104 so bila za nastavljanje in se tu ne uporabijo)

Uporaba:
  python scripts/run_phase_a.py                       # vseh 50 zagonov (nadaljuje, kjer je ostal)
  python scripts/run_phase_a.py --configs H1_tan0.5_p2_kap0.3 --seeds 0 1 2
  python scripts/run_phase_a.py --dry-run             # samo pokaze, kaj bi pognal
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
import time
from pathlib import Path

import numpy as np

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda
from hillcart.env import EnvOutcome, ExperimentSpec, HillCartEnv
from hillcart.initial_states import eval_states
from hillcart.shaping import EnergyPotential
from hillcart.tile_coding import coder_from_bounds
from hillcart.tracking import Run
from hillcart.training import train

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAN = ROOT / "configs" / "plan_v2.json"

PHASE = "A"
CONFIG_IDS = ("H1_tan0.5_p2_kap1.2", "H1_tan0.5_p2_kap0.9", "H1_tan0.5_p2_kap0.7",
              "H1_tan0.5_p2_kap0.5", "H1_tan0.5_p2_kap0.3")
SEEDS = tuple(range(10))
TOTAL_STEPS = 3_000_000
EVAL_EVERY = 100_000
REWARD = "shaped"
SHAPING_C = 1.0
HPARAMS = {"alpha0": 0.5, "lambda": 0.9, "gamma": 0.999, "epsilon": 0.05,
           "optimistic_init": True, "eps_z": 1e-6, "n_tilings": 16, "n_intervals": 8,
           "anneal": True, "alpha_final_frac": 0.1, "epsilon_final_frac": 0.0,
           "source": "zamrznjeno po fazah P2, P3 in P3c"}
QUICK_EVAL_STATES = [(0.0, 0.0, th, td) for th in (-0.05, 0.0, 0.05) for td in (-0.05, 0.0, 0.05)]
FEASIBLE_THRESHOLD = 0.5  # mediana deleza uspeha, pri kateri kappa steje za izvedljivo


def done_runs(base: Path, total_steps: int) -> dict:
    """Ze koncani zagoni faze A z ISTIM proracunom: (config_id, seed) -> povzetek.
    Zagon s krajsim proracunom (npr. preizkus) se NE steje kot koncan."""
    out = {}
    for d in sorted(glob.glob(str(base / "*phaseA_*"))):
        p = Path(d)
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
            summ = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if cfg.get("phase") != PHASE or "final_success_rate" not in summ:
            continue
        if cfg.get("protocol", {}).get("total_steps") != total_steps:
            continue
        summ["run_dir"] = p.name
        out[(summ["config_id"], summ["seed"])] = summ
    return out


def run_one(spec: ExperimentSpec, seed: int, args) -> dict:
    bounds = {"x": [-spec.valley.W, spec.valley.W],
              "theta": [-spec.sim.theta_max, spec.sim.theta_max],
              "x_dot": spec.tile_bounds["x_dot"], "theta_dot": spec.tile_bounds["theta_dot"]}
    coder = coder_from_bounds(bounds, n_tilings=HPARAMS["n_tilings"], n_intervals=HPARAMS["n_intervals"])
    acfg = AgentConfig(alpha0=HPARAMS["alpha0"], lam=HPARAMS["lambda"], gamma=HPARAMS["gamma"],
                       epsilon=HPARAMS["epsilon"], w_init=1.0 / coder.cfg.n_tilings,
                       eps_z=HPARAMS["eps_z"], anneal=HPARAMS["anneal"],
                       alpha_final_frac=HPARAMS["alpha_final_frac"],
                       epsilon_final_frac=HPARAMS["epsilon_final_frac"])
    agent = TrueOnlineSarsaLambda(coder, acfg, seed=seed)
    env = HillCartEnv(spec, check_pole=True, tile_bounds=bounds)
    shaping = EnergyPotential.for_env(env, c=args.c)

    cfg = {"phase": PHASE, "series": "S0_kappa", "config_id": spec.config_id, "seed": seed,
           "reward": REWARD, "shaping": shaping.describe(),
           "plan": {"path": Path(args.plan).name, "sha256": spec.plan_sha256,
                    "version": spec.plan_version},
           "hparams": HPARAMS, "tile_bounds": bounds,
           "env": env.describe(), "agent": agent.describe(),
           "protocol": {"total_steps": args.total_steps, "eval_every": args.eval_every,
                        "final_eval_states": len(eval_states()),
                        "primary_metric": "koncni delez uspeha (z ohlajanjem konvergira)",
                        "feasible_threshold": FEASIBLE_THRESHOLD}}
    run = Run(f"phaseA_{spec.config_id}_s{seed}", cfg, base_dir=ROOT / "runs", seed=seed,
              require_clean=not args.allow_dirty, repo_dir=ROOT)

    t0 = time.time()
    last = {"step": 0}

    def progress(step, ev):
        if step - last["step"] >= args.report_every:
            last["step"] = step
            print(f"      {step / 1e6:.1f}e6  uspeh={ev['success_rate']:.2f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)

    res = train(env, agent, total_steps=args.total_steps, eval_every=args.eval_every,
                eval_states_quick=QUICK_EVAL_STATES, eval_states_final=eval_states(),
                rng=np.random.default_rng(seed), run=run, progress=progress, shaping=shaping)

    records = res.pop("records")
    with open(run.dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        rows = [r.as_row() for r in records]
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    run.save_json("evals.json", res["evals"])
    np.savez_compressed(run.dir / "weights_final.npz", w=agent.w)

    fin = res["final_eval"]
    rates = [e["success_rate"] for e in res["evals"]]
    summary = {"config_id": spec.config_id, "seed": seed, "kappa": spec.forces["kappa"],
               "F_max": spec.F_max, "k": spec.forces["k"], "u": spec.forces["u"],
               "final_success_rate": fin["success_rate"],
               "late_success_rate": sum(rates[-5:]) / min(5, len(rates)) if rates else 0.0,
               "best_success_rate": max(rates) if rates else 0.0,
               "final_outcomes": fin["outcomes"], "exit_sides": fin["exit_sides"],
               "t_event_median": fin["t_event_median"], "t_event_max": fin["t_event_max"],
               "steps_median": fin["steps_median"], "t_event_fail_median": fin["t_event_fail_median"],
               "timeout_reach_median": fin["timeout_reach_median"],
               "episodes": res["episodes"], "env_steps": res["steps"],
               "first_success_step": res["first_success_step"], "w_absmax": res["w_absmax"],
               "train_oob_fraction": res["train_oob_fraction"],
               "wall_time_s": round(time.time() - t0, 1)}
    run.close(summary)
    summary["run_dir"] = run.dir.name
    return summary


def report(results: list[dict], out_csv: Path | None) -> None:
    print("\n" + "=" * 100)
    print(f"{'kappa':>6}{'F_max':>8}{'n':>4}{'povprecje':>11}{'mediana':>9}{'min':>7}{'max':>7}"
          f"{'st.odklon':>11}{'uspelo':>9}{'t_med':>8}{'1.uspeh med.':>14}")
    by_kappa = {}
    for r in results:
        by_kappa.setdefault(r["kappa"], []).append(r)
    rows = []
    for kappa in sorted(by_kappa, reverse=True):
        g = by_kappa[kappa]
        v = [x["final_success_rate"] for x in g]
        solved = sum(1 for x in v if x >= FEASIBLE_THRESHOLD)
        tm = [x["t_event_median"] for x in g if x["t_event_median"] is not None]
        fss = [x["first_success_step"] for x in g if x["first_success_step"] is not None]
        row = {"kappa": kappa, "F_max": g[0]["F_max"], "n": len(g), "mean": statistics.mean(v),
               "median": statistics.median(v), "min": min(v), "max": max(v),
               "sd": statistics.stdev(v) if len(v) > 1 else 0.0, "solved": solved,
               "t_event_median": statistics.median(tm) if tm else None,
               "first_success_median": statistics.median(fss) if fss else None}
        rows.append(row)
        t = f"{row['t_event_median']:.1f}" if row["t_event_median"] is not None else "-"
        fs = f"{row['first_success_median']:.0f}" if row["first_success_median"] is not None else "-"
        print(f"{kappa:>6.1f}{row['F_max']:>8.3f}{len(g):>4}{row['mean']:>11.3f}{row['median']:>9.3f}"
              f"{row['min']:>7.3f}{row['max']:>7.3f}{row['sd']:>11.3f}{solved:>6}/{len(g):<3}{t:>8}{fs:>14}")

    feasible = [r["kappa"] for r in rows if r["median"] >= FEASIBLE_THRESHOLD]
    print(f"\nmeja izvedljivosti (mediana deleza uspeha >= {FEASIBLE_THRESHOLD}):")
    if feasible:
        print(f"  najmanjsi izvedljiv kappa = {min(feasible):.1f}")
        infeasible = [r["kappa"] for r in rows if r["median"] < FEASIBLE_THRESHOLD and r["kappa"] < min(feasible)]
        if infeasible:
            print(f"  najvecji NEizvedljiv kappa = {max(infeasible):.1f}  ->  meja je med "
                  f"{max(infeasible):.1f} in {min(feasible):.1f}")
    else:
        print("  nobena vrednost kappa ni dosegla praga")
    any_succ = [r["kappa"] for r in rows if r["max"] > 0.0]
    if any_succ:
        print(f"  najmanjsi kappa z vsaj enim uspehom = {min(any_succ):.1f}")
    print("\nVse trditve veljajo ZA DANI PRORACUN UCENJA in dano metodo; 'ni resil' ni dokaz "
          "teoreticne neizvedljivosti.")

    if out_csv:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(results[0]))
            wr.writeheader()
            wr.writerows(results)
        print(f"\nzapisano: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(DEFAULT_PLAN))
    ap.add_argument("--configs", nargs="+", default=list(CONFIG_IDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--report-every", type=int, default=500_000)
    ap.add_argument("--c", type=float, default=SHAPING_C)
    ap.add_argument("--no-resume", action="store_true", help="ne preskoci ze koncanih zagonov")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--csv", default=str(ROOT / "runs" / "phase_a_summary.csv"))
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    existing = {} if args.no_resume else done_runs(ROOT / "runs", args.total_steps)
    todo = [(c, s) for c in args.configs for s in args.seeds if (c, s) not in existing]
    print(f"faza A: {len(args.configs)} konfiguracij x {len(args.seeds)} semen = "
          f"{len(args.configs) * len(args.seeds)} zagonov")
    print(f"  ze koncanih: {len(existing)}   za pognati: {len(todo)}")
    if args.dry_run:
        for c, s in todo:
            print(f"    {c}  seme {s}")
        return

    results = [existing[(c, s)] for c in args.configs for s in args.seeds if (c, s) in existing]
    t_start = time.time()
    for i, (cid, seed) in enumerate(todo, 1):
        spec = ExperimentSpec.from_plan(args.plan, cid)
        el = time.time() - t_start
        eta = f", ocena konca cez {(el / (i - 1)) * (len(todo) - i + 1) / 60:.0f} min" if i > 1 else ""
        print(f"\n[{i}/{len(todo)}] {cid} seme {seed}  (kappa={spec.forces['kappa']}, "
              f"F_max={spec.F_max:.3f} N{eta})", flush=True)
        results.append(run_one(spec, seed, args))

    if results:
        report(results, Path(args.csv) if args.csv else None)


if __name__ == "__main__":
    main()
