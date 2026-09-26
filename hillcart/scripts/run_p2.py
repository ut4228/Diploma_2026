"""Faza P2: kontrolni nalogi za preverjanje, da se agent sploh uci.

  P2(i)  crpanje brez omejitve palice  -> ali se nauci DOSECI cilj
  P2(ii) ravni tir z omejitvijo palice -> ali se nauci OHRANJATI ravnotezje in se premikati

To NISTA rezultata diplome, ampak kontrola implementacije (tile coding + Sarsa(lambda)).
Nobena od 35 glavnih konfiguracij se tu ne uporablja.

Uporaba:
  python scripts/run_p2.py                      # obe nalogi, semena 100-102
  python scripts/run_p2.py --task P2ii_flat_balance --seeds 100
  python scripts/run_p2.py --total-steps 100000 # krajsi preizkus
  python scripts/run_p2.py --alpha0 0.5         # vnaprej zapisani rezervni korak 1
  python scripts/run_p2.py --optimistic         # vnaprej zapisani rezervni korak 2 (q0 = +1)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda
from hillcart.env import ExperimentSpec, HillCartEnv
from hillcart.initial_states import eval_states
from hillcart.tile_coding import coder_from_bounds
from hillcart.tracking import Run
from hillcart.tracks import FlatTrack, PowerValley
from hillcart.training import train

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "plan_v1.json"
P2CFG = ROOT / "configs" / "p2_control.json"

QUICK_EVAL_STATES = [(0.0, 0.0, th, td) for th in (-0.05, 0.0, 0.05) for td in (-0.05, 0.0, 0.05)]
CRITERION = 0.80


def build_env(task_name: str, task: dict, spec: ExperimentSpec) -> HillCartEnv:
    t = task["track"]
    if t["type"] == "FlatTrack":
        track = FlatTrack(half_length=t["half_length"])
    else:
        track = PowerValley(H=t["H"], tan_phi_max=t["tan_phi_max"], p=t["p"])
    return HillCartEnv(spec, check_pole=task["check_pole"], track=track, F_max=task["F_max"],
                       tile_bounds=task["tile_bounds"])


def q_absmax_sampled(agent, coder, seed: int, n: int = 5000) -> float:
    """Dejanski max |Q| na nakljucnem vzorcu stanj (kriterij 4). Ni isto kot ohlapna meja
    n_tilings * max|w|, ki je bila prej poroCana."""
    rng = np.random.default_rng(seed)
    m = 0.0
    for _ in range(n):
        obs = tuple(rng.uniform(lo, hi) for lo, hi in zip(coder.lows, coder.highs))
        m = max(m, float(np.abs(agent.w[coder.features_all_actions(obs)].sum(axis=1)).max()))
    return m


def run_one(task_name: str, task: dict, spec: ExperimentSpec, seed: int, args, agent_cfg_json: dict) -> dict:
    coder = coder_from_bounds(task["tile_bounds"], n_tilings=agent_cfg_json["n_tilings"],
                              n_intervals=agent_cfg_json["n_intervals"])
    w_init = 1.0 / coder.cfg.n_tilings if args.optimistic else agent_cfg_json["w_init"]
    acfg = AgentConfig(alpha0=args.alpha0, lam=agent_cfg_json["lambda"], gamma=agent_cfg_json["gamma"],
                       epsilon=agent_cfg_json["epsilon"], w_init=w_init, eps_z=agent_cfg_json["eps_z"])
    agent = TrueOnlineSarsaLambda(coder, acfg, seed=seed)
    env = build_env(task_name, task, spec)

    cfg = {"phase": "P2", "task": task_name, "task_spec": task, "seed": seed,
           "plan_sha256": spec.plan_sha256, "p2_config_version": agent_cfg_json.get("_version"),
           "env": env.describe(), "agent": agent.describe(),
           "protocol": {"total_steps": args.total_steps, "eval_every": args.eval_every,
                        "quick_eval_states": len(QUICK_EVAL_STATES), "final_eval_states": len(eval_states()),
                        "criterion_success_rate": CRITERION},
           "fallback_used": {"alpha0": args.alpha0, "optimistic_init": args.optimistic}}
    run = Run(f"p2_{task_name}_s{seed}", cfg, base_dir=ROOT / "runs", seed=seed,
              require_clean=not args.allow_dirty, repo_dir=ROOT)

    def progress(step, ev):
        print(f"    {step:>8d} korakov  uspeh={ev['success_rate']:.2f}  "
              f"|w|max={np.abs(agent.w).max():.3f}  sledi={agent.max_active}", flush=True)

    rng = np.random.default_rng(seed)
    res = train(env, agent, total_steps=args.total_steps, eval_every=args.eval_every,
                eval_states_quick=QUICK_EVAL_STATES, eval_states_final=eval_states(),
                rng=rng, run=run, progress=progress)

    records = res.pop("records")
    with open(run.dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        rows = [r.as_row() for r in records]
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    run.save_json("evals.json", res["evals"])
    np.savez_compressed(run.dir / "weights_final.npz", w=agent.w)

    fin = res["final_eval"]
    summary = {"task": task_name, "seed": seed, "final_success_rate": fin["success_rate"],
               "final_outcomes": fin["outcomes"], "t_event_median": fin["t_event_median"],
               "t_event_max": fin["t_event_max"], "steps_median": fin["steps_median"],
               "episodes": res["episodes"], "env_steps": res["steps"],
               "first_success_step": res["first_success_step"], "train_outcomes": res["train_outcomes"],
               "w_absmax": res["w_absmax"], "max_active_traces": res["max_active_traces"],
               "train_oob_fraction": res["train_oob_fraction"],
               "eval_oob_steps": fin["oob_steps_total"],
               "q_absmax_sampled": q_absmax_sampled(agent, coder, seed),
               "q_absmax_upper_bound": float(np.abs(agent.w).max() * coder.cfg.n_tilings),
               "criterion_met": fin["success_rate"] >= CRITERION}
    run.close(summary)
    summary["run_dir"] = run.dir.name
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["P2i_pump_no_pole", "P2ii_flat_balance", "both"], default="both")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--total-steps", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--alpha0", type=float, default=None, help="rezervni korak 1: 0.1 -> 0.5")
    ap.add_argument("--optimistic", action="store_true", help="rezervni korak 2: w_i = 1/16 (q0 = +1)")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    if not P2CFG.exists():
        raise SystemExit(f"manjka {P2CFG}; najprej pozeni: python scripts/make_p2_config.py")
    p2 = json.loads(P2CFG.read_text(encoding="utf-8"))
    acfg = dict(p2["agent"])
    acfg["_version"] = p2["version"]
    proto = p2["protocol"]
    args.seeds = args.seeds or proto["seeds"]
    args.total_steps = args.total_steps or proto["total_steps"]
    args.eval_every = args.eval_every or proto["eval_every"]
    args.alpha0 = acfg["alpha0"] if args.alpha0 is None else args.alpha0

    spec = ExperimentSpec.from_plan(PLAN, p2["tasks"]["P2i_pump_no_pole"]["track"]["config_id"])
    tasks = list(p2["tasks"]) if args.task == "both" else [args.task]

    results = []
    for tname in tasks:
        print(f"\n=== {tname} ===  ({p2['tasks'][tname]['purpose']})")
        for seed in args.seeds:
            print(f"  seme {seed}:")
            results.append(run_one(tname, p2["tasks"][tname], spec, seed, args, acfg))

    print("\n" + "=" * 78)
    print(f"{'naloga':<20}{'seme':>6}{'uspeh':>8}{'t_med':>8}{'epizod':>9}{'1. uspeh':>11}{'|Q|max':>9}{'oob':>9}")
    for r in results:
        t = f"{r['t_event_median']:.2f}" if r["t_event_median"] is not None else "-"
        fs = str(r["first_success_step"]) if r["first_success_step"] is not None else "-"
        print(f"{r['task']:<20}{r['seed']:>6}{r['final_success_rate']:>8.3f}{t:>8}{r['episodes']:>9}"
              f"{fs:>11}{r['q_absmax_sampled']:>9.2f}{r['train_oob_fraction']:>9.4f}")

    ok = {}
    for tname in tasks:
        met = sum(1 for r in results if r["task"] == tname and r["criterion_met"])
        ok[tname] = met
        print(f"\n{tname}: kriterij (uspeh >= {CRITERION:.2f}) izpolnjen pri {met}/{len(args.seeds)} semenih"
              f"  -> {'IZPOLNJEN' if met >= 2 else 'NI IZPOLNJEN'}")
    if all(v >= 2 for v in ok.values()) and args.task == "both":
        print("\nP2 kriterija 2 in 3 sta izpolnjena.")
    else:
        print("\nPreveri rezervne korake iz protokola:", proto["fallbacks"])


if __name__ == "__main__":
    main()
