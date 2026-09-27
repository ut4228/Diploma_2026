"""Faza P3b: primerjava R-C in R-C + potencialno oblikovanje Phi_E (Ng, Harada & Russell 1999).

Motivacija (rezultat faze P3): pri kappa = 0.9 je varianca med semeni ogromna - isti problem
konca kot resitev, kot balansiranje na dnu ali kot hitro padanje, odvisno samo od semena.
Faza A z R-C bi zato merila predvsem sum. P3b preveri, ali potencialno oblikovanje po energiji
to varianco zmanjsa.

Zasnova: ISTI konfiguraciji in ISTA semena kot v P3, ISTI proracun korakov, spremeni se SAMO
nagrada. To je primerjava enega dejavnika.

  R-C        +1 uspeh, -1 padec, 0 sicer
  R-C+Phi_E  isto + F = gamma Phi(s') - Phi(s),  Phi_E = c (E - E_0)/(M g H)

Metrike se VEDNO racunajo na neoblikovani nagradi R; oblikovana R' se belezi posebej.
Vrednotenje je neoblikovano in pozresno, torej neposredno primerljivo med obema nagradama.

Uporaba:
  python scripts/run_p3b.py                       # 2 konfiguraciji x 3 semena x 2 nagradi
  python scripts/run_p3b.py --total-steps 1000000 # hitrejsi prvi pogled
  python scripts/run_p3b.py --rewards shaped      # samo oblikovana
  python scripts/run_p3b.py --c 0.5               # utez potenciala (OQ-4)
"""
from __future__ import annotations

import argparse
import csv
import statistics
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
CONFIG_IDS = ("H1_tan0.5_p2_kap0.9", "H1_tan0.5_p2_kap0.3")
SEEDS = (100, 101, 102)
TOTAL_STEPS = 3_000_000  # proračun faze A (odlocitev po P3)
EVAL_EVERY = 100_000
HPARAMS = {"alpha0": 0.5, "lambda": 0.9, "gamma": 0.999, "epsilon": 0.05,
           "optimistic_init": True, "eps_z": 1e-6, "n_tilings": 16, "n_intervals": 8,
           "source": "zamrznjeno po fazi P2 (oznaka p2-v1)"}
QUICK_EVAL_STATES = [(0.0, 0.0, th, td) for th in (-0.05, 0.0, 0.05) for td in (-0.05, 0.0, 0.05)]


def q_absmax_visited(env, agent, states) -> float:
    m = 0.0
    for s0 in states:
        obs = env.reset(s0)
        while env.outcome is EnvOutcome.RUNNING:
            m = max(m, float(np.abs(agent.q_values(obs)).max()))
            obs, *_ = env.step(agent.act(obs, greedy=True))
        m = max(m, float(np.abs(agent.q_values(obs)).max()))
    return m


def run_one(spec: ExperimentSpec, seed: int, reward_name: str, args) -> dict:
    bounds = {"x": [-spec.valley.W, spec.valley.W],
              "theta": [-spec.sim.theta_max, spec.sim.theta_max],
              "x_dot": spec.tile_bounds["x_dot"], "theta_dot": spec.tile_bounds["theta_dot"]}
    coder = coder_from_bounds(bounds, n_tilings=HPARAMS["n_tilings"], n_intervals=HPARAMS["n_intervals"])
    w_init = 1.0 / coder.cfg.n_tilings if HPARAMS["optimistic_init"] else 0.0
    acfg = AgentConfig(alpha0=args.alpha0, lam=HPARAMS["lambda"], gamma=HPARAMS["gamma"],
                       epsilon=HPARAMS["epsilon"], w_init=w_init, eps_z=HPARAMS["eps_z"],
                       anneal=args.anneal)
    agent = TrueOnlineSarsaLambda(coder, acfg, seed=seed)
    env = HillCartEnv(spec, check_pole=True, tile_bounds=bounds)
    shaping = EnergyPotential.for_env(env, c=args.c) if reward_name == "shaped" else None

    cfg = {"phase": "P3b", "config_id": spec.config_id, "seed": seed, "reward": reward_name,
           "shaping": shaping.describe() if shaping else None,
           "plan": {"path": Path(args.plan).name, "sha256": spec.plan_sha256,
                    "version": spec.plan_version},
           "hparams": {**HPARAMS, "alpha0": args.alpha0, "anneal": args.anneal}, "tile_bounds": bounds,
           "env": env.describe(), "agent": agent.describe(),
           "protocol": {"total_steps": args.total_steps, "eval_every": args.eval_every,
                        "metrics_on": "neoblikovana nagrada R; vrednotenje pozresno in neoblikovano"}}
    run = Run(f"p3b_{reward_name}_{spec.config_id}_s{seed}", cfg, base_dir=ROOT / "runs", seed=seed,
              require_clean=not args.allow_dirty, repo_dir=ROOT)

    def progress(step, ev):
        print(f"      {step:>9d}  uspeh={ev['success_rate']:.2f}  padci={ev['outcomes'].get('fail_pole', 0):>2d}"
              f"  timeout={ev['outcomes'].get('timeout', 0):>2d}  |w|max={np.abs(agent.w).max():.3f}", flush=True)

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
    summary = {"config_id": spec.config_id, "seed": seed, "reward": reward_name,
               "kappa": spec.forces["kappa"], "c": args.c if shaping else None,
               "final_success_rate": fin["success_rate"], "final_outcomes": fin["outcomes"],
               "exit_sides": fin["exit_sides"], "t_event_median": fin["t_event_median"],
               "t_event_max": fin["t_event_max"], "t_event_fail_median": fin["t_event_fail_median"],
               "timeout_reach_median": fin["timeout_reach_median"],
               "episodes": res["episodes"], "env_steps": res["steps"],
               "first_success_step": res["first_success_step"],
               "q_absmax_visited": q_absmax_visited(env, agent, eval_states()),
               "w_absmax": res["w_absmax"], "train_oob_fraction": res["train_oob_fraction"],
               "anneal": args.anneal, "alpha0_final": res["alpha0_final"],
               "epsilon_final": res["epsilon_final"]}
    run.close(summary)
    summary["run_dir"] = run.dir.name
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(DEFAULT_PLAN))
    ap.add_argument("--configs", nargs="+", default=list(CONFIG_IDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--rewards", nargs="+", choices=["plain", "shaped"], default=["plain", "shaped"])
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--alpha0", type=float, default=HPARAMS["alpha0"])
    ap.add_argument("--c", type=float, default=1.0, help="utez potenciala Phi_E (OQ-4)")
    ap.add_argument("--anneal", action="store_true",
                    help="linearno ohlajanje alpha0 (-> 10 %%) in epsilon (-> 0) cez proracun")
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    results = []
    for cid in args.configs:
        spec = ExperimentSpec.from_plan(args.plan, cid)
        print(f"\n=== {cid} ===  kappa={spec.forces['kappa']}, F_max={spec.F_max:.3f} N, "
              f"W={spec.valley.W:.1f} m, T_max={spec.sim.t_max:g} s")
        for reward_name in args.rewards:
            label = ("R-C" if reward_name == "plain" else f"R-C + Phi_E (c={args.c:g})") \
                + (" + ohlajanje" if args.anneal else "")
            print(f"  nagrada: {label}")
            for seed in args.seeds:
                print(f"    seme {seed}:")
                results.append(run_one(spec, seed, reward_name, args))

    print("\n" + "=" * 92)
    print(f"{'konfiguracija':<24}{'nagrada':>9}{'seme':>6}{'uspeh':>8}{'t_med':>7}{'padci':>7}"
          f"{'timeout':>9}{'1.uspeh':>10}{'|Q|':>6}")
    for r in results:
        o = r["final_outcomes"]
        tm = f"{r['t_event_median']:.1f}" if r["t_event_median"] is not None else "-"
        fs = str(r["first_success_step"]) if r["first_success_step"] is not None else "-"
        print(f"{r['config_id']:<24}{r['reward']:>9}{r['seed']:>6}{r['final_success_rate']:>8.3f}{tm:>7}"
              f"{o.get('fail_pole', 0):>7}{o.get('timeout', 0):>9}{fs:>10}{r['q_absmax_visited']:>6.2f}")

    print("\n" + "-" * 92)
    print(f"{'konfiguracija':<24}{'nagrada':>9}{'povprecje':>11}{'mediana':>9}{'min':>7}{'max':>7}"
          f"{'razpon':>8}{'st.odklon':>11}")
    agg = {}
    for cid in args.configs:
        for reward_name in args.rewards:
            vals = [r["final_success_rate"] for r in results
                    if r["config_id"] == cid and r["reward"] == reward_name]
            if not vals:
                continue
            sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
            agg[(cid, reward_name)] = (statistics.mean(vals), max(vals) - min(vals), sd)
            print(f"{cid:<24}{reward_name:>9}{statistics.mean(vals):>11.3f}{statistics.median(vals):>9.3f}"
                  f"{min(vals):>7.3f}{max(vals):>7.3f}{max(vals) - min(vals):>8.3f}{sd:>11.3f}")

    print("\nprimerjava (oblikovanje proti R-C):")
    for cid in args.configs:
        a, b = agg.get((cid, "plain")), agg.get((cid, "shaped"))
        if not (a and b):
            continue
        d_mean, d_range = b[0] - a[0], b[1] - a[1]
        print(f"  {cid:<24} povprecje {a[0]:.3f} -> {b[0]:.3f} ({d_mean:+.3f}),"
              f"  razpon {a[1]:.3f} -> {b[1]:.3f} ({d_range:+.3f})")
    print("\nOpomba: invariantnost iz Ng et al. (1999) velja za optimalno politiko MDP, ne za resitev,")
    print("ki jo najde polgradientna metoda z aproksimacijo. Razlika v naucenih politikah je pricakovana.")


if __name__ == "__main__":
    main()
