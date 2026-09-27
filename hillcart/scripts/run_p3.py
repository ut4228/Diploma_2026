"""Faza P3: pilot na problemu (vozicek s palico v kotanji).

Za razliko od P2 to niso kontrolne naloge: tu je palica omejena z |theta| <= 15 stopinj in
voziCek mora priti iz kotanje. Ali je problem sploh resljiv, ni znano - to je raziskovalno
vprasanje diplome. Zato uspeh v P3 ni kriterij; P3 preverja, da se ucenje obnasa zdravo
(brez divergence, meje zadoscajo, T_max zadosca) pred razsiritvijo na vseh 35 konfiguracij.

Zamrznjeni protokol (rezultat faze P2, glej configs/p2_control.json in oznako p2-v1):
  alpha0 = 0.5, optimisticna inicializacija w = 1/16 (q0 = +1), lambda = 0.9,
  gamma = 0.999, epsilon = 0.05, 16 plositev x 8 intervalov, zamiki (1,3,5,7).
Osnovna nastavitev (alpha0 = 0.1, w = 0) je v P2 dala 0.000 uspeha pri obeh kontrolnih
nalogah, zato sta oba rezervna koraka del standardne konfiguracije.

Uporaba:
  python scripts/run_p3.py                       # kappa 0.9 in 0.3, semena 100-102
  python scripts/run_p3.py --configs H1_tan0.5_p2_kap0.9 --seeds 100
  python scripts/run_p3.py --total-steps 200000  # krajsi preizkus
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
from hillcart.training import train

ROOT = Path(__file__).resolve().parents[1]

# --- zamrznjen protokol faze P3 -----------
DEFAULT_PLAN = ROOT / "configs" / "plan_v2.json"
CONFIG_IDS = ("H1_tan0.5_p2_kap0.9", "H1_tan0.5_p2_kap0.3")
SEEDS = (100, 101, 102)
TOTAL_STEPS = 1_000_000
EVAL_EVERY = 50_000
HPARAMS = {"alpha0": 0.5, "lambda": 0.9, "gamma": 0.999, "epsilon": 0.05,
           "optimistic_init": True, "eps_z": 1e-6, "n_tilings": 16, "n_intervals": 8,
           "source": "zamrznjeno po fazi P2 (oznaka p2-v1)"}
QUICK_EVAL_STATES = [(0.0, 0.0, th, td) for th in (-0.05, 0.0, 0.05) for td in (-0.05, 0.0, 0.05)]

# --- vnaprej zapisani kriteriji ------------------------------------
MAX_Q = 2.0            # kriterij 4
MAX_OOB = 0.01         # kriterij 5
TMAX_WARN_FRAC = 0.75  # kriterij 6a: MEDIANA casa uspesnih epizod
TIMEOUT_REACH_MAX = 0.90  # kriterij 6b: timeouti ne smejo biti "skoraj na cilju"
BASELINE_FALL_S = 2.3  # najdaljsi cas do padca kontrolnih politik (zagon H0)


def q_absmax_visited(env, agent, states) -> float:
    """max |Q| na stanjih, ki jih pozresna politika dejansko obisce. To je merodajna
    meritev za kriterij 4: linearna aproksimacija je nevezana tam, kamor agent ne gre,
    zato enakomerno vzorcenje po celotni skatli meja precenjuje |Q|."""
    from hillcart.env import EnvOutcome
    m = 0.0
    for s0 in states:
        obs = env.reset(s0)
        while env.outcome is EnvOutcome.RUNNING:
            m = max(m, float(np.abs(agent.q_values(obs)).max()))
            obs, *_ = env.step(agent.act(obs, greedy=True))
        m = max(m, float(np.abs(agent.q_values(obs)).max()))
    return m


def q_absmax_sampled(agent, coder, seed: int, n: int = 5000) -> float:
    """Diagnostika: max |Q| na enakomerno nakljucnih stanjih iz skatle meja (vkljucno z
    nikoli obiskanimi). Se poroca, a NI kriterij."""
    rng = np.random.default_rng(seed)
    m = 0.0
    for _ in range(n):
        obs = tuple(rng.uniform(lo, hi) for lo, hi in zip(coder.lows, coder.highs))
        m = max(m, float(np.abs(agent.w[coder.features_all_actions(obs)].sum(axis=1)).max()))
    return m


def run_one(spec: ExperimentSpec, seed: int, args) -> dict:
    bounds = {"x": [-spec.valley.W, spec.valley.W],
              "theta": [-spec.sim.theta_max, spec.sim.theta_max],
              "x_dot": spec.tile_bounds["x_dot"], "theta_dot": spec.tile_bounds["theta_dot"]}
    coder = coder_from_bounds(bounds, n_tilings=HPARAMS["n_tilings"], n_intervals=HPARAMS["n_intervals"])
    w_init = 1.0 / coder.cfg.n_tilings if HPARAMS["optimistic_init"] else 0.0
    acfg = AgentConfig(alpha0=args.alpha0, lam=HPARAMS["lambda"], gamma=HPARAMS["gamma"],
                       epsilon=HPARAMS["epsilon"], w_init=w_init, eps_z=HPARAMS["eps_z"])
    agent = TrueOnlineSarsaLambda(coder, acfg, seed=seed)
    env = HillCartEnv(spec, check_pole=True, tile_bounds=bounds)  # PRAVI problem: palica omejena

    cfg = {"phase": "P3", "config_id": spec.config_id, "seed": seed,
           "plan": {"path": str(Path(args.plan).name), "sha256": spec.plan_sha256,
                    "version": spec.plan_version},
           "hparams": {**HPARAMS, "alpha0": args.alpha0}, "tile_bounds": bounds,
           "env": env.describe(), "agent": agent.describe(),
           "protocol": {"total_steps": args.total_steps, "eval_every": args.eval_every,
                        "quick_eval_states": len(QUICK_EVAL_STATES),
                        "final_eval_states": len(eval_states()),
                        "note": "uspeh NI kriterij faze P3"},
           "criteria": {"q_absmax_max": MAX_Q, "oob_fraction_max": MAX_OOB,
                        "t_event_warn": TMAX_WARN_FRAC * spec.sim.t_max,
                        "baseline_fall_s": BASELINE_FALL_S}}
    run = Run(f"p3_{spec.config_id}_s{seed}", cfg, base_dir=ROOT / "runs", seed=seed,
              require_clean=not args.allow_dirty, repo_dir=ROOT)

    def progress(step, ev):
        print(f"    {step:>8d} korakov  uspeh={ev['success_rate']:.2f}  "
              f"padci={ev['outcomes'].get('fail_pole', 0):>2d}  timeout={ev['outcomes'].get('timeout', 0):>2d}  "
              f"|w|max={np.abs(agent.w).max():.3f}", flush=True)

    res = train(env, agent, total_steps=args.total_steps, eval_every=args.eval_every,
                eval_states_quick=QUICK_EVAL_STATES, eval_states_final=eval_states(),
                rng=np.random.default_rng(seed), run=run, progress=progress)

    records = res.pop("records")
    with open(run.dir / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        rows = [r.as_row() for r in records]
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    run.save_json("evals.json", res["evals"])
    np.savez_compressed(run.dir / "weights_final.npz", w=agent.w)

    fin = res["final_eval"]
    q_vis = q_absmax_visited(env, agent, eval_states())
    q = q_absmax_sampled(agent, coder, seed)
    t_max_seen = fin["t_event_max"]
    checks = {
        "C4_q_bounded": bool(q_vis <= MAX_Q and np.isfinite(agent.w).all()),
        "C5_oob": bool(res["train_oob_fraction"] < MAX_OOB),
        "C6a_t_median_ok": bool(fin["t_event_median"] is None
                                or fin["t_event_median"] <= TMAX_WARN_FRAC * spec.sim.t_max),
        "C6b_timeouts_not_truncated": bool(fin["timeout_reach_median"] is None
                                           or fin["timeout_reach_median"] < TIMEOUT_REACH_MAX),
        "C_fall_beats_baseline": bool(fin["t_event_fail_median"] is None
                                      or fin["t_event_fail_median"] > BASELINE_FALL_S),
    }
    summary = {"config_id": spec.config_id, "seed": seed, "kappa": spec.forces["kappa"],
               "F_max": spec.F_max, "t_max_s": spec.sim.t_max,
               "final_success_rate": fin["success_rate"], "final_outcomes": fin["outcomes"],
               "exit_sides": fin["exit_sides"], "t_event_median": fin["t_event_median"],
               "t_event_max": t_max_seen, "steps_median": fin["steps_median"],
               "t_event_fail_median": fin["t_event_fail_median"],
               "timeout_reach_median": fin["timeout_reach_median"],
               "timeout_reach_max": fin["timeout_reach_max"],
               "episodes": res["episodes"], "env_steps": res["steps"],
               "first_success_step": res["first_success_step"], "train_outcomes": res["train_outcomes"],
               "q_absmax_visited": q_vis, "q_absmax_sampled": q, "w_absmax": res["w_absmax"],
               "train_oob_fraction": res["train_oob_fraction"], "eval_oob_steps": fin["oob_steps_total"],
               "max_active_traces": res["max_active_traces"], "checks": checks}
    run.close(summary)
    summary["run_dir"] = run.dir.name
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=str(DEFAULT_PLAN))
    ap.add_argument("--configs", nargs="+", default=list(CONFIG_IDS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--total-steps", type=int, default=TOTAL_STEPS)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--alpha0", type=float, default=HPARAMS["alpha0"])
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    results = []
    for cid in args.configs:
        spec = ExperimentSpec.from_plan(args.plan, cid)
        print(f"\n=== {cid} ===  kappa={spec.forces['kappa']}, F_max={spec.F_max:.3f} N, "
              f"k={spec.forces['k']:.2f}, u={spec.forces['u']:.3f}, W={spec.valley.W:.1f} m, "
              f"T_max={spec.sim.t_max:g} s")
        for seed in args.seeds:
            print(f"  seme {seed}:")
            results.append(run_one(spec, seed, args))

    print("\n" + "=" * 96)
    print(f"{'konfiguracija':<24}{'seme':>5}{'uspeh':>7}{'L/D':>8}{'t_med':>7}{'t_max':>7}"
          f"{'padci':>7}{'t_pad':>7}{'|Q|obi':>7}{'dosegTO':>9}{'oob':>8}")
    for r in results:
        o = r["final_outcomes"]
        sides = f"{r['exit_sides']['left']}/{r['exit_sides']['right']}"
        tm = f"{r['t_event_median']:.1f}" if r["t_event_median"] is not None else "-"
        tx = f"{r['t_event_max']:.1f}" if r["t_event_max"] is not None else "-"
        tp = f"{r['t_event_fail_median']:.1f}" if r["t_event_fail_median"] is not None else "-"
        print(f"{r['config_id']:<24}{r['seed']:>5}{r['final_success_rate']:>7.3f}{sides:>8}{tm:>7}{tx:>7}"
              f"{o.get('fail_pole', 0):>7}{tp:>7}{r['q_absmax_visited']:>7.2f}"
              f"{(f'{rr:.2f}' if (rr := r['timeout_reach_median']) is not None else '-'):>9}"
              f"{r['train_oob_fraction']:>8.4f}")

    print("\nkriteriji faze P3 (uspeh ni med njimi):")
    names = {"C4_q_bounded": f"|Q| na OBISKANIH stanjih <= {MAX_Q} in koncne utezi",
             "C5_oob": f"delez korakov zunaj meja < {MAX_OOB}",
             "C6a_t_median_ok": f"MEDIANA casa uspesnih epizod pred {TMAX_WARN_FRAC:.0%} T_max",
             "C6b_timeouts_not_truncated": f"timeouti ne dosezejo {TIMEOUT_REACH_MAX:.0%} poti do cilja "
                                           "(sicer T_max res reze resitve)",
             "C_fall_beats_baseline": f"mediana casa do padca > {BASELINE_FALL_S} s (kontrolne politike)"}
    for key, label in names.items():
        bad = [f"{r['config_id'].split('_')[-1]}/s{r['seed']}" for r in results if not r["checks"][key]]
        print(f"  {'OK ' if not bad else 'NE '} {label}" + (f"   krsitve: {', '.join(bad)}" if bad else ""))

    timeouts = sum(r["final_outcomes"].get("timeout", 0) for r in results)
    print(f"\ntimeouti pri vrednotenju: {timeouts}/{len(results) * len(eval_states())}")
    print("Naslednji korak dolocijo krsitve zgoraj; ce jih ni, gremo na fazo A (vseh 35 konfiguracij).")


if __name__ == "__main__":
    main()
