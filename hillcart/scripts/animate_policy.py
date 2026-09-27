"""Animacija naucene politike iz zapisanih utezi.

Bere weights_final.npz in config.json iz mape zagona, sestavi okolje z ISTIMI mejami in
konfiguracijo kot med ucenjem, ter posname eno epizodo s pozresno politiko (epsilon = 0).

Uporaba:
  python scripts/animate_policy.py --run runs/20260926-...._p3_H1_tan0.5_p2_kap0.9_s100
  python scripts/animate_policy.py --run <mapa> --state 0 0 0 0 --out politika.gif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda
from hillcart.animation import animate, rollout
from hillcart.env import ExperimentSpec, HillCartEnv
from hillcart.initial_states import eval_states
from hillcart.tile_coding import coder_from_bounds
from hillcart.tracks import FlatTrack, PowerValley

ROOT = Path(__file__).resolve().parents[1]


def build(run_dir: Path):
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    w = np.load(run_dir / "weights_final.npz")["w"]
    bounds = cfg.get("tile_bounds") or cfg["task_spec"]["tile_bounds"]
    ag_cfg = cfg["agent"]
    coder = coder_from_bounds(bounds, n_tilings=ag_cfg["tile_coding"]["n_tilings"],
                              n_intervals=ag_cfg["tile_coding"]["n_intervals"])
    if w.size != coder.n_features:
        raise SystemExit(f"utezi ({w.size}) se ne ujemajo s tile codingom ({coder.n_features})")
    agent = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=0.0), seed=0)
    agent.w = w

    plan_name = cfg.get("plan", {}).get("path", "plan_v2.json")
    env_desc = cfg["env"]
    tr = env_desc["track"]
    if "config_id" in cfg:  # P3
        spec = ExperimentSpec.from_plan(ROOT / "configs" / plan_name, cfg["config_id"])
        track = spec.valley
    else:  # P2
        spec = ExperimentSpec.from_plan(ROOT / "configs" / "plan_v1.json",
                                        cfg["task_spec"]["track"].get("config_id", "H1_tan0.5_p2_kap0.5"))
        track = FlatTrack(half_length=tr["half_length"]) if tr["type"] == "FlatTrack" else \
            PowerValley(H=tr["H"], tan_phi_max=tr["tan_phi_max"], p=tr["p"])
    env = HillCartEnv(spec, check_pole=env_desc["check_pole"], track=track,
                      F_max=env_desc["F_max"], tile_bounds=bounds)
    return cfg, env, agent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="mapa zagona v runs/")
    ap.add_argument("--state", type=float, nargs=4, default=None, metavar=("X", "XD", "TH", "THD"))
    ap.add_argument("--best", action="store_true", help="izberi zacetno stanje z najkrajso uspesno epizodo")
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=3)
    args = ap.parse_args()

    run_dir = Path(args.run)
    cfg, env, agent = build(run_dir)

    states = [tuple(args.state)] if args.state else eval_states()
    if args.best and not args.state:
        best = None
        for s0 in states:
            env.reset(s0)
            obs = env.obs
            while env.outcome.value == "running":
                obs, *_ = env.step(agent.act(obs, greedy=True))
            if env.outcome.value == "success" and (best is None or env.t_event < best[1]):
                best = (s0, env.t_event)
        if best is None:
            print("nobena epizoda ni uspesna; uporabljam (0,0,0,0)")
            states = [(0.0, 0.0, 0.0, 0.0)]
        else:
            states = [best[0]]
            print(f"najkrajsa uspesna epizoda: s0={best[0]}, t={best[1]:.2f} s")
    else:
        states = states[:1] if not args.state else states

    sim = env._sim
    policy = lambda obs: env.forces[agent.act(obs, greedy=True)]
    frames, outcome = rollout(sim, policy, states[0])
    name = args.out or f"{run_dir.name}.gif"
    label = cfg.get("config_id") or cfg.get("task", "")
    animate(sim, frames, outcome, name, fps=args.fps, stride=args.stride,
            title=f"{label}  (naucena politika, epsilon = 0)")
    print(f"{outcome.value} pri t = {sim.t_event:.2f} s -> {Path(name).resolve()}")


if __name__ == "__main__":
    main()
