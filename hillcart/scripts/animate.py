"""Animacija ene epizode (GIF). Za vizualno preverjanje simulatorja (in kasneje naučenih politik).

Primer:  python scripts/animate.py --policy pumping --k 0.6 --no-pole-check --out pumping.gif
Politike 'zero', 'random', 'pumping' so diagnostične, ne metode diplome.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from hillcart.animation import animate, rollout
from hillcart.diagnostic_policies import pumping_policy, random_policy
from hillcart.dynamics import PoleCartParams, f_max_from_k, one_way_force
from hillcart.simulator import HillCartSimulator, SimConfig
from hillcart.tracks import PowerValley


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=float, default=1.0)
    ap.add_argument("--tan", type=float, default=0.5)
    ap.add_argument("--p", type=float, default=2.0)
    ap.add_argument("--k", type=float, default=None, help="F_max / F_krit")
    ap.add_argument("--kappa", type=float, default=None, help="F_max / F_E (privzeto 0.6)")
    ap.add_argument("--policy", choices=["zero", "random", "pumping"], default="pumping")
    ap.add_argument("--t-max", type=float, default=10.0)
    ap.add_argument("--no-pole-check", action="store_true", help="izklopi pogoj |theta| <= 15 (diagnostika)")
    ap.add_argument("--x0", type=float, nargs=4, default=[0.0, 0.0, 0.0, 0.0], metavar=("X", "XD", "TH", "THD"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="episode.gif")
    args = ap.parse_args()

    prm = PoleCartParams()
    v = PowerValley(H=args.H, tan_phi_max=args.tan, p=args.p)
    F = f_max_from_k(args.k, v, prm) if args.k is not None else (args.kappa or 0.6) * one_way_force(v, prm)
    sim = HillCartSimulator(v, F_max=F, params=prm, cfg=SimConfig(t_max=args.t_max),
                            check_pole=not args.no_pole_check)
    pol = {"zero": lambda o: 0.0, "random": random_policy(np.random.default_rng(args.seed), F),
           "pumping": pumping_policy(F)}[args.policy]
    frames, outcome = rollout(sim, pol, tuple(args.x0))
    animate(sim, frames, outcome, args.out)
    print(f"{outcome.value} pri t = {sim.t_event:.2f} s -> {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
