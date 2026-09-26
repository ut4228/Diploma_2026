"""Ponovna kalibracija meja za tile coding iz NAUCENIH politik (korak po pilotu).

Zakaj: meje v plan_v1.json in p2_control.json so izmerjene z diagnosticnimi hevristikami
(nakljucna politika, crpanje). Naucena politika lahko doseze druge hitrosti, zato lahko
delez stanj zunaj meja preseze kriterij 5 (< 1 %). Ta skripta izmeri dejanske razpone pod
POZRESNO naucene politiko in meje po potrebi razsiri.

Pravilo (f): nova meja = max(stara meja, max |opazovano pod naucено politiko| * (1 + rezerva)).
Meje se samo SIRIJO, nikoli ozijo, da razsiritev ne more poslabsati ze doseze ne pokritosti.
Kalibracija se izvede ENKRAT po pilotu; nato se meje zamrznejo.

Uporaba:
  python scripts/recalibrate_bounds.py --task P2ii_flat_balance
  python scripts/recalibrate_bounds.py --task P2ii_flat_balance --dry-run
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda
from hillcart.env import EnvOutcome, ExperimentSpec, HillCartEnv
from hillcart.initial_states import eval_states
from hillcart.tile_coding import coder_from_bounds
from hillcart.tracking import git_info
from hillcart.tracks import FlatTrack, PowerValley

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "plan_v1.json"
P2CFG = ROOT / "configs" / "p2_control.json"
KEYS = ("x", "x_dot", "theta", "theta_dot")


def build_env(task, spec, bounds):
    t = task["track"]
    track = FlatTrack(half_length=t["half_length"]) if t["type"] == "FlatTrack" else \
        PowerValley(H=t["H"], tan_phi_max=t["tan_phi_max"], p=t["p"])
    return HillCartEnv(spec, check_pole=task["check_pole"], track=track, F_max=task["F_max"],
                       tile_bounds=bounds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--runs", default=None, help="glob za mape zagonov (privzeto runs/*p2_<task>_s*)")
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(P2CFG.read_text(encoding="utf-8"))
    task = cfg["tasks"][args.task]
    bounds = task["tile_bounds"]
    spec = ExperimentSpec.from_plan(PLAN, cfg["tasks"]["P2i_pump_no_pole"]["track"]["config_id"])
    coder = coder_from_bounds(bounds, n_tilings=cfg["agent"]["n_tilings"],
                              n_intervals=cfg["agent"]["n_intervals"])

    pattern = args.runs or str(ROOT / "runs" / f"*p2_{args.task}_s*")
    dirs = [d for d in sorted(glob.glob(pattern)) if Path(d, "weights_final.npz").exists()]
    if not dirs:
        raise SystemExit(f"ni zagonov z utezmi: {pattern}")

    lo = np.full(4, np.inf)
    hi = np.full(4, -np.inf)
    n_ep = 0
    for d in dirs:
        w = np.load(Path(d) / "weights_final.npz")["w"]
        agent = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=0.0), seed=0)
        agent.w = w
        env = build_env(task, spec, bounds)
        for s0 in eval_states():
            obs = env.reset(s0)
            while env.outcome is EnvOutcome.RUNNING:
                lo, hi = np.minimum(lo, obs), np.maximum(hi, obs)
                obs, *_ = env.step(agent.act(obs, greedy=True))
            lo, hi = np.minimum(lo, obs), np.maximum(hi, obs)
            n_ep += 1

    new_bounds = dict(bounds)
    changed = {}
    for i, k in enumerate(KEYS):
        if k in ("x", "theta"):
            continue  # doloceni s pravili epizode, se ne sirita
        need = max(abs(lo[i]), abs(hi[i])) * (1 + args.margin)
        old = bounds[k][1]
        if need > old:
            new_bounds[k] = [-need, need]
            changed[k] = {"old": old, "new": need, "observed_absmax": max(abs(lo[i]), abs(hi[i]))}

    print(f"zagonov: {len(dirs)}, epizod: {n_ep}")
    for i, k in enumerate(KEYS):
        mark = "  <- razsirjeno" if k in changed else ""
        print(f"  {k:<10} opazovano [{lo[i]:+8.3f}, {hi[i]:+8.3f}]   meja "
              f"[{new_bounds[k][0]:+8.3f}, {new_bounds[k][1]:+8.3f}]{mark}")
    if not changed:
        print("nobene meje ni treba razsiriti.")
        return
    if args.dry_run:
        print("--dry-run: konfiguracija ni spremenjena.")
        return

    task["tile_bounds_previous"] = bounds
    task["tile_bounds"] = new_bounds
    task["recalibration"] = {"script": "scripts/recalibrate_bounds.py", "git": git_info(ROOT),
                             "runs": [Path(d).name for d in dirs], "margin": args.margin,
                             "changed": changed, "episodes": n_ep,
                             "rule": "nova meja = max(stara, |opazovano pod pozresno politiko| * (1 + rezerva))"}
    cfg["version"] = "p2_control_v2"
    P2CFG.write_text(json.dumps(cfg, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"posodobljeno: {P2CFG}  (verzija {cfg['version']})")
    print("ponovno pozeni nalogo z novimi mejami.")


if __name__ == "__main__":
    main()
