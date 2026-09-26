"""Kontrolni zagon H0: naivne kontrolne politike A, B, C na fiksnih stanjih za vrednotenje.

To NISO metode diplome, ampak naivne spodnje meje.
  A  pi(o) = 0                          (F se ne uporablja -> samo 7 kotanj)
  B  pi(o) = +F_max
  C  pi(o) = F_max * sgn(x_dot), sgn(0) = +1

Vnaprej napovedan rezultat (H0): nobena politika ne doseže uspeha; edini timeouti pri A so
v stanju (0,0,0,0), ki je točka ravnovesja.

Uporaba:  python scripts/run_baselines.py [--allow-dirty]
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

from hillcart.env import EnvOutcome, ExperimentSpec, HillCartEnv
from hillcart.initial_states import eval_states
from hillcart.tracking import Run

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "plan_v1.json"

POLICIES = {
    "A_zero": lambda o: 1,
    "B_const": lambda o: 2,
    "C_pump": lambda o: 2 if o[1] >= 0.0 else 0,
}


def config_ids(plan_rows, policy: str) -> list[str]:
    if policy != "A_zero":
        return [r["id"] for r in plan_rows]
    seen, out = set(), []
    for r in plan_rows:  # F se pri A ne uporablja -> ena konfiguracija na kotanjo
        key = (r["valley"]["H"], r["valley"]["tan_phi_max"], r["valley"]["p"])
        if key not in seen:
            seen.add(key)
            out.append(r["id"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true")
    args = ap.parse_args()

    import json
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    states = eval_states()
    run = Run("baselines", {"policies": list(POLICIES), "plan_version": plan["version"],
                            "n_eval_states": len(states), "hypothesis_H0":
                                "0 % uspeha; timeouti pri A samo v ravnovesnem stanju (0,0,0,0)"},
              base_dir=ROOT / "runs", require_clean=not args.allow_dirty, repo_dir=ROOT)

    rows, summary = [], {}
    plan_sha = None
    for pname, policy in POLICIES.items():
        counts = collections.Counter()
        t_fail_max = 0.0
        for cid in config_ids(plan["configs"], pname):
            spec = ExperimentSpec.from_plan(PLAN, cid)
            plan_sha = spec.plan_sha256
            env = HillCartEnv(spec)
            for si, s0 in enumerate(states):
                obs = env.reset(s0)
                max_theta, total_r = abs(obs[2]), 0.0
                while env.outcome is EnvOutcome.RUNNING:
                    obs, r, term, trunc, info = env.step(policy(obs))
                    total_r += r
                    max_theta = max(max_theta, abs(obs[2]))
                counts[env.outcome.value] += 1
                if env.outcome is EnvOutcome.FAIL_POLE:
                    t_fail_max = max(t_fail_max, env.t_event)
                rows.append({"policy": pname, "config_id": cid, "state_idx": si, "x0": s0[0], "xdot0": s0[1],
                             "theta0": s0[2], "thetadot0": s0[3], "outcome": env.outcome.value,
                             "t_event": env.t_event, "steps": env.steps, "max_abs_theta": max_theta,
                             "x_final": obs[0], "exit_side": info["exit_side"], "reward_sum": total_r,
                             "oob_steps": env.oob_steps})
        n = sum(counts.values())
        summary[pname] = {"episodes": n, "outcomes": dict(counts),
                          "success_rate": counts["success"] / n, "max_t_fail_pole": t_fail_max,
                          "timeouts_only_in_equilibrium": all(
                              r["outcome"] != "timeout" or (r["x0"], r["xdot0"], r["theta0"], r["thetadot0"]) == (0, 0, 0, 0)
                              for r in rows if r["policy"] == pname)}
        run.scalar(f"baseline/success_rate/{pname}", summary[pname]["success_rate"], 0)
        print(pname, summary[pname]["outcomes"], f"uspeh={summary[pname]['success_rate']:.3f}",
              f"max t do padca={t_fail_max:.2f}s")

    with open(run.dir / "baselines.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    h0_ok = all(s["success_rate"] == 0.0 and s["timeouts_only_in_equilibrium"] for s in summary.values())
    run.close({"plan_sha256": plan_sha, "summary": summary, "H0_reproduced": h0_ok})
    print("H0 reproduciran:", h0_ok)
    print("zapis:", run.dir)


if __name__ == "__main__":
    main()
