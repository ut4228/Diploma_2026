"""Naknadna analiza ze opravljenih zagonov - NE pozene nobenega ucenja.

Bere runs/<mapa>/{config.json, summary.json, evals.json} in loci dva razlicna izida:

  KONCNI  delez uspeha na koncu ucenja (to je politika, ki bi jo dejansko uporabili),
  NAJBOLJSI  najvisji delez uspeha na katerem koli vmesnem vrednotenju.

Ce je NAJBOLJSI mnogo visji od KONCNEGA, agent je resitev nasel in jo nato izgubil.
To je drugacna napaka kot "se ni nikoli naucil" in jo je treba v diplomi poroCati loceno.

Uporaba:
  python scripts/analyze_runs.py --pattern "runs/*p3b_*"
  python scripts/analyze_runs.py --pattern "runs/*p3_*" --csv analiza_p3.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import statistics
from pathlib import Path


def load(run_dir: Path) -> dict | None:
    try:
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        summ = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        evals = json.loads((run_dir / "evals.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        return None
    if not evals:
        return None
    rates = [e["success_rate"] for e in evals]
    best_i = max(range(len(rates)), key=lambda i: rates[i])
    return {
        "run": run_dir.name,
        "phase": cfg.get("phase", "?"),
        "config_id": cfg.get("config_id", summ.get("config_id", "?")),
        "reward": cfg.get("reward", "plain"),
        "seed": summ.get("seed"),
        "total_steps": cfg.get("protocol", {}).get("total_steps"),
        "final": summ.get("final_success_rate"),
        "best": rates[best_i],
        "best_step": evals[best_i].get("step"),
        "first_success_step": summ.get("first_success_step"),
        "t_event_median": summ.get("t_event_median"),
        "curve": "".join("#" if r >= 0.8 else ("+" if r > 0 else ".") for r in rates),
        "n_evals": len(rates),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/*p3b_*")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = [r for d in sorted(glob.glob(args.pattern)) if (r := load(Path(d))) is not None]
    if not rows:
        raise SystemExit(f"ni zagonov z evals.json: {args.pattern}")
    rows.sort(key=lambda r: (r["config_id"], r["reward"], r["seed"] if r["seed"] is not None else -1))

    print(f"{'konfiguracija':<24}{'nagrada':>8}{'seme':>6}{'koncni':>8}{'najboljsi':>11}"
          f"{'pri koraku':>12}{'1.uspeh':>10}  krivulja (. = 0, + = delno, # = >= 0.8)")
    for r in rows:
        fs = str(r["first_success_step"]) if r["first_success_step"] is not None else "-"
        print(f"{r['config_id']:<24}{r['reward']:>8}{str(r['seed']):>6}{r['final']:>8.3f}{r['best']:>11.3f}"
              f"{r['best_step']:>12}{fs:>10}  {r['curve']}")

    print("\n" + "-" * 100)
    print(f"{'konfiguracija':<24}{'nagrada':>8}{'koncni povp.':>14}{'najboljsi povp.':>17}"
          f"{'izgubljenih':>13}{'1.uspeh med.':>14}")
    groups = {}
    for r in rows:
        groups.setdefault((r["config_id"], r["reward"]), []).append(r)
    for (cid, rew), g in sorted(groups.items()):
        fin = [x["final"] for x in g]
        best = [x["best"] for x in g]
        lost = sum(1 for x in g if x["best"] - x["final"] > 0.2)
        fss = [x["first_success_step"] for x in g if x["first_success_step"] is not None]
        fsm = f"{statistics.median(fss):.0f}" if fss else "-"
        print(f"{cid:<24}{rew:>8}{statistics.mean(fin):>14.3f}{statistics.mean(best):>17.3f}"
              f"{lost:>8}/{len(g):<4}{fsm:>14}")

    print("\n'izgubljenih' = zagoni, kjer je najboljsi delez uspeha vsaj 0.2 visji od koncnega\n"
          "(agent je resitev nasel in jo nato izgubil - to ni isto kot 'se ni naucil').")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"zapisano: {args.csv}")


if __name__ == "__main__":
    main()
