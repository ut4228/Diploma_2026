"""Naknadna analiza ze opravljenih zagonov - ne pozene nobenega ucenja.

Bere runs/<mapa>/{config.json, summary.json, evals.json} in loci dva razlicna izida:

  KONCNI     delez uspeha na koncu ucenja,
  POZNI      povprecje zadnjih K vrednotenj (privzeto 5) - stabilnejsa mera pozne uspesnosti,
  NAJBOLJSI  najvisji delez uspeha na katerem koli vmesnem vrednotenju,
  NIHANJ     kolikokrat v drugi polovici ucenja krivulja preckа mejo 0.5.

Ce je najbolsi mnogo visji od koncnega, je agent resitev nasel in jo nato izgubil; ce je poleg
tega nihanj veliko, politika sploh ne konvergira in koncni delez uspeha meri le, kje se je ucenje
slucajno ustavilo. Takrat je POZNI ustreznejsa primarna metrika.

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


LATE_K = 5  # stevilo zadnjih vrednotenj za "pozno" povprecje


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
    k = min(LATE_K, len(rates))
    late = sum(rates[-k:]) / k
    half = rates[len(rates) // 2:]
    osc = sum(1 for a, b in zip(half, half[1:]) if (a >= 0.5) != (b >= 0.5))
    return {
        "run": run_dir.name,
        "phase": cfg.get("phase", "?"),
        "config_id": cfg.get("config_id", summ.get("config_id", "?")),
        "reward": cfg.get("reward", "plain"),
        "anneal": bool(cfg.get("hparams", {}).get("anneal", cfg.get("agent", {}).get("anneal", False))),
        "seed": summ.get("seed"),
        "total_steps": cfg.get("protocol", {}).get("total_steps"),
        "final": summ.get("final_success_rate"),
        "late": late,
        "late_k": k,
        "oscillations": osc,
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
    rows.sort(key=lambda r: (r["config_id"], r["reward"], r["anneal"],
                             r["seed"] if r["seed"] is not None else -1))

    print(f"{'konfiguracija':<24}{'nagrada':>8}{'ohlaj.':>7}{'seme':>6}{'koncni':>8}{'pozni':>8}{'najb.':>7}"
          f"{'nihanj':>8}{'1.uspeh':>10}  krivulja (. = 0, + = delno, # = >= 0.8)")
    for r in rows:
        fs = str(r["first_success_step"]) if r["first_success_step"] is not None else "-"
        print(f"{r['config_id']:<24}{r['reward']:>8}{('da' if r['anneal'] else 'ne'):>7}{str(r['seed']):>6}"
              f"{r['final']:>8.3f}{r['late']:>8.3f}{r['best']:>7.3f}{r['oscillations']:>8}{fs:>10}  {r['curve']}")

    print("\n" + "-" * 100)
    print(f"{'konfiguracija':<24}{'nagrada':>8}{'ohlaj.':>7}{'n':>4}{'koncni povp.':>14}{'pozni povp.':>13}"
          f"{'najb. povp.':>13}{'izgubljenih':>13}{'nihanj med.':>13}{'1.uspeh med.':>14}")
    groups = {}
    for r in rows:
        groups.setdefault((r["config_id"], r["reward"], r["anneal"]), []).append(r)
    for (cid, rew, ann), g in sorted(groups.items()):
        fin = [x["final"] for x in g]
        late = [x["late"] for x in g]
        best = [x["best"] for x in g]
        lost = sum(1 for x in g if x["best"] - x["final"] > 0.2)
        fss = [x["first_success_step"] for x in g if x["first_success_step"] is not None]
        fsm = f"{statistics.median(fss):.0f}" if fss else "-"
        osc = statistics.median([x["oscillations"] for x in g])
        print(f"{cid:<24}{rew:>8}{('da' if ann else 'ne'):>7}{len(g):>4}{statistics.mean(fin):>14.3f}"
              f"{statistics.mean(late):>13.3f}{statistics.mean(best):>13.3f}{lost:>8}/{len(g):<4}"
              f"{osc:>13.1f}{fsm:>14}")

    print(f"\n'izgubljenih' = zagoni, kjer je najboljsi delez uspeha vsaj 0.2 visji od koncnega.\n"
          f"'pozni' = povprecje zadnjih {LATE_K} vrednotenj; 'nihanj' = prehodi cez 0.5 v drugi polovici ucenja.\n"
          "Veliko nihanj pomeni, da politika ne konvergira in je 'koncni' nezanesljiva metrika.")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print(f"zapisano: {args.csv}")


if __name__ == "__main__":
    main()
