"""Zažene validacijske teste T1-T12 in shrani poročilo v runs/ (z git hashem).

Dodatno izmeri podatke za diplomo:
  - drift energije (F = 0) v odvisnosti od podkoraka za RK4 in Euler (tabela, graf, opaženi red),
  - napako bilance dela (T5) in negativno kontrolo s tangentno silo.

Uporaba:  python scripts/run_validation.py [--allow-dirty]
"""
from __future__ import annotations

import argparse
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))

from hillcart.tracking import Run  # noqa: E402
from test_physics import energy_drift, work_balance_error  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-dirty", action="store_true", help="dovoli necommitano kodo (samo za razvoj)")
    args = ap.parse_args()

    steps = [0.02, 0.01, 0.005, 0.0025, 0.00125]
    cfg = {"purpose": "physics validation T1-T12", "energy_test": {"T_s": 2.0, "substeps_h": steps}}
    run = Run("validation", cfg, base_dir=ROOT / "runs", require_clean=not args.allow_dirty, repo_dir=ROOT)

    res = subprocess.run([sys.executable, "-m", "pytest", "-q", f"--junitxml={run.dir / 'pytest.xml'}"],
                         cwd=ROOT, capture_output=True, text=True)
    (run.dir / "pytest.txt").write_text(res.stdout + res.stderr, encoding="utf-8")

    table = []
    for i, h in enumerate(steps):
        rk, eu = energy_drift("rk4", h), energy_drift("euler", h)
        table.append({"h": h, "rk4_rel_drift": rk, "euler_rel_drift": eu})
        run.scalar("validation/energy_drift_rk4", rk, i)
        run.scalar("validation/energy_drift_euler", eu, i)
    orders = [math.log2(a["rk4_rel_drift"] / b["rk4_rel_drift"]) for a, b in zip(table, table[1:])]
    wb = {"horizontal": work_balance_error("horizontal"),
          "tangential_negative_control": work_balance_error("tangential")}

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.loglog(steps, [r["rk4_rel_drift"] for r in table], "o-", label="RK4")
    ax.loglog(steps, [r["euler_rel_drift"] for r in table], "s-", label="Euler")
    ax.set_xlabel("podkorak integracije h [s]")
    ax.set_ylabel("max |E - E0| / (M g H)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(run.dir / "energy_drift.png", dpi=150)

    last = res.stdout.strip().splitlines()[-1] if res.stdout.strip() else ""
    run.close({"pytest_returncode": res.returncode, "pytest_summary": last, "energy_drift": table,
               "rk4_observed_order": orders, "work_balance": wb})
    print(last)
    for r in table:
        print(f"h={r['h']:.5f}  RK4={r['rk4_rel_drift']:.2e}  Euler={r['euler_rel_drift']:.2e}")
    print("opaženi red RK4:", [round(o, 2) for o in orders])
    print("bilanca dela:", {k: f"{v:.2e}" for k, v in wb.items()})
    print("poročilo:", run.dir)
    sys.exit(res.returncode)


if __name__ == "__main__":
    main()
