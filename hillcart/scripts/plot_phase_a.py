"""Glavni graf faze A: delez uspeha in strosek odkritja v odvisnosti od kappa.

Bere runs/phase_a_summary.csv (ali pot podano z --csv) in narise dvodelno sliko:
  (a) delez uspeha na 81 fiksnih zacetnih stanjih - vsako seme posebej + mediana,
  (b) korak prvega uspeha (logaritemska os) - strosek odkritja.

Posamezna semena so narisana kot tocke, ker je porazdelitev pri nizkih kappa
DVOVRHNA (agent bodisi resi bodisi ostane v lokalnem optimumu). Stolpci s
standardnim odklonom bi to skrili.

Uporaba:  python scripts/plot_phase_a.py [--csv pot] [--out slika.png]
"""
from __future__ import annotations

import argparse
import ast
import csv
import statistics as st
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Barvni zetoni (referencna paleta, svetli nacin - diploma se tiska na belo)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES = "#2a78d6"

FEASIBLE = 0.5


def load(paths: list[Path]) -> tuple[dict, dict]:
    """Zdruzi vec povzetkov. Ce se ista konfiguracija pojavi v vec datotekah,
    obvelja ZADNJA podana (ponovitev povozi prvotni zagon). Vrne tudi, od kod je kaj."""
    by, src = {}, {}
    for p in paths:
        seen = {}
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if not r.get("config_id"):
                continue
            seen.setdefault(float(r["kappa"]), []).append(r)
        for k, rows in seen.items():
            if k in by:
                print(f"  kappa {k:g}: {len(by[k])} vrstic iz {src[k]} -> zamenjano z "
                      f"{len(rows)} iz {p.name}")
            by[k] = rows
            src[k] = p.name
    return dict(sorted(by.items())), src


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
        ax.spines[s].set_linewidth(1.0)
    ax.tick_params(colors=INK_2, labelsize=9, length=3, width=1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+",
                    default=[str(ROOT / "runs" / "phase_a_summary.csv"),
                             str(ROOT / "runs" / "phase_a_v2_summary.csv")],
                    help="en ali vec povzetkov; pri podvojeni konfiguraciji obvelja ZADNJI")
    ap.add_argument("--out", default=str(ROOT / "runs" / "phase_a.png"))
    ap.add_argument("--dpi", type=int, default=200)
    args = ap.parse_args()

    paths = [p for c in args.csv if (p := Path(c)).exists()]
    if not paths:
        raise SystemExit(f"ni nobenega povzetka: {args.csv}")
    print("berem:", ", ".join(p.name for p in paths))
    by, src = load(paths)
    ks = list(by)
    rng = np.random.default_rng(0)  # determinističen razmik prekrivajocih se tock

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=SURFACE)

    # ---------------------------------------------------------------- (a) delez uspeha
    med = [st.median([float(r["final_success_rate"]) for r in by[k]]) for k in ks]
    ax1.axhline(FEASIBLE, color=AXIS, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax1.text(1.255, FEASIBLE + 0.02, "prag 0,5", color=INK_2, fontsize=8, ha="right")
    ax1.plot(ks, med, "-", color=SERIES, lw=2, zorder=3)
    for k in ks:
        v = [float(r["final_success_rate"]) for r in by[k]]
        jitter = rng.uniform(-0.018, 0.018, size=len(v))
        ax1.plot(np.array([k] * len(v)) + jitter, v, "o", ms=7, color=SERIES,
                 alpha=0.35, mew=0, zorder=2)
    ax1.plot(ks, med, "o", ms=9, color=SERIES, mec=SURFACE, mew=2, zorder=4)
    for k, m in zip(ks, med):
        ax1.annotate(f"{m:.2f}".replace(".", ","), (k, m), textcoords="offset points",
                     xytext=(0, 13), ha="center", fontsize=9, color=INK)
    ax1.set_ylim(-0.06, 1.14)
    ax1.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax1.set_yticklabels(["0", "0,25", "0,50", "0,75", "1"])
    ax1.set_title("(a) Delež uspeha po učenju", fontsize=11, color=INK, loc="left", pad=10)
    ax1.set_ylabel("delež uspešnih od 81 začetnih stanj", fontsize=9, color=INK_2)

    # ---------------------------------------------------------------- (b) strosek odkritja
    fs_med, fs_k = [], []
    for k in ks:
        v = [int(r["first_success_step"]) for r in by[k] if r["first_success_step"]]
        if not v:
            continue
        fs_k.append(k)
        fs_med.append(st.median(v))
        jitter = rng.uniform(-0.018, 0.018, size=len(v))
        ax2.plot(np.array([k] * len(v)) + jitter, v, "o", ms=7, color=SERIES,
                 alpha=0.35, mew=0, zorder=2)
    ax2.plot(fs_k, fs_med, "-", color=SERIES, lw=2, zorder=3)
    ax2.plot(fs_k, fs_med, "o", ms=9, color=SERIES, mec=SURFACE, mew=2, zorder=4)
    ax2.set_yscale("log")
    ax2.set_ylim(3.5e4, 6e6)
    ax2.set_yticks([5e4, 1e5, 3e5, 1e6, 3e6])
    ax2.set_yticklabels(["50 tisoč", "100 tisoč", "300 tisoč", "1 milijon", "3 milijoni"])
    ax2.set_title("(b) Korak prvega uspeha", fontsize=11, color=INK, loc="left", pad=10)
    ax2.set_ylabel("korakov okolja (log)", fontsize=9, color=INK_2)
    notes = [f"pri κ = {k:g} je sploh kdaj uspelo le {sum(1 for r in by[k] if r['first_success_step'])}"
             f" od {len(by[k])} semen".replace(".", ",")
             for k in ks if 0 < sum(1 for r in by[k] if r["first_success_step"]) < len(by[k])]
    if notes:
        ax2.text(0.02, 0.04, "\n".join(notes), transform=ax2.transAxes,
                 fontsize=8, color=INK_2, ha="left", va="bottom")

    for ax in (ax1, ax2):
        style(ax)
        ax.set_xlim(0.22, 1.28)
        ax.set_xticks(ks)
        ax.set_xticklabels([f"{k:g}".replace(".", ",") for k in ks])
        ax.set_xlabel(r"$\kappa = F_{max} / F_E$   (manjši = šibkejši motor)", fontsize=9, color=INK_2)

    fig.suptitle("Meja izvedljivosti: voziček s palico v kotanji, 10 semen na vrednost κ",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    foot = ("Osnovna kotanja (H = 1 m, tan φ_max = 0,5, p = 2) · true online Sarsa(λ) + tile coding · "
            "R-C + Φ_E · 3·10⁶ korakov · svetle točke = posamezna semena, polna črta = mediana")
    if len(set(src.values())) > 1:
        zadnji = [p.name for p in paths][-1]
        novejsi = [f"{k:g}".replace(".", ",") for k in ks if src[k] == zadnji]
        foot += (f"\nκ = {', '.join(novejsi)}: ponovitev s popravljenimi mejami tile codinga "
                 f"(plan_v3, oznaka phase-a-v2); ostale vrednosti iz phase-a-v1")
    fig.text(0.012, 0.015, foot, fontsize=8, color=INK_2, ha="left")
    fig.tight_layout(rect=(0, 0.045, 1, 0.945))
    out = Path(args.out)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print("shranjeno:", out)


if __name__ == "__main__":
    main()
