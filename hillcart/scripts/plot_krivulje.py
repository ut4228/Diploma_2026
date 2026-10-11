"""Ucne krivulje: delez uspeha pri vmesnem vrednotenju v odvisnosti od korakov.

Namenjeno poglavju o rezultatih: pokaze, kaj se med ucenjem dogaja, ne le koncnega
stevila. Najbolj pove pri primerjavi nagrade z oblikovanjem in brez njega - krivulji
koncata enako visoko, oblikovana pa se vzpne prej.

Zagone zdruzi v skupine po tem, kar jih loci (nagrada, ohlajanje, konfiguracija),
in za vsako skupino narise mediano cez semena ter pas med najmanjso in najvecjo
vrednostjo. Posameznih semen NE risemo, ker bi bilo pri petih ali vec skupinah
necitljivo; za prikaz posameznih semen sluzi plot_phase_a.py.

Pri vec kot dveh skupinah se pasovi prekrivajo in slika postane neberljiva, zato
privzeto narisemo locene ploscice (eno na skupino, s sivimi medianami ostalih
skupin v ozadju). Z --skupaj to izsilimo na eno ploscico.

Krivulje med ucenjem nihajo med 0 in 1; to nihanje je ugotovitev in ga privzeto
ne glajamo. Z --glajenje N se poleg surove krivulje narise drseca mediana sirine
N vrednotenj; sirina se zapise v nogo, da jo je mogoce navesti v podnapisu.

Uporaba:
  python scripts/plot_krivulje.py --pattern "runs/*p3b_*"
  python scripts/plot_krivulje.py --pattern "runs/*phaseA_*" --by config_id
  python scripts/plot_krivulje.py --pattern "runs/*p3b_*" --out runs/krivulje.png
"""
from __future__ import annotations

import argparse
import glob
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# barvni zetoni (svetli nacin - diploma se tiska na belo)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# fiksen vrstni red barv; nikoli ne krozimo, pri vec kot 6 skupinah raje loci sliko
SERIES = ["#2a78d6", "#d1600f", "#3c8f5a", "#8a5cd6", "#b03a48", "#52514e"]

IME = {"plain": "R-C (brez oblikovanja)", "shaped": "R-C + $\\Phi_E$"}


def oznaka(cfg: dict, summ: dict, by: tuple) -> str:
    deli = []
    for k in by:
        if k == "reward":
            deli.append(IME.get(cfg.get("reward", "plain"), cfg.get("reward", "?")))
        elif k == "anneal":
            ann = cfg.get("hparams", {}).get("anneal", False)
            deli.append("z ohlajanjem" if ann else "brez ohlajanja")
        elif k == "config_id":
            kap = summ.get("kappa")
            deli.append(f"$\\kappa$ = {kap:g}".replace(".", ",") if kap is not None
                        else str(cfg.get("config_id", "?")))
        else:
            deli.append(str(cfg.get(k, "?")))
    return ", ".join(deli)


def load(pattern: str, by: tuple, izloci: tuple) -> tuple[dict, dict]:
    """Vrne skupine krivulj in za vsako skupino opis zajetih zagonov.

    Opis je namenjen preverjanju: ce se v isti skupini znajdejo zagoni z razlicno
    konfiguracijo, ohlajanjem ali T_max, primerjava NI enofaktorska in slika zavaja.
    """
    skupine: dict[str, list] = {}
    popis: dict[str, list] = {}
    for d in sorted(glob.glob(pattern)):
        p = Path(d)
        if any(x in p.name for x in izloci):
            continue
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
            summ = json.loads((p / "summary.json").read_text(encoding="utf-8"))
            evals = json.loads((p / "evals.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not evals:
            continue
        lab = oznaka(cfg, summ, by)
        skupine.setdefault(lab, []).append(([e["step"] for e in evals],
                                            [e["success_rate"] for e in evals]))
        popis.setdefault(lab, []).append({
            "zagon": p.name,
            "config_id": cfg.get("config_id", "?"),
            "anneal": bool(cfg.get("hparams", {}).get("anneal", False)),
            "t_max": cfg.get("env", {}).get("spec", {}).get("sim", {}).get("t_max"),
            "koraki": cfg.get("protocol", {}).get("total_steps"),
            "seme": summ.get("seed"),
        })
    return skupine, popis


def preveri(popis: dict) -> list:
    """Opozorila, ce skupine niso homogene ali niso enako velike."""
    opozorila = []
    for lab, zagoni in popis.items():
        for klj in ("config_id", "anneal", "t_max", "koraki"):
            vrednosti = {z[klj] for z in zagoni}
            if len(vrednosti) > 1:
                opozorila.append(f"skupina '{lab}' meša {klj}: {sorted(map(str, vrednosti))}")
    velikosti = {lab: len(z) for lab, z in popis.items()}
    if len(set(velikosti.values())) > 1:
        opozorila.append(f"skupine niso enako velike: {velikosti}")
    semena = {lab: {z['seme'] for z in zagoni} for lab, zagoni in popis.items()}
    if len(set(map(frozenset, semena.values()))) > 1:
        opozorila.append(f"skupine ne uporabljajo istih semen: "
                         f"{ {k: sorted(v) for k, v in semena.items()} }")
    return opozorila


def poravnaj(zagoni: list) -> tuple:
    """Vrne skupno os korakov in matriko delezev; krajse zagone odrezemo."""
    n = min(len(k) for k, _ in zagoni)
    x = np.array(zagoni[0][0][:n], dtype=float)
    y = np.array([d[:n] for _, d in zagoni], dtype=float)
    return x, y


def ovij(besedilo: str, sirina: int = 118) -> str:
    """Noga naj ne tece cez rob slike."""
    return "\n".join(textwrap.wrap(besedilo, sirina))


def drseca_mediana(v: np.ndarray, okno: int) -> np.ndarray:
    """Drseca mediana s sredinskim oknom; robove skrajsa, ne podaljsuje."""
    if okno < 2:
        return v
    out = np.empty_like(v, dtype=float)
    pol = okno // 2
    for i in range(len(v)):
        a, b = max(0, i - pol), min(len(v), i + pol + 1)
        out[i] = float(np.median(v[a:b]))
    return out


def olepsaj(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_ylim(-0.04, 1.08)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0,25", "0,50", "0,75", "1"])
    ax.grid(True, axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: "0" if v == 0 else f"{v/1e6:g} mio".replace(".", ",")))


def narisi_locene(skupine: dict, okno: int, naslov: str) -> "matplotlib.figure.Figure":
    """Ena ploscica na skupino; mediane ostalih skupin so sivo v ozadju."""
    labs = sorted(skupine)
    stolpcev = 2 if len(labs) > 1 else 1
    vrstic = -(-len(labs) // stolpcev)
    fig, osi = plt.subplots(vrstic, stolpcev, figsize=(9.6, 2.5 * vrstic + 0.9),
                            facecolor=SURFACE, sharex=True, sharey=True,
                            squeeze=False)
    podatki = {lab: poravnaj(skupine[lab]) for lab in labs}
    for i, lab in enumerate(labs):
        ax = osi[i // stolpcev][i % stolpcev]
        for drug in labs:
            if drug == lab:
                continue
            xo, yo = podatki[drug]
            ax.plot(xo, drseca_mediana(np.median(yo, axis=0), okno),
                    color=AXIS, lw=1.0, zorder=2)
        x, y = podatki[lab]
        med = np.median(y, axis=0)
        col = SERIES[i]
        ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=col, alpha=0.16, lw=0)
        if okno >= 2:
            ax.plot(x, med, color=col, lw=0.9, alpha=0.45, zorder=3)
        ax.plot(x, drseca_mediana(med, okno), color=col, lw=2.2, zorder=4)
        ax.set_title(f"{lab}  ($n$ = {len(skupine[lab])})", fontsize=9.5,
                     color=col, loc="left", pad=5)
        olepsaj(ax)
    for i in range(len(labs), vrstic * stolpcev):
        osi[i // stolpcev][i % stolpcev].set_visible(False)
    for j in range(stolpcev):
        osi[vrstic - 1][j].set_xlabel("korakov okolja", fontsize=9, color=INK_2)
    for i in range(vrstic):
        osi[i][0].set_ylabel("delež uspeha", fontsize=9, color=INK_2)
    fig.suptitle(naslov, fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/*p3b_*")
    ap.add_argument("--by", nargs="+", default=["reward"],
                    choices=["reward", "anneal", "config_id"],
                    help="kaj loci skupine (privzeto nagrada)")
    ap.add_argument("--izloci", nargs="*", default=["tmax"],
                    help="preskoci zagone, katerih ime vsebuje katerega od teh nizov "
                         "(privzeto 'tmax', ker so to diagnosticni zagoni)")
    ap.add_argument("--vseeno", action="store_true",
                    help="narisi tudi, ce skupine niso enofaktorske (sicer se ustavi)")
    ap.add_argument("--skupaj", action="store_true",
                    help="vse skupine na eno ploscico (privzeto locene pri vec kot dveh)")
    ap.add_argument("--glajenje", type=int, default=0, metavar="N",
                    help="drseca mediana sirine N vrednotenj; 0 pomeni brez glajenja")
    ap.add_argument("--out", default=str(ROOT / "runs" / "krivulje.png"))
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--naslov", default="Potek učenja")
    args = ap.parse_args()

    skupine, popis = load(args.pattern, tuple(args.by), tuple(args.izloci))
    if not skupine:
        raise SystemExit(f"ni zagonov z evals.json: {args.pattern}")

    for lab, zagoni in sorted(popis.items()):
        print(f"\n{lab}  ({len(zagoni)} zagonov)")
        for z in zagoni:
            print(f"   {z['config_id']:<24} seme {str(z['seme']):>4}  "
                  f"ohlajanje={'da' if z['anneal'] else 'ne'}  "
                  f"T_max={z['t_max']}  korakov={z['koraki']}")
    opozorila = preveri(popis)
    if opozorila:
        print("\nOPOZORILO: primerjava ni enofaktorska —")
        for o in opozorila:
            print("  -", o)
        print("\nZozi --pattern ali dodaj --by anneal, da se razlike locijo v svoje skupine.")
        if not args.vseeno:
            raise SystemExit("ustavljeno; za vseeno narisati uporabi --vseeno")
    print()
    if len(skupine) > len(SERIES):
        raise SystemExit(f"{len(skupine)} skupin je prevec za eno sliko; "
                         f"zozi --pattern ali razdeli na vec slik")

    opis = [f"{lab}: n = {len(z)}" for lab, z in sorted(skupine.items())]
    okno = max(0, args.glajenje)
    noga = ("Polna črta je mediana čez semena, senčeni pas sega od najmanjše do "
            "največje vrednosti. Vrednotenje je požrešno in na neoblikovani nagradi.")
    if okno >= 2:
        noga += (f" Krepka črta je drseča mediana širine {okno} vrednotenj, "
                 f"bleda pa neglajena mediana.")

    locene = len(skupine) > 2 and not args.skupaj
    if locene:
        fig = narisi_locene(skupine, okno, args.naslov)
        noga += " Sive črte v ozadju so mediane ostalih skupin."
        fig.text(0.012, 0.012, ovij(noga), fontsize=8, color=INK_2, va="bottom")
        fig.tight_layout(rect=(0, 0.055 if len(skupine) > 2 else 0.08, 1, 0.945))
    else:
        fig, ax = plt.subplots(figsize=(9.6, 4.8), facecolor=SURFACE)
        konci = []
        for i, (lab, zagoni) in enumerate(sorted(skupine.items())):
            x, y = poravnaj(zagoni)
            med = np.median(y, axis=0)
            col = SERIES[i]
            ax.fill_between(x, y.min(axis=0), y.max(axis=0), color=col, alpha=0.14, lw=0)
            if okno >= 2:
                ax.plot(x, med, color=col, lw=0.9, alpha=0.45, zorder=2)
            gl = drseca_mediana(med, okno)
            ax.plot(x, gl, color=col, lw=2.2, zorder=3,
                    label=f"{lab}  ($n$ = {len(zagoni)})")
            konci.append([float(gl[-1]), lab, col, float(x[-1])])

        # neposredne oznake ob koncu krivulj; ce koncajo blizu skupaj, jih razmaknemo,
        # da se ne prekrivajo (legenda ostane, a identiteta ni odvisna samo od nje)
        konci.sort(key=lambda r: r[0])
        MIN_RAZMIK = 0.075                       # v enotah osi y
        for j in range(1, len(konci)):
            if konci[j][0] - konci[j - 1][0] < MIN_RAZMIK:
                konci[j][0] = konci[j - 1][0] + MIN_RAZMIK
        for yy, lab, col, xx in konci:
            ax.annotate(lab, (xx, yy), textcoords="offset points", xytext=(8, 0),
                        va="center", fontsize=9, color=col, annotation_clip=False)

        olepsaj(ax)
        ax.set_xlabel("korakov okolja", fontsize=9, color=INK_2)
        ax.set_ylabel("delež uspeha pri vrednotenju", fontsize=9, color=INK_2)

        # legenda nad ploscico, da ne lezi na podatkih
        leg = ax.legend(frameon=False, fontsize=9, loc="lower left",
                        bbox_to_anchor=(0, 1.02), ncols=min(2, len(skupine)),
                        borderaxespad=0)
        for t in leg.get_texts():
            t.set_color(INK_2)

        fig.suptitle(args.naslov, fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
        fig.text(0.012, 0.012, ovij(noga), fontsize=8, color=INK_2, va="bottom")
        fig.tight_layout(rect=(0, 0.07, 0.78, 0.97))

    out = Path(args.out)
    fig.savefig(out, dpi=args.dpi, facecolor=SURFACE)
    print("shranjeno:", out)
    if out.suffix.lower() != ".pdf":
        pdf = out.with_suffix(".pdf")
        fig.savefig(pdf, facecolor=SURFACE)
        print("shranjeno:", pdf, " (za diplomo uporabi tega)")
    for o in opis:
        print("  ", o)


if __name__ == "__main__":
    main()
