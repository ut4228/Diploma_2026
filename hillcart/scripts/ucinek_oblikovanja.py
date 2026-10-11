"""Stevilke za razdelek o ucinku oblikovanja nagrade.

Slika krivulj pokaze potek, trditev v besedilu pa mora stati na stevilkah.
Ta skript iz istih zagonov, iz katerih je narisana slika, izpise za vsako
skupino mediano, najmanjso in najvecjo vrednost korakov do prvega uspeha ter
koncni delez uspeha, in izracuna razmerje med skupinami. Nicesar ne oceni na
oko in nicesar ne povzame iz pogovora.

Skupine loci enako kot plot_krivulje.py (--by), da se slika in besedilo
nanasata na iste zagone.

Uporaba:
  python scripts/ucinek_oblikovanja.py --pattern "runs/*p3b_*kap0.9_s*" --by reward anneal
  python scripts/ucinek_oblikovanja.py --pattern "runs/*p3b_*" --by reward --izloci tmax
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

IME = {"plain": "R-C (brez oblikovanja)", "shaped": "R-C + Phi_E"}


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
            deli.append(f"kappa={kap:g}" if kap is not None else str(cfg.get("config_id", "?")))
        else:
            deli.append(str(cfg.get(k, "?")))
    return ", ".join(deli)


def zberi(pattern: str, by: tuple, izloci: tuple) -> dict:
    skupine: dict[str, list] = {}
    for d in sorted(glob.glob(pattern)):
        p = Path(d)
        if any(x in p.name for x in izloci):
            continue
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
            summ = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            evals = json.loads((p / "evals.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            evals = []
        skupine.setdefault(oznaka(cfg, summ, by), []).append({
            "zagon": p.name,
            "seme": summ.get("seed"),
            "prvi": summ.get("first_success_step"),
            "konec": summ.get("final_success_rate"),
            "najboljsi": summ.get("best_success_rate"),
            "koraki": cfg.get("protocol", {}).get("total_steps"),
            "config_id": cfg.get("config_id", "?"),
            "evals": evals,
        })
    return skupine


def prvi_dosezek(evals: list, prag: float):
    """Korak, pri katerem vrednotenje prvic doseze prag; None, ce ga nikoli."""
    for e in evals:
        if e.get("success_rate") is not None and e["success_rate"] >= prag:
            return e["step"]
    return None


def dvig_skupine(zagoni: list, prag: float):
    """Korak, pri katerem MEDIANA skupine prvic doseze prag.

    To je tisto, kar se na sliki vidi kot dvig krivulje, in NI isto kot korak
    prvega uspeha: slednji pove, kdaj je agent cilj prvic sploh zadel, ta pa,
    kdaj je politika postala zanesljiva na vseh zacetnih stanjih vrednotenja.
    """
    serije = [z["evals"] for z in zagoni if z["evals"]]
    if not serije:
        return None
    n = min(len(s) for s in serije)
    for i in range(n):
        med = st.median([s[i]["success_rate"] for s in serije])
        if med >= prag:
            return serije[0][i]["step"]
    return None


def povzetek(zagoni: list) -> dict:
    prvi = [z["prvi"] for z in zagoni if z["prvi"] is not None]
    konec = [z["konec"] for z in zagoni if z["konec"] is not None]
    najb = [z["najboljsi"] for z in zagoni if z["najboljsi"] is not None]
    return {
        "n": len(zagoni),
        "n_uspelih": len(prvi),
        "prvi_med": st.median(prvi) if prvi else None,
        "prvi_min": min(prvi) if prvi else None,
        "prvi_max": max(prvi) if prvi else None,
        "konec_med": st.median(konec) if konec else None,
        "konec_min": min(konec) if konec else None,
        "konec_max": max(konec) if konec else None,
        "najb_med": st.median(najb) if najb else None,
    }


def st_ali(v, obl="{:.0f}") -> str:
    return obl.format(v) if v is not None else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/*p3b_*")
    ap.add_argument("--by", nargs="+", default=["reward"],
                    choices=["reward", "anneal", "config_id"])
    ap.add_argument("--izloci", nargs="*", default=["tmax"])
    args = ap.parse_args()

    skupine = zberi(args.pattern, tuple(args.by), tuple(args.izloci))
    if not skupine:
        raise SystemExit(f"ni zagonov: {args.pattern}")

    povz = {}
    for lab in sorted(skupine):
        zagoni = skupine[lab]
        p = povzetek(zagoni)
        povz[lab] = p
        print(f"\n{lab}   (n = {p['n']})")
        for z in sorted(zagoni, key=lambda z: (z["seme"] is None, z["seme"])):
            print(f"   seme {str(z['seme']):>4}  prvi uspeh {st_ali(z['prvi']):>10}  "
                  f"konec {st_ali(z['konec'], '{:.3f}'):>6}  "
                  f"najboljsi {st_ali(z['najboljsi'], '{:.3f}'):>6}  "
                  f"({z['zagon']})")
        print(f"   prvi uspeh: med {st_ali(p['prvi_med'])}  "
              f"[{st_ali(p['prvi_min'])}, {st_ali(p['prvi_max'])}]  "
              f"uspelo {p['n_uspelih']}/{p['n']}")
        print(f"   koncni delez: med {st_ali(p['konec_med'], '{:.3f}')}  "
              f"[{st_ali(p['konec_min'], '{:.3f}')}, {st_ali(p['konec_max'], '{:.3f}')}]")
        d5, d9 = dvig_skupine(zagoni, 0.5), dvig_skupine(zagoni, 0.9)
        p["dvig50"], p["dvig90"] = d5, d9
        print(f"   dvig mediane: >=0,5 pri {st_ali(d5)}   >=0,9 pri {st_ali(d9)}")
        posam5 = [prvi_dosezek(z["evals"], 0.5) for z in zagoni if z["evals"]]
        if posam5 and all(v is not None for v in posam5):
            print(f"   po semenih >=0,5: {sorted(posam5)}")

    # razmerja: vsaka skupina proti prvi po abecedi, ki je navadno neoblikovana
    labs = sorted(povz)
    osnova = labs[0]
    if povz[osnova]["prvi_med"]:
        print(f"\nrazmerja prvega uspeha glede na '{osnova}':")
        for lab in labs[1:]:
            a, b = povz[osnova]["prvi_med"], povz[lab]["prvi_med"]
            if b:
                print(f"   {lab}: {a / b:.2f}x hitreje  ({st_ali(a)} -> {st_ali(b)})")
            else:
                print(f"   {lab}: ni primerljivo (nobeno seme ni uspelo)")

    print(f"\ndvig mediane glede na '{osnova}':")
    for prag, klj in ((0.5, "dvig50"), (0.9, "dvig90")):
        a = povz[osnova].get(klj)
        vrstica = [f"   prag {prag:g}: {osnova} = {st_ali(a)}"]
        for lab in labs[1:]:
            b = povz[lab].get(klj)
            raz = f" ({a / b:.2f}x)" if (a and b) else ""
            vrstica.append(f"{lab} = {st_ali(b)}{raz}")
        print("\n      ".join(vrstica))

    print("\nOpozorilo: pri treh semenih na skupino je mediana groba ocena; "
          "razmerje navedi z razponom, ne kot natancno vrednost.")
    print("Korak prvega uspeha in dvig mediane NISTA ista kolicina: prvi pove, "
          "kdaj je agent cilj prvic zadel,\ndrugi pa, kdaj je politika postala "
          "zanesljiva na vseh stanjih vrednotenja. V podnapisu slike\nnavajaj "
          "dvig, v tabeli prvi uspeh.")


if __name__ == "__main__":
    main()
