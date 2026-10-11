"""Stevilke o straneh izhoda za razdelek o simetriji naucenih politik.

NE pozene nobenega ucenja in nobene simulacije. Bere samo polje "exit_sides" iz
runs/<mapa>/summary.json, ki ga zapise ze koncno vrednotenje.

Za vsako seme izracuna delez izhodov na desno med uspesnimi epizodami in iz tega
dvoje:

  - ali je POSAMEZNO seme enostransko (delez blizu 0 ali blizu 1),
  - ali so CEZ semena strani izenacene (povprecje delezev blizu 0,5).

To dvoje je mogoce zamenjati, zato ju izpisujemo loceno. Navede tudi najbolj
enostransko seme, da ga je mogoce v besedilu omeniti kot primer.

Ce je ista konfiguracija v vec zagonih (ponovitev), obvelja ZADNJI po casu v
imenu mape; enako kot pri plot_phase_a.py.

Uporaba:
  python scripts/strani_izhoda.py
  python scripts/strani_izhoda.py --pattern "runs/*phaseA_*" --min-uspehov 10
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def zberi(pattern: str, faza: str) -> tuple[dict, dict]:
    """Vrne {kappa: {seme: zapis}} in popis najdenih faz.

    Pri podvojeni (kappa, seme) obvelja ZADNJI; imena map se zacnejo z datumom,
    zato je abecedni vrstni red hkrati casovni in ponovitev povozi prvotni zagon.
    """
    po_kappi: dict = {}
    faze: dict = {}
    for d in sorted(glob.glob(pattern)):
        p = Path(d)
        try:
            summ = json.loads((p / "summary.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        try:
            cfg = json.loads((p / "config.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            cfg = {}
        oznaka_faze = str(cfg.get("phase", "?"))
        faze[oznaka_faze] = faze.get(oznaka_faze, 0) + 1
        if faza and faza.lower() not in oznaka_faze.lower():
            continue
        sides = summ.get("exit_sides")
        if not isinstance(sides, dict):
            continue
        kap, seme = summ.get("kappa"), summ.get("seed")
        if kap is None or seme is None:
            continue
        levo, desno = int(sides.get("left", 0)), int(sides.get("right", 0))
        po_kappi.setdefault(kap, {})[seme] = {
            "zagon": p.name, "levo": levo, "desno": desno, "uspehov": levo + desno,
            "delez_desno": (desno / (levo + desno)) if (levo + desno) else None,
            "koncni": summ.get("final_success_rate"),
        }
    return po_kappi, faze


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", default="runs/*")
    ap.add_argument("--faza", default="a",
                    help="obdrzi le zagone, katerih polje 'phase' v config.json vsebuje ta niz (privzeto 'a' za fazo A); prazen niz pomeni vse")
    ap.add_argument("--min-uspehov", type=int, default=5,
                    help="semena z manj uspesnimi epizodami izpusti iz povprecij "
                         "(pri enem ali dveh uspehih delez ni informativen)")
    ap.add_argument("--prag", type=float, default=0.9,
                    help="delez, nad katerim seme stejemo za enostransko")
    args = ap.parse_args()

    po_kappi, faze = zberi(args.pattern, args.faza)
    if not po_kappi:
        print("najdene faze med zagoni:", ", ".join(
            f"{k} ({n})" for k, n in sorted(faze.items())) or "(nobene)")
        raise SystemExit(
            f"ni zagonov faze '{args.faza}' s poljem exit_sides pri vzorcu "
            f"{args.pattern}; popravi --faza ali --pattern")

    for kap in sorted(po_kappi, reverse=True):
        semena = po_kappi[kap]
        print(f"\nkappa = {kap:g}   ({len(semena)} semen)")
        upostevana = []
        for seme in sorted(semena):
            z = semena[seme]
            d = z["delez_desno"]
            opomba = ""
            if z["uspehov"] < args.min_uspehov:
                opomba = "  <- premalo uspehov, izpusceno iz povprecij"
            else:
                upostevana.append((d, seme, z))
            print(f"   seme {seme:>4}  levo {z['levo']:>3}  desno {z['desno']:>3}  "
                  f"delez desno {('-' if d is None else f'{d:.3f}')}{opomba}")

        if not upostevana:
            print("   (nobeno seme nima dovolj uspehov)")
            continue

        delezi = [d for d, _, _ in upostevana]
        enostranska = sum(1 for d in delezi if d >= args.prag or d <= 1 - args.prag)
        print(f"   enostranskih semen (delez >= {args.prag:g} ali <= {1-args.prag:g}): "
              f"{enostranska}/{len(delezi)}")
        print(f"   povprecje delezev cez semena: {st.mean(delezi):.3f}  "
              f"(mediana {st.median(delezi):.3f})")
        skupaj_l = sum(z["levo"] for _, _, z in upostevana)
        skupaj_d = sum(z["desno"] for _, _, z in upostevana)
        print(f"   vse epizode skupaj: levo {skupaj_l}, desno {skupaj_d}  "
              f"(delez desno {skupaj_d/(skupaj_l+skupaj_d):.3f})")
        naj = max(upostevana, key=lambda r: abs(r[0] - 0.5))
        print(f"   najbolj enostransko seme: {naj[1]} "
              f"(levo {naj[2]['levo']}, desno {naj[2]['desno']}, "
              f"zagon {naj[2]['zagon']})")

    print("\nTi dve stvari nista isto in ju v besedilu locuj: posamezno seme je "
          "lahko mocno enostransko,\nhkrati pa je povprecje cez semena blizu 0,5. "
          "Prvo je lastnost politike, drugo pa posledica\ntega, da je naloga "
          "simetricna in da izbiro strani doloci naključje.")


if __name__ == "__main__":
    main()
