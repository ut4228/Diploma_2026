"""Eksperimentalni načrt v1: en dejavnik naenkrat okoli osnovne točke.

Glavna normalizacija moči motorja:  kappa = F_max / F_E,  F_E = M g H / W
  kappa >= 1: voziček (brez palice) lahko pride iz kotanje z enosmerno silo,
  kappa <  1: črpanje energije je nujno.
Dodatno se beležita k = F_max / F_krit (F_krit = M g tan_phi_max) in u = F_max / (M g).

Serije (vsaka spreminja EN geometrijski parameter, ostala dva sta na osnovni vrednosti):
  S0_kappa : osnovna kotanja, vseh 5 kappa (tu primerjamo RL metode)
  S1_depth : H         in {0.5, 1.0, 1.5}  -> pri fiksnih tan, p se kotanja samo raztegne
                                               (W ~ H), F_E = M g tan/p ostane enak
  S2_slope : tan_phi   in {0.3, 0.5, 0.7}  -> W = pH/tan se spremeni; pri fiksnem kappa je F_max ~ tan
  S3_shape : p         in {2, 3, 4}        -> W = pH/tan se spremeni; pri fiksnem kappa je F_max ~ 1/p
Posledice sklopitve (W, F_max) so nujne in se v diplomi eksplicitno navedejo.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dynamics import PoleCartParams, critical_force, one_way_force
from .tracks import PowerValley

BASE = dict(H=1.0, tan_phi_max=0.5, p=2.0)
KAPPA_VALUES = (1.2, 0.9, 0.7, 0.5, 0.3)

GEOMETRY_SERIES = {
    "S1_depth": [dict(BASE, H=h) for h in (0.5, 1.0, 1.5)],
    "S2_slope": [dict(BASE, tan_phi_max=t) for t in (0.3, 0.5, 0.7)],
    "S3_shape": [dict(BASE, p=p) for p in (2.0, 3.0, 4.0)],
}


@dataclass(frozen=True)
class ExperimentConfig:
    valley: PowerValley
    kappa: float

    @property
    def id(self) -> str:
        v = self.valley
        return f"H{v.H:g}_tan{v.tan_phi_max:g}_p{v.p:g}_kap{self.kappa:g}"

    def forces(self, prm: PoleCartParams = PoleCartParams()) -> dict:
        F_E = one_way_force(self.valley, prm)
        F = self.kappa * F_E
        return {"F_max": F, "F_E": F_E, "F_krit": critical_force(self.valley, prm), "kappa": self.kappa,
                "k": F / critical_force(self.valley, prm), "u": F / (prm.M * prm.g)}


def unique_valleys() -> list[PowerValley]:
    seen, out = set(), []
    for cfgs in GEOMETRY_SERIES.values():
        for c in cfgs:
            key = (c["H"], c["tan_phi_max"], c["p"])
            if key not in seen:
                seen.add(key)
                out.append(PowerValley(**c))
    return out


def all_configs() -> list[ExperimentConfig]:
    return [ExperimentConfig(v, k) for v in unique_valleys() for k in KAPPA_VALUES]


def series_of(cfg: ExperimentConfig) -> list[str]:
    """Kateri seriji pripada konfiguracija (osnovna kotanja pripada vsem)."""
    v = cfg.valley
    out = []
    if (v.H, v.tan_phi_max, v.p) == (BASE["H"], BASE["tan_phi_max"], BASE["p"]):
        out.append("S0_kappa")
    for name, cfgs in GEOMETRY_SERIES.items():
        if any((c["H"], c["tan_phi_max"], c["p"]) == (v.H, v.tan_phi_max, v.p) for c in cfgs):
            out.append(name)
    return out
