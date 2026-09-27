"""Potencialno oblikovanje nagrade (Ng, Harada & Russell 1999).

Oblika (izrek 1 v viru):

    F(s, a, s') = gamma * Phi(s') - Phi(s)

Ta oblika je zadosten pogoj, da se mnozica optimalnih politik ne spremeni; brez dodatnega
znanja o MDP je tudi potreben. Terminalno (absorbirajoce) stanje ima po definiciji Phi = 0,
zato je ob terminaciji F = -Phi(s). Prekinitev (timeout) ni terminalno stanje, zato se tam
uporabi obicajna oblika F = gamma * Phi(s') - Phi(s) in bootstrap se ohrani.

Potencial po energiji (nasa izbira, glej specifikacijo razd. 7):

    Phi_E(s) = c * (E(s) - E_0) / (M g H)

  E(s)  celotna mehanska energija sistema (dynamics.energy),
  E_0   energija zacetnega stanja (0, 0, 0, 0) - dno kotanje, mirovanje, palica pokoncno,
  M g H referencna energija za dvig celotne mase za globino kotanje.

Normalizacija je izbrana tako, da velja Phi_E = 0 na dnu in Phi_E = 1 na robu kotanje
(kvazistaticno, torej pri niclni hitrosti in pokoncni palici).

Motivacija: pri kappa < 1 je ovira energija, ne polozaj. Potencial po polozaju (Phi ~ x) bi
med ucenjem kaznoval gibanje v levo, ki je za crpanje nujno; potencial po energiji raste pri
crpanju v obe smeri. Padec palice energije ne ustvari.

"""
from __future__ import annotations

from dataclasses import dataclass

from .dynamics import energy


@dataclass(frozen=True)
class EnergyPotential:
    """Phi_E = c * (E - E_0) / scale. Brez gamma: gamma pripada oblikovanju, ne potencialu."""

    c: float
    E0: float
    scale: float

    @classmethod
    def for_env(cls, env, c: float = 1.0) -> "EnergyPotential":
        track, prm = env.track, env.spec.params
        H = getattr(track, "H", None)
        if H is None:
            raise ValueError("Phi_E zahteva kotanjo z globino H (PowerValley)")
        return cls(c=float(c), E0=energy((0.0, 0.0, 0.0, 0.0), track, prm), scale=prm.M * prm.g * H)

    def phi(self, env) -> float:
        return self.c * (env.mechanical_energy - self.E0) / self.scale

    @staticmethod
    def shaping_reward(phi_s: float, phi_s_next: float, gamma: float, terminated: bool) -> float:
        """F = gamma * Phi(s') - Phi(s); ob terminaciji je Phi(s') = 0 (absorbirajoce stanje)."""
        return (0.0 if terminated else gamma * phi_s_next) - phi_s

    def describe(self) -> dict:
        return {"type": "potential_based", "potential": "Phi_E (energija)", "c": self.c,
                "E0_J": self.E0, "scale_MgH_J": self.scale,
                "form": "F = gamma * Phi(s') - Phi(s), Phi(terminal) = 0",
                "reference": "Ng, Harada & Russell (1999), izrek 1",
                "note": "Phi_E = 0 na dnu, 1 na robu kotanje (kvazistaticno)"}
