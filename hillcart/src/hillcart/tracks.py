"""Geometrija tira y = h(x).

Vsak tir implementira h(x), h'(x) in h''(x). Dinamični model (E1, E2) potrebuje
samo h, h' in h'' (h mora biti C^2), ker integriramo v absolutnem kotu psi.

- PowerValley: glavna družina kotanj h(x) = H * |x / W|^p, p >= 2.
  Parametrizirana z opisnimi parametri (H, tan_phi_max, p); W je izpeljan:
  W = p * H / tan_phi_max  (največji naklon je pri x = +-W).
- FlatTrack, LinearTrack: samo za validacijske teste (T2, T3, T8, T12).
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PowerValley:
    """Simetrična kotanja h(x) = H |x/W|^p.

    H           globina kotanje [m] (višina roba nad dnom)
    tan_phi_max največji naklon tira (pri x = +-W) [-]
    p           eksponent oblike, p >= 2 (p = 2 je parabola)
    """

    H: float
    tan_phi_max: float
    p: float = 2.0

    def __post_init__(self) -> None:
        if not self.H > 0:
            raise ValueError("H mora biti > 0")
        if not self.tan_phi_max > 0:
            raise ValueError("tan_phi_max mora biti > 0")
        if not self.p >= 2:
            raise ValueError("p mora biti >= 2 (h mora biti C^2)")

    @property
    def W(self) -> float:
        """Polovična širina kotanje; cilj je pri x = W, neuspeh pri x = -W."""
        return self.p * self.H / self.tan_phi_max

    @property
    def x_goal(self) -> float:
        return self.W

    @property
    def x_fail_left(self) -> float:
        return -self.W

    def h(self, x: float) -> float:
        return self.H * abs(x / self.W) ** self.p

    def dh(self, x: float) -> float:
        u = x / self.W
        return math.copysign(self.p * self.H / self.W * abs(u) ** (self.p - 1.0), u)

    def d2h(self, x: float) -> float:
        u = abs(x / self.W)
        return self.p * (self.p - 1.0) * self.H / self.W**2 * u ** (self.p - 2.0)

    def describe(self) -> dict:
        return {
            "type": "PowerValley",
            "H": self.H,
            "tan_phi_max": self.tan_phi_max,
            "phi_max_deg": math.degrees(math.atan(self.tan_phi_max)),
            "p": self.p,
            "W": self.W,
        }


@dataclass(frozen=True)
class FlatTrack:
    """h(x) = 0. Samo za teste (limita Florian 2007)."""

    half_length: float = 10.0

    @property
    def x_goal(self) -> float:
        return self.half_length

    @property
    def x_fail_left(self) -> float:
        return -self.half_length

    def h(self, x: float) -> float:
        return 0.0

    def dh(self, x: float) -> float:
        return 0.0

    def d2h(self, x: float) -> float:
        return 0.0

    def describe(self) -> dict:
        return {"type": "FlatTrack", "half_length": self.half_length}


@dataclass(frozen=True)
class LinearTrack:
    """h(x) = x * tan_phi. Raven nagnjen tir, samo za test T8."""

    tan_phi: float
    half_length: float = 10.0

    @property
    def x_goal(self) -> float:
        return self.half_length

    @property
    def x_fail_left(self) -> float:
        return -self.half_length

    def h(self, x: float) -> float:
        return x * self.tan_phi

    def dh(self, x: float) -> float:
        return self.tan_phi

    def d2h(self, x: float) -> float:
        return 0.0

    def describe(self) -> dict:
        return {"type": "LinearTrack", "tan_phi": self.tan_phi}
