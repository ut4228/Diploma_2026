"""Dinamični model vozička s palico na ukrivljenem tiru y = h(x).

Posplošeni koordinati: x (vodoravni položaj vozička) in psi (absolutni kot palice
glede na globalno navpičnico, pozitiven = težišče v smeri +x).
Stanje integratorja: (x, v, psi, om), v = dx/dt, om = dpsi/dt.

Enačbi gibanja (Lagrangeove enačbe druge vrste, simbolno preverjeno):

  (E1)  M a x'' + m_p l B psi'' = F - M h' h'' v^2 + m_p l C om^2 - M g h'
  (E2)  B x'' + (4/3) l psi''   = (g + h'' v^2) sin(psi)

  a = 1 + h'^2,  B = cos(psi) - h' sin(psi),  C = sin(psi) + h' cos(psi),  M = m_c + m_p

Predpostavke: homogena palica dolžine 2l (Florian 2007), g > 0, voziček točkasta masa,
zgib na nivoju tira (d = 0), brez trenja, dvostranska vez tira, sila F vodoravna (Q_x = F).

Lokalni kot glede na normalo tira (mentorjeva definicija):
  theta = psi + arctan(h'(x)),   theta' = om + h''/(1 + h'^2) * v
"""
from __future__ import annotations

import math
from dataclasses import dataclass

State = tuple[float, float, float, float]  # (x, v, psi, om)
Obs = tuple[float, float, float, float]  # (x, x_dot, theta, theta_dot)


@dataclass(frozen=True)
class PoleCartParams:
    m_c: float = 1.0  # masa vozička [kg] (Barto et al. 1983)
    m_p: float = 0.1  # masa palice [kg] (Barto et al. 1983)
    l: float = 0.5  # POLOVIČNA dolžina palice [m]; palica ima dolžino 2l (Florian 2007)
    g: float = 9.8  # težni pospešek [m/s^2], POZITIVEN (Florian 2007, razd. 3)

    @property
    def M(self) -> float:
        return self.m_c + self.m_p

    def describe(self) -> dict:
        return {"m_c": self.m_c, "m_p": self.m_p, "l": self.l, "g": self.g}


def accelerations(x: float, v: float, psi: float, om: float, F: float, track, prm: PoleCartParams) -> tuple[float, float]:
    """Vrne (x'', psi'') iz E1 in E2 (Cramerjevo pravilo; D > 0 vedno)."""
    h1 = track.dh(x)
    h2 = track.d2h(x)
    s = math.sin(psi)
    c = math.cos(psi)
    m_p, l, g = prm.m_p, prm.l, prm.g
    M = prm.m_c + m_p
    a = 1.0 + h1 * h1
    B = c - h1 * s
    C = s + h1 * c
    R1 = F - M * h1 * h2 * v * v + m_p * l * C * om * om - M * g * h1
    R2 = (g + h2 * v * v) * s
    D = l * ((4.0 / 3.0) * M * a - m_p * B * B)
    xdd = l * ((4.0 / 3.0) * R1 - m_p * B * R2) / D
    psidd = (M * a * R2 - B * R1) / D
    return xdd, psidd


def rhs(state: State, F: float, track, prm: PoleCartParams) -> State:
    x, v, psi, om = state
    xdd, psidd = accelerations(x, v, psi, om, F, track, prm)
    return (v, xdd, om, psidd)


def mass_matrix_det(x: float, psi: float, track, prm: PoleCartParams) -> float:
    h1 = track.dh(x)
    B = math.cos(psi) - h1 * math.sin(psi)
    return prm.l * ((4.0 / 3.0) * prm.M * (1.0 + h1 * h1) - prm.m_p * B * B)


def energy(state: State, track, prm: PoleCartParams) -> float:
    """Mehanska energija E = T + V."""
    x, v, psi, om = state
    h1 = track.dh(x)
    a = 1.0 + h1 * h1
    B = math.cos(psi) - h1 * math.sin(psi)
    T = 0.5 * prm.M * a * v * v + prm.m_p * prm.l * B * v * om + 0.5 * (4.0 / 3.0) * prm.m_p * prm.l**2 * om * om
    V = prm.M * prm.g * track.h(x) + prm.m_p * prm.g * prm.l * math.cos(psi)
    return T + V


def to_observation(state: State, track) -> Obs:
    """(x, v, psi, om) -> (x, x_dot, theta, theta_dot); theta glede na normalo tira."""
    x, v, psi, om = state
    h1 = track.dh(x)
    theta = psi + math.atan(h1)
    theta_dot = om + track.d2h(x) / (1.0 + h1 * h1) * v
    return (x, v, theta, theta_dot)


def from_observation(obs: Obs, track) -> State:
    x, xd, theta, theta_dot = obs
    h1 = track.dh(x)
    psi = theta - math.atan(h1)
    om = theta_dot - track.d2h(x) / (1.0 + h1 * h1) * xd
    return (x, xd, psi, om)


def normal_force(state: State, F: float, track, prm: PoleCartParams) -> float:
    """Reakcija tira N vzdolž normale n. N < 0: brez dvostranske vezi bi voziček odletel."""
    x, v, psi, om = state
    xdd, psidd = accelerations(x, v, psi, om, F, track, prm)
    h1 = track.dh(x)
    h2 = track.d2h(x)
    s, c = math.sin(psi), math.cos(psi)
    acx, acy = xdd, h1 * xdd + h2 * v * v
    apx = acx + prm.l * (psidd * c - om * om * s)
    apy = acy - prm.l * (psidd * s + om * om * c)
    norm = math.sqrt(1.0 + h1 * h1)
    nx, ny = -h1 / norm, 1.0 / norm  # n = (-sin phi, cos phi)
    sin_phi, cos_phi = h1 / norm, 1.0 / norm
    return (nx * (prm.m_c * acx + prm.m_p * apx) + ny * (prm.m_c * acy + prm.m_p * apy)
            + prm.M * prm.g * cos_phi + F * sin_phi)


def critical_force(valley, prm: PoleCartParams) -> float:
    """F_krit = M g tan(phi_max): sila za kvazistatično vožnjo po najstrmejšem delu (iz E1)."""
    return prm.M * prm.g * valley.tan_phi_max


def one_way_force(valley, prm: PoleCartParams) -> float:
    """F_E = M g H / W: najmanjša vodoravna sila, s katero točkasta masa M iz mirovanja na dnu
    doseže x = W brez obračanja. Delo vodoravne sile je natanko F * dx, zato mora veljati
    F x >= M g h(x) za vse x v [0, W]; za h = H |x/W|^p je to F >= M g H / W = F_krit / p.
    Velja za voziček brez palice (m_p kot del M, palica ne vpliva) - s palico je le približek."""
    return prm.M * prm.g * valley.H / valley.W


def f_max_from_k(k: float, valley, prm: PoleCartParams) -> float:
    return k * critical_force(valley, prm)
