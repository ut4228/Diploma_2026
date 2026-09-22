"""Numerična integratorja. RK4 je privzet; Euler je tu samo za primerjavo v testu T4."""
from __future__ import annotations

from typing import Callable, Sequence

Vec = tuple[float, ...]


def rk4_step(f: Callable[[Vec], Vec], y: Vec, h: float) -> Vec:
    k1 = f(y)
    k2 = f(tuple(yi + 0.5 * h * ki for yi, ki in zip(y, k1)))
    k3 = f(tuple(yi + 0.5 * h * ki for yi, ki in zip(y, k2)))
    k4 = f(tuple(yi + h * ki for yi, ki in zip(y, k3)))
    return tuple(yi + h / 6.0 * (a + 2.0 * b + 2.0 * c + d) for yi, a, b, c, d in zip(y, k1, k2, k3, k4))


def euler_step(f: Callable[[Vec], Vec], y: Vec, h: float) -> Vec:
    k = f(y)
    return tuple(yi + h * ki for yi, ki in zip(y, k))


INTEGRATORS = {"rk4": rk4_step, "euler": euler_step}


def integrate(f: Callable[[Vec], Vec], y0: Sequence[float], h: float, n_steps: int, method: str = "rk4") -> list[Vec]:
    """Surova integracija brez dogodkov (za validacijske teste). Vrne seznam stanj dolžine n_steps+1."""
    step = INTEGRATORS[method]
    y = tuple(y0)
    out = [y]
    for _ in range(n_steps):
        y = step(f, y, h)
        out.append(y)
    return out
