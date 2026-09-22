"""DIAGNOSTIČNE politike - samo za merjenje razponov stanj (OQ-12) in časovne skale
črpanja energije (OQ-8). NISO metode diplome in se ne primerjajo z RL metodami."""
from __future__ import annotations

import numpy as np


def random_policy(rng: np.random.Generator, F_max: float):
    actions = (-F_max, 0.0, F_max)
    return lambda obs: actions[int(rng.integers(3))]


def pumping_policy(F_max: float, first: int = +1):
    """F = F_max * sign(x_dot): dovaja energijo (moč F*x_dot >= 0).
    first: smer sile pri x_dot == 0 (npr. na začetku iz mirovanja); +1 desno, -1 levo."""
    def pol(obs):
        if obs[1] > 0.0:
            return F_max
        if obs[1] < 0.0:
            return -F_max
        return first * F_max
    return pol
