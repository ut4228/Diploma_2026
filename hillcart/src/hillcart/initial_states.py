"""Začetna stanja v opazovalnih koordinatah (x, x_dot, theta, theta_dot).

Učenje: enakomeren šum okoli s0 = (0, 0, 0, 0).
Vrednotenje: fiksna, deterministična množica (vogali in središča škatle šuma, 3^4 = 81 stanj).
Amplitude šuma so PREDLOG in se zapišejo v konfiguracijo vsakega zagona.
"""
from __future__ import annotations

import itertools

import numpy as np

# (x [m], x_dot [m/s], theta [rad], theta_dot [rad/s])
TRAIN_NOISE = (0.05, 0.05, 0.05, 0.05)


def sample_train_state(rng: np.random.Generator, noise=TRAIN_NOISE) -> tuple:
    return tuple(float(rng.uniform(-a, a)) for a in noise)


def eval_states(noise=TRAIN_NOISE) -> list[tuple]:
    levels = [(-a, 0.0, a) for a in noise]
    return [tuple(s) for s in itertools.product(*levels)]
