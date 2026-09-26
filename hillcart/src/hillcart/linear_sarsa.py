"""Čisti posodobitveni pravili za linearno aproksimacijo (brez učne zanke, brez tile codinga).

V fazi P0 sta tu samo zato, da lahko preverimo ekvivalenco pri lambda = 0
(test P0-18). Agent, izbira akcij in tile coding pridejo kasneje.

Oznake po Sutton & Barto (2018):
  q(s,a) = w^T x(s,a), delta = R + gamma * q(S',A') - q(S,A)
  True online Sarsa(lambda), razd. 12.7 (str. 307), nizozemska (dutch) sled.
Ob terminaciji je q(S',A') = 0 in bootstrapa ni; ob prekinitvi (timeout) bootstrap ostane.
"""
from __future__ import annotations

import numpy as np


def sarsa0_update(w: np.ndarray, x: np.ndarray, x_next: np.ndarray, reward: float,
                  alpha: float, gamma: float, terminated: bool) -> np.ndarray:
    """Enostopenjska Sarsa (lambda = 0), polgradientna: w <- w + alpha * delta * x."""
    q = float(w @ x)
    q_next = 0.0 if terminated else float(w @ x_next)
    delta = reward + gamma * q_next - q
    return w + alpha * delta * x


def true_online_update(w: np.ndarray, z: np.ndarray, q_old: float, x: np.ndarray, x_next: np.ndarray,
                       reward: float, alpha: float, gamma: float, lam: float, terminated: bool):
    """En korak true online Sarsa(lambda) (S&B 2018, razd. 12.7, str. 307).

    Vrne (w, z, q_old_next). Klicatelj po koncu epizode postavi z = 0 in q_old = 0.
    Pri lambda = 0 se reducira natanko na sarsa0_update (test P0-18).
    """
    q = float(w @ x)
    q_next = 0.0 if terminated else float(w @ x_next)
    delta = reward + gamma * q_next - q
    z = gamma * lam * z + (1.0 - alpha * gamma * lam * float(z @ x)) * x
    w = w + alpha * (delta + q - q_old) * z - alpha * (q - q_old) * x
    return w, z, q_next
