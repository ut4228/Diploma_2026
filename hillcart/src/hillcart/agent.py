"""True online Sarsa(lambda) z linearno aproksimacijo in binarnimi znacilkami (tile coding).

Po Sutton & Barto (2018), razd. 12.7, str. 307:

  q       = w^T x,      q' = 0 (terminalno) ali w^T x' (tudi ob PREKINITVI)
  delta   = R + gamma q' - q
  z      <- gamma lambda z + (1 - alpha gamma lambda z^T x) x        (nizozemska sled)
  w      <- w + alpha (delta + q - q_old) z - alpha (q - q_old) x
  q_old  <- q'

Ob zacetku IN koncu epizode: z = 0, q_old = 0.

Ucinkovitost: z ima ~3e5 komponent, zato se sled hrani gosto, decay pa se izvaja samo
nad mnozico aktivnih indeksov, ki se obrezuje pod pragom eps_z (privzeto 1e-6).
Pri gamma*lambda = 0.8991 sled pade pod prag po ~130 korakih -> do ~2000 aktivnih indeksov.
Enakovrednost z gosto referencno implementacijo preverja test P2-07.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgentConfig:
    alpha0: float = 0.1  # alpha0 = alpha * n_tilings (brezdimenzijsko, primerljivo)
    lam: float = 0.9
    gamma: float = 0.999
    epsilon: float = 0.05
    w_init: float = 0.0  # zacetna vrednost vsake utezi (q0 = n_tilings * w_init)
    eps_z: float = 1e-6  # prag obrezovanja sledi

    def describe(self, n_tilings: int) -> dict:
        return {"algorithm": "true_online_sarsa_lambda", "alpha0": self.alpha0,
                "alpha": self.alpha0 / n_tilings, "lambda": self.lam, "gamma": self.gamma,
                "epsilon": self.epsilon, "w_init": self.w_init, "q_init": n_tilings * self.w_init,
                "eps_z": self.eps_z, "trace": "dutch (true online)",
                "tie_breaking": "nakljucno med enakimi najvecjimi vrednostmi"}


class TrueOnlineSarsaLambda:
    def __init__(self, coder, cfg: AgentConfig = AgentConfig(), seed: int = 0):
        self.coder = coder
        self.cfg = cfg
        self.n_tilings = coder.cfg.n_tilings
        self.alpha = cfg.alpha0 / self.n_tilings
        self.rng = np.random.default_rng(seed)
        self.w = np.full(coder.n_features, float(cfg.w_init))
        self._z = np.zeros(coder.n_features)
        self._active = np.empty(0, dtype=np.int64)
        self._in_active = np.zeros(coder.n_features, dtype=bool)
        self.q_old = 0.0
        self.max_active = 0

    # ------------------------------------------------------------------ vrednosti
    def q_values(self, obs) -> np.ndarray:
        idx = self.coder.features_all_actions(obs)  # (n_actions, n_tilings)
        return self.w[idx].sum(axis=1)

    def q(self, idx: np.ndarray) -> float:
        return float(self.w[idx].sum())

    def act(self, obs, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.cfg.epsilon:
            return int(self.rng.integers(self.coder.cfg.n_actions))
        qs = self.q_values(obs)
        best = np.flatnonzero(qs == qs.max())
        return int(best[0]) if best.size == 1 else int(self.rng.choice(best))

    # ------------------------------------------------------------------ sledi
    def begin_episode(self) -> None:
        if self._active.size:
            self._z[self._active] = 0.0
        self._active = np.empty(0, dtype=np.int64)
        self._in_active[:] = False
        self.q_old = 0.0

    end_episode = begin_episode

    def _add_active(self, idx: np.ndarray) -> None:
        new = idx[~self._in_active[idx]]
        if new.size:
            new = np.unique(new)
            self._in_active[new] = True
            self._active = np.concatenate((self._active, new))

    def _prune(self) -> None:
        if not self._active.size:
            return
        vals = self._z[self._active]
        keep = np.abs(vals) > self.cfg.eps_z
        if not keep.all():
            drop = self._active[~keep]
            self._z[drop] = 0.0
            self._in_active[drop] = False
            self._active = self._active[keep]

    # ------------------------------------------------------------------ posodobitev
    def update(self, idx: np.ndarray, reward: float, idx_next, terminated: bool) -> float:
        """En korak. idx = znacilke (S, A), idx_next = znacilke (S', A') ali None pri terminaciji.
        Ob PREKINITVI (timeout) se poda idx_next in terminated = False (bootstrap)."""
        glam = self.cfg.gamma * self.cfg.lam
        q = self.q(idx)
        q_next = 0.0 if terminated else self.q(idx_next)
        delta = reward + self.cfg.gamma * q_next - q

        z_dot_x = float(self._z[idx].sum())
        if self._active.size:
            self._z[self._active] *= glam
        self._z[idx] += 1.0 - self.alpha * glam * z_dot_x
        self._add_active(idx)

        coef = self.alpha * (delta + q - self.q_old)
        act = self._active
        self.w[act] += coef * self._z[act]
        self.w[idx] -= self.alpha * (q - self.q_old)

        self.q_old = q_next
        self._prune()
        self.max_active = max(self.max_active, int(self._active.size))
        return delta

    # ------------------------------------------------------------------ pomozno
    def describe(self) -> dict:
        return {**self.cfg.describe(self.n_tilings), "tile_coding": self.coder.describe()}

    def snapshot(self) -> np.ndarray:
        return self.w.copy()
