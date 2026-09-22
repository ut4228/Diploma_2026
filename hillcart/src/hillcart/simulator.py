"""Simulator epizode: fizika + pravila terminacije. Brez nagrade (ta pride v RL okolju).

Pravila (dogovorjeno z mentorjem):
  - odločitev agenta vsakih dt = 0.02 s, sila konstantna čez korak (zero-order hold),
  - integracija RK4 z n_substeps podkoraki,
  - dogodki se preverjajo po VSAKEM podkoraku, v tem vrstnem redu:
      1. |theta| > theta_max          -> FAIL_POLE   (palica ima prednost pred ciljem)
      2. x >= x_goal (= W)            -> SUCCESS
      3. x <= -W                      -> FAIL_LEFT
  - če do t_max ni dogodka          -> TIMEOUT (prekinitev / truncation, NI terminalno stanje)

t_max = 10 s je ZAČETNA vrednost, ki se preveri empirično (scripts/check_tmax.py).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

from .dynamics import PoleCartParams, from_observation, rhs, to_observation
from .integrators import INTEGRATORS


class Outcome(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAIL_POLE = "fail_pole"
    FAIL_LEFT = "fail_left"
    TIMEOUT = "timeout"

    @property
    def terminated(self) -> bool:
        return self in (Outcome.SUCCESS, Outcome.FAIL_POLE, Outcome.FAIL_LEFT)

    @property
    def truncated(self) -> bool:
        return self is Outcome.TIMEOUT


@dataclass(frozen=True)
class SimConfig:
    dt: float = 0.02  # korak odločanja agenta [s]
    n_substeps: int = 4  # RK4 podkoraki na korak odločanja (preverjeno s T10)
    theta_max_deg: float = 15.0  # meja odklona palice od normale (mentor)
    t_max: float = 10.0  # EMPIRIČNO PREVERITI (OQ-8) - ni fizikalna konstanta
    integrator: str = "rk4"

    @property
    def max_steps(self) -> int:
        return int(round(self.t_max / self.dt))

    @property
    def theta_max(self) -> float:
        return math.radians(self.theta_max_deg)


class HillCartSimulator:
    def __init__(self, track, F_max: float, params: PoleCartParams = PoleCartParams(), cfg: SimConfig = SimConfig(),
                 check_pole: bool = True):
        if F_max < 0:
            raise ValueError("F_max mora biti >= 0")
        self.track = track
        self.F_max = F_max
        self.prm = params
        self.cfg = cfg
        self.check_pole = check_pole  # False samo za diagnostiko (časovna skala črpanja)
        self._step_fn = INTEGRATORS[cfg.integrator]
        self._h = cfg.dt / cfg.n_substeps
        self.state = None
        self.steps = 0
        self.t_event = None
        self.outcome = Outcome.RUNNING

    def reset(self, obs0) -> tuple:
        """obs0 = (x, x_dot, theta, theta_dot); theta glede na normalo."""
        self.state = from_observation(tuple(float(v) for v in obs0), self.track)
        self.steps = 0
        self.t_event = None
        self.outcome = Outcome.RUNNING
        ev = self._check_events(self.state)
        if ev is not None:
            raise ValueError(f"Začetno stanje je že terminalno: {ev}")
        return self.observation

    @property
    def observation(self) -> tuple:
        return to_observation(self.state, self.track)

    @property
    def t(self) -> float:
        return self.steps * self.cfg.dt

    def _check_events(self, state):
        x = state[0]
        if self.check_pole:
            theta = to_observation(state, self.track)[2]
            if abs(theta) > self.cfg.theta_max:
                return Outcome.FAIL_POLE
        if x >= self.track.x_goal:
            return Outcome.SUCCESS
        if x <= self.track.x_fail_left:
            return Outcome.FAIL_LEFT
        return None

    def step(self, F: float) -> tuple[tuple, Outcome]:
        if self.outcome is not Outcome.RUNNING:
            raise RuntimeError("Epizoda je končana; pokliči reset().")
        if abs(F) > self.F_max * (1 + 1e-12) + 1e-15:
            raise ValueError(f"|F| = {abs(F)} presega F_max = {self.F_max}")
        f = lambda y: rhs(y, F, self.track, self.prm)
        y = self.state
        for i in range(self.cfg.n_substeps):
            y = self._step_fn(f, y, self._h)
            ev = self._check_events(y)
            if ev is not None:
                self.state = y
                self.steps += 1
                self.t_event = (self.steps - 1) * self.cfg.dt + (i + 1) * self._h
                self.outcome = ev
                return self.observation, self.outcome
        self.state = y
        self.steps += 1
        if self.steps >= self.cfg.max_steps:
            self.outcome = Outcome.TIMEOUT
            self.t_event = self.t
        return self.observation, self.outcome

    def describe(self) -> dict:
        return {"track": self.track.describe(), "params": self.prm.describe(), "sim": asdict(self.cfg),
                "F_max": self.F_max, "check_pole": self.check_pole}
