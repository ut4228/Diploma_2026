"""RL okolje (sloj nad validiranim simulatorjem). Fizike ta modul ne spreminja.

  SUCCESS   : |x| >= W in |theta| <= 15 stopinj  (cilj je dosegljiv na OBEH straneh)
  FAIL_POLE : |theta| > 15 stopinj
  TIMEOUT   : PREKINITEV (truncation), NI terminalno stanje

Nagrada R-C:  +1 uspeh, -1 padec palice, 0 sicer in ob prekinitvi.
Okolje NE pozna gamma; diskontiranje je last agenta in metrik.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np

from .dynamics import PoleCartParams, energy
from .initial_states import TRAIN_NOISE, sample_train_state
from .simulator import HillCartSimulator, Outcome, SimConfig
from .tracks import PowerValley

N_ACTIONS = 3


class EnvOutcome(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    FAIL_POLE = "fail_pole"
    TIMEOUT = "timeout"

    @property
    def terminated(self) -> bool:
        return self in (EnvOutcome.SUCCESS, EnvOutcome.FAIL_POLE)

    @property
    def truncated(self) -> bool:
        return self is EnvOutcome.TIMEOUT


_SIM_TO_ENV = {
    Outcome.RUNNING: EnvOutcome.RUNNING,
    Outcome.SUCCESS: EnvOutcome.SUCCESS,
    Outcome.FAIL_LEFT: EnvOutcome.SUCCESS,
    Outcome.FAIL_POLE: EnvOutcome.FAIL_POLE,
    Outcome.TIMEOUT: EnvOutcome.TIMEOUT,
}


@dataclass(frozen=True)
class RewardRC:
    success: float = 1.0
    failure: float = -1.0
    step: float = 0.0
    potential: None = None

    def __call__(self, outcome: EnvOutcome) -> float:
        if outcome is EnvOutcome.SUCCESS:
            return self.success
        if outcome is EnvOutcome.FAIL_POLE:
            return self.failure
        return self.step

    def describe(self) -> dict:
        return {"type": "R-C", "success": self.success, "failure": self.failure, "step": self.step,
                "potential": None}


@dataclass(frozen=True)
class ExperimentSpec:
    config_id: str
    valley: PowerValley
    params: PoleCartParams
    sim: SimConfig
    F_max: float
    forces: dict
    tile_bounds: dict
    plan_sha256: str
    plan_version: str
    train_noise: tuple = TRAIN_NOISE

    @staticmethod
    def from_plan(plan_path, config_id: str) -> "ExperimentSpec":
        p = Path(plan_path)
        raw = p.read_bytes()
        plan = json.loads(raw.decode("utf-8"))
        rows = [r for r in plan["configs"] if r["id"] == config_id]
        if not rows:
            raise KeyError(f"konfiguracije {config_id!r} ni v {p}")
        row = rows[0]
        v = row["valley"]
        return ExperimentSpec(
            config_id=config_id,
            valley=PowerValley(H=v["H"], tan_phi_max=v["tan_phi_max"], p=v["p"]),
            params=PoleCartParams(**plan["params"]),
            sim=SimConfig(dt=plan["sim"]["dt"], n_substeps=plan["sim"]["n_substeps"],
                          theta_max_deg=plan["sim"]["theta_max_deg"], t_max=plan["sim"]["t_max"],
                          integrator=plan["sim"]["integrator"]),
            F_max=row["forces"]["F_max"], forces=row["forces"], tile_bounds=row["tile_bounds"],
            plan_sha256=hashlib.sha256(raw).hexdigest(), plan_version=plan["version"],
            train_noise=tuple(plan["initial_states"]["train_noise_uniform"]))

    def describe(self) -> dict:
        return {"config_id": self.config_id, "plan_version": self.plan_version, "plan_sha256": self.plan_sha256,
                "valley": self.valley.describe(), "params": self.params.describe(),
                "sim": {"dt": self.sim.dt, "n_substeps": self.sim.n_substeps, "t_max": self.sim.t_max,
                        "theta_max_deg": self.sim.theta_max_deg, "integrator": self.sim.integrator,
                        "max_steps": self.sim.max_steps},
                "forces": self.forces, "tile_bounds": self.tile_bounds, "train_noise": list(self.train_noise)}


class HillCartEnv:
    """Okolje z diskretnimi akcijami {0, 1, 2} -> {-F_max, 0, +F_max}."""

    def __init__(self, spec, reward=RewardRC(), check_pole: bool = True, track=None, F_max=None,
                 tile_bounds=None):
        self.spec = spec
        self.reward_fn = reward
        self.check_pole = check_pole
        self.track = track if track is not None else spec.valley
        self.F_max = spec.F_max if F_max is None else float(F_max)
        self.tile_bounds = spec.tile_bounds if tile_bounds is None else tile_bounds
        self.forces = (-self.F_max, 0.0, +self.F_max)
        self._sim = HillCartSimulator(self.track, self.F_max, spec.params, spec.sim, check_pole=check_pole)
        self.outcome = EnvOutcome.RUNNING
        self.obs = None
        self.oob_steps = 0

    @property
    def steps(self) -> int:
        return self._sim.steps

    @property
    def t(self) -> float:
        return self._sim.t

    @property
    def t_event(self):
        return self._sim.t_event

    @property
    def mechanical_energy(self) -> float:
        """Celotna mehanska energija trenutnega stanja [J]. Opazljiva kolicina, brez diskonta."""
        return energy(self._sim.state, self.track, self.spec.params)

    def _oob(self, obs) -> dict:
        b = self.tile_bounds
        return {"x_dot": not (b["x_dot"][0] <= obs[1] <= b["x_dot"][1]),
                "theta_dot": not (b["theta_dot"][0] <= obs[3] <= b["theta_dot"][1])}

    def reset(self, obs0=None, rng=None):
        if obs0 is None:
            if rng is None:
                raise ValueError("podaj obs0 ali rng")
            obs0 = sample_train_state(rng, self.spec.train_noise)
        self.obs = self._sim.reset(obs0)
        self.outcome = EnvOutcome.RUNNING
        self.oob_steps = 0
        return self.obs

    def step(self, action: int):
        if self.outcome is not EnvOutcome.RUNNING:
            raise RuntimeError("epizoda je koncana; poklici reset()")
        if not isinstance(action, (int, np.integer)) or not 0 <= int(action) < N_ACTIONS:
            raise ValueError(f"akcija mora biti 0, 1 ali 2, dobil {action!r}")
        F = self.forces[int(action)]
        obs, sim_out = self._sim.step(F)
        self.obs = obs
        self.outcome = _SIM_TO_ENV[sim_out]
        r = self.reward_fn(self.outcome)
        oob = self._oob(obs)
        self.oob_steps += int(oob["x_dot"] or oob["theta_dot"])
        info = {"outcome": self.outcome.value, "sim_outcome": sim_out.value, "F": F, "steps": self._sim.steps,
                "t": self._sim.t, "t_event": self._sim.t_event, "oob": oob, "oob_steps": self.oob_steps,
                "exit_side": ("right" if obs[0] > 0 else "left") if self.outcome is EnvOutcome.SUCCESS else None}
        return obs, r, self.outcome.terminated, self.outcome.truncated, info

    def describe(self) -> dict:
        return {"spec": self.spec.describe(), "reward": self.reward_fn.describe(), "n_actions": N_ACTIONS,
                "actions_N": list(self.forces), "check_pole": self.check_pole,
                "track": self.track.describe(), "F_max": self.F_max, "tile_bounds": self.tile_bounds,
                "outcome_mapping": {k.value: v.value for k, v in _SIM_TO_ENV.items()}}


def td_target(reward: float, q_next: float, terminated: bool, gamma: float) -> float:
    return reward if terminated else reward + gamma * q_next


def discounted_return(rewards, gamma: float) -> float:
    return float(sum((gamma**i) * r for i, r in enumerate(rewards)))


def mirror_obs(obs) -> tuple:
    return tuple(-v for v in obs)


def mirror_action(action: int) -> int:
    return N_ACTIONS - 1 - int(action)
