"""Diagnostične meritve za načrtovanje eksperimentov (NE metode diplome, NE del RL treninga).

1) pumping_reference: idealni čas črpanja BREZ omejitve palice (check_pole=False) z
   hevristiko F = F_max sign(x_dot), oba začetna potiska (desno/levo). Referenčna časovna skala.
2) state_ranges: razponi x_dot in theta_dot po virih, ločeno:
     "random"         naključna politika, učna začetna stanja, s pogojem palice
     "pumping_pole"   hevristika črpanja, učna začetna stanja, s pogojem palice
     "pumping_nopole" hevristika črpanja brez pogoja palice; samo x_dot in hitrost vrtenja
                      normale r = phi'(x) x_dot (theta tu fizikalno ni smiseln)
   Beležijo se samo veljavna stanja (|theta| <= theta_max; pri nopole do konca epizode).
"""
from __future__ import annotations

import math

import numpy as np

from .diagnostic_policies import pumping_policy, random_policy
from .dynamics import PoleCartParams
from .initial_states import sample_train_state
from .simulator import HillCartSimulator, Outcome, SimConfig


def pumping_reference(valley, F_max: float, prm: PoleCartParams = PoleCartParams(), horizon: float = 300.0) -> dict:
    out = {}
    for name, first in (("first_right", +1), ("first_left", -1)):
        sim = HillCartSimulator(valley, F_max, prm, SimConfig(t_max=horizon), check_pole=False)
        pol = pumping_policy(F_max, first=first)
        obs = sim.reset((0.0, 0.0, 0.0, 0.0))
        rev, prev = 0, 0.0
        while sim.outcome is Outcome.RUNNING:
            a = pol(obs)
            if prev and a != prev:
                rev += 1
            prev = a
            obs, _ = sim.step(a)
        out[name] = {"outcome": sim.outcome.value, "t": sim.t_event, "reversals": rev}
    succ = [d for d in out.values() if d["outcome"] == "success"]
    best = min(succ, key=lambda d: d["t"]) if succ else None
    out["t_pump"] = best["t"] if best else math.inf
    out["reversals"] = best["reversals"] if best else None
    return out


def _new_range():
    return {"x_dot": [math.inf, -math.inf], "theta_dot": [math.inf, -math.inf], "r": [math.inf, -math.inf],
            "episodes": 0, "steps": 0, "outcomes": {}}


def _upd(rng_, key, val):
    rng_[key][0] = min(rng_[key][0], val)
    rng_[key][1] = max(rng_[key][1], val)


def state_ranges(valley, F_max: float, sim_cfg: SimConfig, episodes: int, seed: int,
                 prm: PoleCartParams = PoleCartParams()) -> dict:
    rng = np.random.default_rng(seed)
    res = {}
    for src in ("random", "pumping_pole"):
        r = _new_range()
        pol = random_policy(rng, F_max) if src == "random" else pumping_policy(F_max)
        for _ in range(episodes):
            sim = HillCartSimulator(valley, F_max, prm, sim_cfg)
            obs = sim.reset(sample_train_state(rng))
            while True:
                _upd(r, "x_dot", obs[1])
                _upd(r, "theta_dot", obs[3])
                if sim.outcome is not Outcome.RUNNING:
                    break
                obs, out = sim.step(pol(obs))
                r["steps"] += 1
                if out is Outcome.FAIL_POLE:
                    break
            r["episodes"] += 1
            r["outcomes"][sim.outcome.value] = r["outcomes"].get(sim.outcome.value, 0) + 1
        del r["r"]
        res[src] = r
    r = _new_range()
    del r["theta_dot"]
    for first in (+1, -1):
        sim = HillCartSimulator(valley, F_max, prm, SimConfig(dt=sim_cfg.dt, n_substeps=sim_cfg.n_substeps, t_max=300.0),
                                check_pole=False)
        pol = pumping_policy(F_max, first=first)
        obs = sim.reset((0.0, 0.0, 0.0, 0.0))
        while True:
            x, xd = obs[0], obs[1]
            h1 = valley.dh(x)
            _upd(r, "x_dot", xd)
            _upd(r, "r", valley.d2h(x) / (1 + h1 * h1) * xd)
            if sim.outcome is not Outcome.RUNNING:
                break
            obs, _ = sim.step(pol(obs))
            r["steps"] += 1
        r["episodes"] += 1
        r["outcomes"][sim.outcome.value] = r["outcomes"].get(sim.outcome.value, 0) + 1
    res["pumping_nopole"] = r
    return res


def bounds_from_ranges(ranges: dict, margin: float) -> dict:
    """Simetrične meje (problem je simetričen).
    x_dot     = max |x_dot| čez vse vire * (1 + margin)
    theta_dot = (max |theta_dot| iz random, pumping_pole + max |r| iz pumping_nopole) * (1 + margin)
    """
    ax = lambda a: max(abs(a[0]), abs(a[1]))
    m_xd = max(ax(ranges[s]["x_dot"]) for s in ("random", "pumping_pole", "pumping_nopole")) * (1 + margin)
    m_td = (max(ax(ranges[s]["theta_dot"]) for s in ("random", "pumping_pole"))
            + ax(ranges["pumping_nopole"]["r"])) * (1 + margin)
    return {"x_dot": [-m_xd, m_xd], "theta_dot": [-m_td, m_td], "margin": margin}
