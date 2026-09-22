"""Testi pravil epizode (niso fizikalni): cilj, neuspehi, prekinitev, prednost dogodkov."""
from __future__ import annotations

import math

import pytest

from hillcart.dynamics import PoleCartParams, f_max_from_k
from hillcart.initial_states import eval_states
from hillcart.simulator import HillCartSimulator, Outcome, SimConfig
from hillcart.tracks import PowerValley

V = PowerValley(H=1.0, tan_phi_max=0.5, p=2.0)
PRM = PoleCartParams()


def test_timeout_is_truncation_after_max_steps():
    sim = HillCartSimulator(V, F_max=1.0, cfg=SimConfig(t_max=0.2))
    sim.reset((0.0, 0.0, 0.0, 0.0))  # ravnovesje: F = 0 -> ostane
    while sim.outcome is Outcome.RUNNING:
        sim.step(0.0)
    assert sim.outcome is Outcome.TIMEOUT
    assert sim.steps == 10 and sim.outcome.truncated and not sim.outcome.terminated


def test_default_t_max_is_500_decision_steps():
    assert SimConfig().max_steps == 500


def test_success_at_right_rim():
    sim = HillCartSimulator(V, F_max=1.0, check_pole=False)
    sim.reset((V.W - 0.01, 2.0, 0.0, 0.0))
    sim.step(0.0)
    assert sim.outcome is Outcome.SUCCESS and sim.outcome.terminated


def test_fail_left_rim():
    sim = HillCartSimulator(V, F_max=1.0, check_pole=False)
    sim.reset((-V.W + 0.01, -2.0, 0.0, 0.0))
    sim.step(0.0)
    assert sim.outcome is Outcome.FAIL_LEFT


def test_fail_pole():
    sim = HillCartSimulator(V, F_max=1.0)
    sim.reset((0.0, 0.0, math.radians(14.9), 3.0))
    sim.step(0.0)
    assert sim.outcome is Outcome.FAIL_POLE


def test_pole_failure_has_priority_over_goal():
    sim = HillCartSimulator(V, F_max=1.0)
    sim.reset((V.W - 0.01, 2.0, math.radians(14.99), 5.0))
    sim.step(0.0)
    assert sim.outcome is Outcome.FAIL_POLE


def test_force_limit_and_step_after_end():
    sim = HillCartSimulator(V, F_max=1.0, cfg=SimConfig(t_max=0.02))
    sim.reset((0.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        sim.step(1.5)
    sim.step(0.0)
    with pytest.raises(RuntimeError):
        sim.step(0.0)


def test_terminal_initial_state_rejected():
    sim = HillCartSimulator(V, F_max=1.0)
    with pytest.raises(ValueError):
        sim.reset((0.0, 0.0, math.radians(20), 0.0))


def test_eval_set_is_fixed_and_symmetric():
    s = eval_states()
    assert len(s) == 81 and (0.0, 0.0, 0.0, 0.0) in s
    assert all(tuple(-v for v in e) in s for e in s)


def test_f_max_from_k():
    assert f_max_from_k(1.0, V, PRM) == pytest.approx(PRM.M * PRM.g * 0.5)
