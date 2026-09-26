"""Testi faze P0: RL okolje, pravila epizode, nagrada R-C, cilj posodobitve, ekvivalenca lambda = 0.

Oznake P0-xx se ujemajo s specifikacijo. Fizika se v tej fazi ne spreminja (glej test_physics.py).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from hillcart.env import (ExperimentSpec, HillCartEnv, EnvOutcome, N_ACTIONS, RewardRC, discounted_return,
                          mirror_action, mirror_obs, td_target)
from hillcart.initial_states import TRAIN_NOISE, eval_states, sample_train_state
from hillcart.linear_sarsa import sarsa0_update, true_online_update

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "plan_v1.json"
CONFIG_ID = "H1_tan0.5_p2_kap0.5"
GAMMA = 0.999


@pytest.fixture(scope="module")
def spec():
    return ExperimentSpec.from_plan(PLAN, CONFIG_ID)


def make_env(spec, **kw):
    return HillCartEnv(spec, **kw)


def run_actions(env, obs0, actions):
    """Izvede zaporedje akcij; vrne (obs_list, rewards, outcome, info)."""
    obs = env.reset(obs0)
    obs_list, rewards, info = [obs], [], None
    for a in actions:
        obs, r, term, trunc, info = env.step(a)
        obs_list.append(obs)
        rewards.append(r)
        if term or trunc:
            break
    return obs_list, rewards, env.outcome, info


# ---------------------------------------------------------------- P0-01
def test_P0_01_spec_loaded_from_plan(spec):
    plan = json.loads(PLAN.read_text(encoding="utf-8"))
    row = [r for r in plan["configs"] if r["id"] == CONFIG_ID][0]
    assert spec.sim.max_steps == 2000
    assert spec.sim.t_max == 40.0
    assert spec.sim.dt == 0.02 and spec.sim.n_substeps == 4
    assert spec.sim.theta_max_deg == 15.0
    assert spec.F_max == row["forces"]["F_max"]
    assert spec.valley.W == pytest.approx(row["valley"]["W"])
    assert spec.plan_sha256 == hashlib.sha256(PLAN.read_bytes()).hexdigest()
    assert spec.train_noise == tuple(TRAIN_NOISE)


# ---------------------------------------------------------------- P0-02
def test_P0_02_action_mapping(spec):
    env = make_env(spec)
    expected = (-spec.F_max, 0.0, spec.F_max)
    for a in range(N_ACTIONS):
        env.reset((0.0, 0.0, 0.0, 0.0))
        _, _, _, _, info = env.step(a)
        assert info["F"] == expected[a]
    env.reset((0.0, 0.0, 0.0, 0.0))
    for bad in (-1, 3, 1.5, "1", None):
        with pytest.raises(ValueError):
            env.step(bad)


# ---------------------------------------------------------------- P0-03
def test_P0_03_zero_reward_on_normal_step(spec):
    env = make_env(spec)
    env.reset((0.0, 0.0, 0.0, 0.0))
    _, r, term, trunc, _ = env.step(1)
    assert r == 0.0 and term is False and trunc is False


# ---------------------------------------------------------------- P0-04, P0-05
def test_P0_04_success_right(spec):
    env = make_env(spec)
    W = spec.valley.W
    env.reset((W - 0.01, 2.0, 0.0, 0.0))
    obs, r, term, trunc, info = env.step(1)
    assert env.outcome is EnvOutcome.SUCCESS
    assert r == 1.0 and term is True and trunc is False
    assert obs[0] >= W and info["exit_side"] == "right"


def test_P0_05_goal_boundary_inclusive(spec):
    env = make_env(spec)
    W = spec.valley.W
    env.reset((W - 1e-6, 1.0, 0.0, 0.0))
    obs, r, term, _, _ = env.step(1)
    assert term and obs[0] >= W and r == 1.0


# ---------------------------------------------------------------- P0-06
def test_P0_06_fail_pole(spec):
    env = make_env(spec)
    env.reset((0.0, 0.0, math.radians(14.9), 3.0))
    _, r, term, trunc, info = env.step(1)
    assert env.outcome is EnvOutcome.FAIL_POLE
    assert r == -1.0 and term is True and trunc is False and info["exit_side"] is None


# ---------------------------------------------------------------- P0-07 (nova definicija)
def test_P0_07_left_rim_is_success(spec):
    """Levi rob NI več neuspeh: |x| >= W je cilj na obeh straneh."""
    env = make_env(spec)
    W = spec.valley.W
    env.reset((-W + 0.01, -2.0, 0.0, 0.0))
    obs, r, term, trunc, info = env.step(1)
    assert env.outcome is EnvOutcome.SUCCESS
    assert r == 1.0 and term is True and trunc is False
    assert obs[0] <= -W and info["exit_side"] == "left"
    assert info["sim_outcome"] == "fail_left"  # simulator ostaja nespremenjen


# ---------------------------------------------------------------- P0-08
def test_P0_08_pole_failure_has_priority(spec):
    env = make_env(spec)
    W = spec.valley.W
    env.reset((W - 0.01, 2.0, math.radians(14.99), 5.0))
    _, r, term, _, _ = env.step(1)
    assert env.outcome is EnvOutcome.FAIL_POLE and r == -1.0 and term is True


# ---------------------------------------------------------------- P0-09
def test_P0_09_timeout_is_truncation(spec):
    env = make_env(spec)
    obs = env.reset((0.0, 0.0, 0.0, 0.0))
    r = term = trunc = None
    while env.outcome is EnvOutcome.RUNNING:
        obs, r, term, trunc, _ = env.step(1)
        assert obs == (0.0, 0.0, 0.0, 0.0)  # točno ravnovesje (T9)
    assert env.outcome is EnvOutcome.TIMEOUT
    assert env.steps == 2000 and env.t == pytest.approx(40.0)
    assert r == 0.0 and term is False and trunc is True


# ---------------------------------------------------------------- P0-10
def test_P0_10_event_time_resolution(spec):
    env = make_env(spec)
    env.reset((0.0, 0.0, math.radians(14.0), 3.0))
    while env.outcome is EnvOutcome.RUNNING:
        env.step(1)
    n, dt, h = env.steps, spec.sim.dt, spec.sim.dt / spec.sim.n_substeps
    assert (n - 1) * dt < env.t_event <= n * dt + 1e-12
    assert env.t_event / h == pytest.approx(round(env.t_event / h))


# ---------------------------------------------------------------- P0-11, P0-12
def test_P0_11_terminal_initial_state_rejected(spec):
    env = make_env(spec)
    W = spec.valley.W
    for bad in ((0.0, 0.0, math.radians(20), 0.0), (W, 0.0, 0.0, 0.0), (-W, 0.0, 0.0, 0.0)):
        with pytest.raises(ValueError):
            env.reset(bad)


def test_P0_12_step_after_episode_end(spec):
    env = make_env(spec)
    env.reset((spec.valley.W - 0.01, 2.0, 0.0, 0.0))
    env.step(1)
    with pytest.raises(RuntimeError):
        env.step(1)


# ---------------------------------------------------------------- P0-13
def test_P0_13_episode_reward_sum(spec):
    env = make_env(spec)
    rng = np.random.default_rng(0)
    for _ in range(20):
        obs0 = sample_train_state(rng, spec.train_noise)
        actions = [int(rng.integers(N_ACTIONS)) for _ in range(2000)]
        _, rewards, outcome, _ = run_actions(env, obs0, actions)
        total = sum(rewards)
        expected = {EnvOutcome.SUCCESS: 1.0, EnvOutcome.FAIL_POLE: -1.0, EnvOutcome.TIMEOUT: 0.0}[outcome]
        assert total == expected
        assert all(r == 0.0 for r in rewards[:-1])


# ---------------------------------------------------------------- P0-14 (nova definicija)
def test_P0_14_mirror_symmetry_of_environment(spec):
    """Zrcaljenje stanja in akcij ne spremeni izida ne nagrade (cilj na obeh straneh)."""
    env = make_env(spec)
    rng = np.random.default_rng(14)
    for _ in range(10):
        obs0 = sample_train_state(rng, spec.train_noise)
        actions = [int(rng.integers(N_ACTIONS)) for _ in range(2000)]
        obs_a, rew_a, out_a, _ = run_actions(env, obs0, actions)
        obs_b, rew_b, out_b, _ = run_actions(env, mirror_obs(obs0), [mirror_action(a) for a in actions])
        assert out_a is out_b
        assert rew_a == rew_b
        assert len(obs_a) == len(obs_b)
        for ya, yb in zip(obs_a, obs_b):
            assert max(abs(u + w) for u, w in zip(ya, yb)) < 1e-9


# ---------------------------------------------------------------- P0-15
def test_P0_15_determinism(spec):
    def fingerprint():
        env = make_env(spec)
        rng = np.random.default_rng(7)
        h = hashlib.sha256()
        for _ in range(5):
            obs0 = sample_train_state(rng, spec.train_noise)
            actions = [int(rng.integers(N_ACTIONS)) for _ in range(2000)]
            obs_list, rewards, outcome, info = run_actions(env, obs0, actions)
            h.update(repr((obs_list, rewards, outcome.value, info["t_event"], info["steps"])).encode())
        return h.hexdigest()

    assert fingerprint() == fingerprint()


# ---------------------------------------------------------------- P0-16
def test_P0_16_initial_states(spec):
    rng = np.random.default_rng(3)
    for _ in range(500):
        s = sample_train_state(rng, spec.train_noise)
        assert all(abs(v) <= a for v, a in zip(s, spec.train_noise))
    ev = eval_states()
    assert len(ev) == 81 and (0.0, 0.0, 0.0, 0.0) in ev
    assert all(mirror_obs(e) in ev for e in ev)
    assert eval_states() == ev


# ---------------------------------------------------------------- P0-17
def test_P0_17_out_of_bounds_counter(spec):
    env = make_env(spec)
    b = spec.tile_bounds
    env.reset((0.0, 0.0, 0.0, 0.0))
    assert env._oob((0.0, 0.0, 0.0, 0.0)) == {"x_dot": False, "theta_dot": False}
    assert env._oob((0.0, b["x_dot"][1] + 1.0, 0.0, 0.0))["x_dot"] is True
    assert env._oob((0.0, 0.0, 0.0, b["theta_dot"][0] - 1.0))["theta_dot"] is True
    env.reset((0.0, 0.0, 0.0, 0.0))
    _, _, _, _, info = env.step(1)
    assert info["oob_steps"] == 0 and env.oob_steps == 0


# ---------------------------------------------------------------- P0-18
def test_P0_18_true_online_lambda0_equals_sarsa0():
    rng = np.random.default_rng(18)
    d, n_active, alpha = 100, 16, 0.00625
    w_a = np.zeros(d)
    w_b = np.zeros(d)
    z = np.zeros(d)
    q_old = 0.0

    def feat():
        x = np.zeros(d)
        x[rng.choice(d, n_active, replace=False)] = 1.0
        return x

    x = feat()
    for t in range(100):
        terminated = (t % 17 == 16)
        x_next = feat()
        r = float(rng.normal())
        w_b, z, q_old = true_online_update(w_b, z, q_old, x, x_next, r, alpha, GAMMA, 0.0, terminated)
        w_a = sarsa0_update(w_a, x, x_next, r, alpha, GAMMA, terminated)
        assert np.allclose(w_a, w_b, atol=1e-12, rtol=0)
        if terminated:  # nova epizoda
            z = np.zeros(d)
            q_old = 0.0
            x = feat()
        else:
            x = x_next


# ---------------------------------------------------------------- P0-19, P0-20
def test_P0_19_target_on_termination():
    assert td_target(1.0, 0.7, True, GAMMA) == 1.0
    assert td_target(-1.0, 123.0, True, GAMMA) == -1.0


def test_P0_20_bootstrap_on_truncation():
    assert td_target(0.0, 0.7, False, GAMMA) == pytest.approx(0.6993, abs=1e-12)


# ---------------------------------------------------------------- P0-21
@pytest.mark.parametrize("T,expected", [(1, 1.0), (3, 0.998001), (500, 0.999**499), (2000, 0.999**1999)])
def test_P0_21_discounted_return(T, expected):
    rewards = [0.0] * (T - 1) + [1.0]
    assert discounted_return(rewards, GAMMA) == pytest.approx(expected, rel=1e-12)
    assert discounted_return([0.0] * 1999 + [-1.0], GAMMA) == pytest.approx(-(0.999**1999), rel=1e-12)


# ---------------------------------------------------------------- P0-22, P0-23
def test_P0_22_env_has_no_gamma(spec):
    """Okolje ne pozna gamma; diskontiranje je last agenta in metrik."""
    import inspect

    env = make_env(spec)
    d = env.describe()
    assert "gamma" not in json.dumps(d)
    for cls in (HillCartEnv, RewardRC, ExperimentSpec):
        assert "gamma" not in inspect.getsource(cls)


def test_P0_23_shaping_disabled_by_default(spec):
    env = make_env(spec)
    assert env.reward_fn.potential is None
    assert env.reward_fn.describe()["potential"] is None


# ---------------------------------------------------------------- P0-24, P0-25
def test_P0_24_physics_untouched():
    files = ["src/hillcart/dynamics.py", "src/hillcart/tracks.py", "src/hillcart/integrators.py",
             "src/hillcart/simulator.py"]
    try:
        out = subprocess.check_output(["git", "diff", "plan-v1", "--", *files], cwd=ROOT, text=True,
                                      stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git ali oznaka plan-v1 ni na voljo")
    assert out.strip() == ""


def test_P0_25_plan_unchanged():
    try:
        prefix = subprocess.check_output(["git", "rev-parse", "--show-prefix"], cwd=ROOT, text=True,
                                         stderr=subprocess.DEVNULL).strip()
        res = subprocess.run(["git", "diff", "--quiet", "plan-v1", "--", f"{prefix}configs/plan_v1.json"],
                             cwd=ROOT, stderr=subprocess.DEVNULL)
    except OSError:
        pytest.skip("git ni na voljo")
    if res.returncode not in (0, 1):
        pytest.skip("oznaka plan-v1 ni na voljo")
    assert res.returncode == 0, "configs/plan_v1.json se je spremenil glede na oznako plan-v1"