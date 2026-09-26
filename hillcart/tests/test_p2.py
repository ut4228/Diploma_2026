"""Testi faze P2: tile coding, redke sledi, true online Sarsa(lambda), vrednotenje.

Oznake P2-xx se ujemajo s specifikacijo.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda
from hillcart.env import ExperimentSpec, HillCartEnv
from hillcart.linear_sarsa import sarsa0_update, true_online_update
from hillcart.tile_coding import TileCoder, TileCoderConfig, coder_from_bounds
from hillcart.training import evaluate, run_episode

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "plan_v1.json"
CONFIG_ID = "H1_tan0.5_p2_kap0.5"
GAMMA = 0.999


@pytest.fixture(scope="module")
def spec():
    return ExperimentSpec.from_plan(PLAN, CONFIG_ID)


@pytest.fixture(scope="module")
def coder(spec):
    b = dict(spec.tile_bounds)
    b["x"] = [-spec.valley.W, spec.valley.W]
    return coder_from_bounds(b)


def small_coder(n_intervals=4, n_tilings=16):
    return TileCoder(TileCoderConfig(lows=(-1.0, -1.0, -1.0, -1.0), highs=(1.0, 1.0, 1.0, 1.0),
                                     n_tilings=n_tilings, n_intervals=n_intervals, n_actions=3))


# ---------------------------------------------------------------- P2-01
def test_P2_01_exactly_n_tilings_active(coder):
    rng = np.random.default_rng(1)
    for _ in range(200):
        obs = tuple(rng.uniform(lo, hi) for lo, hi in zip(coder.lows, coder.highs))
        a = int(rng.integers(3))
        idx = coder.features(obs, a)
        assert idx.shape == (coder.cfg.n_tilings,)
        x = coder.dense(obs, a)
        assert x.sum() == pytest.approx(len(np.unique(idx)))
        assert set(np.unique(x)) <= {0.0, 1.0}


# ---------------------------------------------------------------- P2-02
def test_P2_02_action_blocks_are_disjoint(coder):
    rng = np.random.default_rng(2)
    for _ in range(100):
        obs = tuple(rng.uniform(lo, hi) for lo, hi in zip(coder.lows, coder.highs))
        sets = [set(coder.features(obs, a).tolist()) for a in range(3)]
        assert sets[0].isdisjoint(sets[1]) and sets[1].isdisjoint(sets[2]) and sets[0].isdisjoint(sets[2])
        for s in sets:
            assert all(0 <= i < coder.n_features for i in s)
    with pytest.raises(ValueError):
        coder.features((0.0, 0.0, 0.0, 0.0), 3)


# ---------------------------------------------------------------- P2-03
def test_P2_03_offsets_and_index_range(coder):
    n, k = coder.cfg.n_tilings, coder.k
    off = coder._offsets
    assert off.shape == (n, k)
    for i in range(k):  # vseh 16 zamikov razlicnih in enakomerno porazdeljenih
        vals = np.sort(off[:, i] * n).astype(int)
        assert np.array_equal(vals, np.arange(n))
    rng = np.random.default_rng(3)
    for _ in range(300):
        u = rng.uniform(0.0, 1.0, size=k)
        idx = np.floor(u * coder.cfg.n_intervals + off).astype(int)
        assert idx.min() >= 0 and idx.max() <= coder.cfg.n_intervals
    # tudi na robovih in izven njih
    for u in (np.zeros(k), np.ones(k), -np.ones(k), 2 * np.ones(k)):
        idx = np.floor(np.clip(u, 0, 1) * coder.cfg.n_intervals + off).astype(int)
        assert idx.min() >= 0 and idx.max() <= coder.cfg.n_intervals


# ---------------------------------------------------------------- P2-04
def test_P2_04_generalization(coder):
    """Bliznji stanji si delita vecino plosic; nasprotna robova nobene."""
    mid = tuple((lo + hi) / 2 for lo, hi in zip(coder.lows, coder.highs))
    assert np.array_equal(coder.features(mid, 1), coder.features(mid, 1))
    # premik za 1/8 sirine plosice (= 1/64 razpona) v eni dimenziji
    delta = coder.span[1] / (coder.cfg.n_intervals * 8)
    near = (mid[0], mid[1] + delta, mid[2], mid[3])
    shared = len(set(coder.features(mid, 1).tolist()) & set(coder.features(near, 1).tolist()))
    assert shared >= 14, shared
    far = tuple(lo + 1e-9 for lo in coder.lows)
    other = tuple(hi - 1e-9 for hi in coder.highs)
    assert not (set(coder.features(far, 1).tolist()) & set(coder.features(other, 1).tolist()))


# ---------------------------------------------------------------- P2-05
def test_P2_05_clipping_and_oob_flags(coder):
    inside = tuple((lo + hi) / 2 for lo, hi in zip(coder.lows, coder.highs))
    assert not coder.out_of_bounds(inside).any()
    beyond = (inside[0], coder.highs[1] + 100.0, inside[2], coder.lows[3] - 100.0)
    flags = coder.out_of_bounds(beyond)
    assert flags[1] and flags[3] and not flags[0] and not flags[2]
    idx = coder.features(beyond, 0)  # ne sme vreci izjeme
    assert idx.min() >= 0 and idx.max() < coder.n_features
    # clipping: se bolj oddaljeno stanje da enake znacilke
    farther = (inside[0], coder.highs[1] + 1000.0, inside[2], coder.lows[3] - 1000.0)
    assert np.array_equal(idx, coder.features(farther, 0))


# ---------------------------------------------------------------- P2-06
def test_P2_06_features_deterministic_and_consistent(coder):
    rng = np.random.default_rng(6)
    for _ in range(100):
        obs = tuple(rng.uniform(lo, hi) for lo, hi in zip(coder.lows, coder.highs))
        a = int(rng.integers(3))
        assert np.array_equal(coder.features(obs, a), coder.features(obs, a))
        assert np.array_equal(coder.features_all_actions(obs)[a], coder.features(obs, a))


# ---------------------------------------------------------------- P2-07
def test_P2_07_sparse_traces_equal_dense_reference():
    """Redka implementacija sledi == gosta referenca (linear_sarsa.true_online_update)."""
    c = small_coder(n_intervals=3, n_tilings=8)
    cfg = AgentConfig(alpha0=0.4, lam=0.9, gamma=GAMMA, epsilon=0.0, eps_z=0.0)
    ag = TrueOnlineSarsaLambda(c, cfg, seed=0)
    w = np.zeros(c.n_features)
    z = np.zeros(c.n_features)
    q_old = 0.0
    rng = np.random.default_rng(7)
    obs = lambda: tuple(rng.uniform(-1, 1, size=4))
    x_idx = c.features(obs(), int(rng.integers(3)))
    ag.begin_episode()
    for t in range(200):
        terminated = (t % 23 == 22)
        r = float(rng.normal())
        x2_idx = None if terminated else c.features(obs(), int(rng.integers(3)))
        x = np.zeros(c.n_features)
        x[x_idx] = 1.0
        x2 = np.zeros(c.n_features)
        if x2_idx is not None:
            x2[x2_idx] = 1.0
        w, z, q_old = true_online_update(w, z, q_old, x, x2, r, ag.alpha, cfg.gamma, cfg.lam, terminated)
        ag.update(x_idx, r, x2_idx, terminated)
        assert np.allclose(ag.w, w, atol=1e-10, rtol=0), t
        assert np.allclose(ag._z, z, atol=1e-10, rtol=0), t
        if terminated:
            ag.end_episode()
            z[:] = 0.0
            q_old = 0.0
            x_idx = c.features(obs(), int(rng.integers(3)))
        else:
            x_idx = x2_idx


def test_P2_07b_pruning_keeps_result_close():
    """Obrezovanje pri eps_z = 1e-6 ne spremeni utezi bistveno (< 1e-4 relativno)."""
    c = small_coder(n_intervals=3, n_tilings=8)
    base = AgentConfig(alpha0=0.4, lam=0.9, gamma=GAMMA, epsilon=0.0)
    runs = {}
    for eps_z in (0.0, 1e-6):
        ag = TrueOnlineSarsaLambda(c, AgentConfig(**{**base.__dict__, "eps_z": eps_z}), seed=0)
        rng = np.random.default_rng(8)
        ag.begin_episode()
        x_idx = c.features(tuple(rng.uniform(-1, 1, size=4)), 0)
        for t in range(400):
            terminated = (t % 51 == 50)
            r = float(rng.normal())
            x2 = None if terminated else c.features(tuple(rng.uniform(-1, 1, size=4)), int(rng.integers(3)))
            ag.update(x_idx, r, x2, terminated)
            if terminated:
                ag.end_episode()
                x_idx = c.features(tuple(rng.uniform(-1, 1, size=4)), 0)
            else:
                x_idx = x2
        runs[eps_z] = ag.w.copy()
    err = np.abs(runs[0.0] - runs[1e-6]).max() / max(np.abs(runs[0.0]).max(), 1e-12)
    assert err < 1e-4, err


# ---------------------------------------------------------------- P2-08
def test_P2_08_true_online_lambda0_equals_sarsa0_with_tiles():
    """Razsiritev P0-18 na prave plosice: pri lambda = 0 sta pravili enaki."""
    c = small_coder(n_intervals=4, n_tilings=16)
    cfg = AgentConfig(alpha0=0.1, lam=0.0, gamma=GAMMA, epsilon=0.0, eps_z=0.0)
    ag = TrueOnlineSarsaLambda(c, cfg, seed=0)
    w_ref = np.zeros(c.n_features)
    rng = np.random.default_rng(18)
    ag.begin_episode()
    x_idx = c.features(tuple(rng.uniform(-1, 1, size=4)), 0)
    for t in range(150):
        terminated = (t % 17 == 16)
        r = float(rng.normal())
        x2_idx = None if terminated else c.features(tuple(rng.uniform(-1, 1, size=4)), int(rng.integers(3)))
        x = np.zeros(c.n_features)
        x[x_idx] = 1.0
        x2 = np.zeros(c.n_features)
        if x2_idx is not None:
            x2[x2_idx] = 1.0
        w_ref = sarsa0_update(w_ref, x, x2, r, ag.alpha, cfg.gamma, terminated)
        ag.update(x_idx, r, x2_idx, terminated)
        assert np.allclose(ag.w, w_ref, atol=1e-12, rtol=0), t
        if terminated:
            ag.end_episode()
            x_idx = c.features(tuple(rng.uniform(-1, 1, size=4)), 0)
        else:
            x_idx = x2_idx


# ---------------------------------------------------------------- P2-09
def test_P2_09_greedy_evaluation_is_deterministic_and_readonly(spec, coder):
    env = HillCartEnv(spec, check_pole=False)
    ag = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=0.5), seed=0)
    ag.w = np.random.default_rng(9).normal(size=coder.n_features) * 0.01
    states = [(0.0, 0.0, 0.0, 0.0), (0.1, 0.2, 0.0, 0.0), (-0.1, -0.2, 0.01, 0.0)]
    w0 = ag.w.copy()
    a = evaluate(env, ag, states, GAMMA)
    b = evaluate(env, ag, states, GAMMA)
    assert a == b
    assert np.array_equal(ag.w, w0)


def test_P2_09b_epsilon_greedy_tie_breaking_is_uniform(coder):
    """Pri w = 0 (vse vrednosti enake) izbira ne sme biti pristranska k akciji 0."""
    ag = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=0.0), seed=0)
    counts = np.zeros(3, dtype=int)
    for _ in range(3000):
        counts[ag.act((0.0, 0.0, 0.0, 0.0), greedy=True)] += 1
    assert counts.min() > 800, counts


# ---------------------------------------------------------------- pomozno
def test_feature_counts_match_specification(coder):
    d = coder.describe()
    assert d["n_tilings"] == 16 and d["n_intervals"] == 8 and d["n_indices_per_dim"] == 9
    assert d["n_features_per_action"] == 16 * 9**4 == 104976
    assert d["n_features_total"] == 3 * 104976 == 314928


def test_run_episode_respects_budget_accounting(spec, coder):
    env = HillCartEnv(spec, check_pole=False)
    ag = TrueOnlineSarsaLambda(coder, AgentConfig(), seed=0)
    out = run_episode(env, ag, (0.0, 0.0, 0.0, 0.0), greedy=False, learn=True, gamma=GAMMA)
    outcome, steps, t_event, total_r, disc, max_theta, oob, side = out
    assert steps == env.steps and 0 < steps <= spec.sim.max_steps
    assert abs(total_r) in (0.0, 1.0)
    if outcome.value == "success":
        assert disc == pytest.approx(GAMMA ** (steps - 1), rel=1e-9)
