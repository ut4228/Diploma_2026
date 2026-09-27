"""Testi potencialnega oblikovanja nagrade (Ng, Harada & Russell 1999) - faza P3b."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from hillcart.agent import AgentConfig, TrueOnlineSarsaLambda
from hillcart.env import EnvOutcome, ExperimentSpec, HillCartEnv
from hillcart.shaping import EnergyPotential
from hillcart.tile_coding import coder_from_bounds
from hillcart.training import run_episode

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs" / "plan_v2.json"
CONFIG_ID = "H1_tan0.5_p2_kap0.9"


@pytest.fixture(scope="module")
def parts():
    spec = ExperimentSpec.from_plan(PLAN, CONFIG_ID)
    bounds = {"x": [-spec.valley.W, spec.valley.W],
              "theta": [-spec.sim.theta_max, spec.sim.theta_max],
              "x_dot": spec.tile_bounds["x_dot"], "theta_dot": spec.tile_bounds["theta_dot"]}
    env = HillCartEnv(spec, check_pole=True, tile_bounds=bounds)
    coder = coder_from_bounds(bounds)
    return spec, env, coder, bounds


# ---------------------------------------------------------------- S-01
def test_S01_potential_is_zero_at_bottom_and_one_at_rim(parts):
    """Phi_E = 0 na dnu (0,0,0,0); na robu je natanko 1, ce je palica GLOBALNO navpicna,
    in 0.995 pri theta = 0 (palica vzdolz normale), ker je tam tezisce palice nekoliko nizje."""
    spec, env, _, _ = parts
    sh = EnergyPotential.for_env(env, c=1.0)
    env.reset((0.0, 0.0, 0.0, 0.0))
    assert sh.phi(env) == pytest.approx(0.0, abs=1e-12)
    # rob je terminalen, zato stanje nastavimo neposredno (brez preverjanja dogodkov)
    env._sim.state = (spec.valley.W, 0.0, 0.0, 0.0)  # psi = 0: globalno navpicna palica
    assert sh.phi(env) == pytest.approx(1.0, rel=1e-12)
    from hillcart.dynamics import from_observation
    env._sim.state = from_observation((spec.valley.W, 0.0, 0.0, 0.0), env.track)  # theta = 0
    assert 0.99 < sh.phi(env) < 1.0


def test_S01b_potential_scales_with_c(parts):
    _, env, _, _ = parts
    env.reset((0.3, 0.5, 0.0, 0.0))
    assert EnergyPotential.for_env(env, c=2.0).phi(env) == pytest.approx(
        2.0 * EnergyPotential.for_env(env, c=1.0).phi(env), rel=1e-12)


# ---------------------------------------------------------------- S-02
def test_S02_shaping_reward_formula():
    """F = gamma Phi(s') - Phi(s); ob terminaciji Phi(s') = 0."""
    f = EnergyPotential.shaping_reward
    assert f(0.3, 0.5, 0.999, False) == pytest.approx(0.999 * 0.5 - 0.3, abs=1e-15)
    assert f(0.3, 0.5, 0.999, True) == pytest.approx(-0.3, abs=1e-15)
    assert f(0.0, 0.0, 1.0, False) == 0.0


# ---------------------------------------------------------------- S-03
@pytest.mark.parametrize("s0", [(0.0, 0.0, 0.0, 0.0), (0.04, -0.03, 0.02, 0.01), (-0.05, 0.05, -0.05, 0.05)])
def test_S03_telescoping_at_gamma_one(parts, s0):
    """Pri gamma = 1 se vsota oblikovanih nagrad teleskopsko skrci:
    za koncano epizodo velja  sum F = Phi(terminal) - Phi(s0) = -Phi(s0).
    To je jedro izreka 1 (Ng et al. 1999): oblikovanje ne spremeni razvrstitve politik."""
    _, env, coder, _ = parts
    sh = EnergyPotential.for_env(env, c=1.0)
    agent = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=1.0), seed=3)
    env.reset(s0)
    phi0 = sh.phi(env)
    res = run_episode(env, agent, s0, greedy=False, learn=False, gamma=1.0, shaping=sh)
    outcome, _, _, total_r, _, _, _, _, total_rs = res
    if outcome is EnvOutcome.TIMEOUT:
        pytest.skip("prekinitev ni terminalno stanje; velja druga oblika (glej S-04)")
    assert total_rs - total_r == pytest.approx(-phi0, abs=1e-9)


# ---------------------------------------------------------------- S-04
def test_S04_truncation_keeps_bootstrap_form(parts):
    """Ob prekinitvi (timeout) se ne uporabi -Phi(s): vsota je Phi(s_zadnji) - Phi(s0)."""
    spec, _, coder, bounds = parts
    from hillcart.simulator import SimConfig
    short = SimConfig(dt=spec.sim.dt, n_substeps=spec.sim.n_substeps,
                      theta_max_deg=spec.sim.theta_max_deg, t_max=0.4)
    import dataclasses
    env = HillCartEnv(dataclasses.replace(spec, sim=short), check_pole=True, tile_bounds=bounds)
    sh = EnergyPotential.for_env(env, c=1.0)
    agent = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=0.0), seed=0)
    env.reset((0.0, 0.0, 0.0, 0.0))
    phi0 = sh.phi(env)
    res = run_episode(env, agent, (0.0, 0.0, 0.0, 0.0), greedy=True, learn=False, gamma=1.0, shaping=sh)
    outcome, _, _, total_r, _, _, _, _, total_rs = res
    assert outcome is EnvOutcome.TIMEOUT
    phi_last = sh.phi(env)
    assert total_rs - total_r == pytest.approx(phi_last - phi0, abs=1e-9)


# ---------------------------------------------------------------- S-05
def test_S05_no_shaping_is_regression_safe(parts):
    """shaping = None: oblikovana in neoblikovana nagrada sta enaki, ucenje nespremenjeno."""
    _, env, coder, _ = parts
    rng = np.random.default_rng(5)
    for _ in range(5):
        s0 = tuple(rng.uniform(-0.05, 0.05, size=4))
        agent = TrueOnlineSarsaLambda(coder, AgentConfig(), seed=5)
        res = run_episode(env, agent, s0, greedy=False, learn=True, gamma=0.999, shaping=None)
        assert res[8] == res[3]


# ---------------------------------------------------------------- S-06
def test_S06_falling_does_not_create_energy(parts):
    """Padec palice ne more povecati Phi_E: energija se brez dela motorja ne ustvarja."""
    _, env, coder, _ = parts
    sh = EnergyPotential.for_env(env, c=1.0)
    agent = TrueOnlineSarsaLambda(coder, AgentConfig(epsilon=0.0), seed=0)
    env.reset((0.0, 0.0, 0.0, 0.0))
    phis = [sh.phi(env)]
    for _ in range(200):  # akcija 1 = brez sile -> energija se mora ohranjati
        if env.outcome is not EnvOutcome.RUNNING:
            break
        env.step(1)
        phis.append(sh.phi(env))
    assert max(phis) - min(phis) < 1e-6, (min(phis), max(phis))


# ---------------------------------------------------------------- S-07
def test_S07_pumping_increases_potential(parts):
    """Crpanje energije (F = F_max sign(x_dot)) mora Phi_E povecati."""
    _, env, _, _ = parts
    sh = EnergyPotential.for_env(env, c=1.0)
    obs = env.reset((0.1, 0.0, 0.0, 0.0))
    phi0 = sh.phi(env)
    for _ in range(300):
        if env.outcome is not EnvOutcome.RUNNING:
            break
        obs, *_ = env.step(2 if obs[1] >= 0 else 0)
    assert sh.phi(env) > phi0 + 0.01
