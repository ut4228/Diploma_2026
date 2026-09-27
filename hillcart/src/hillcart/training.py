"""Ucna zanka in vrednotenje za true online Sarsa(lambda).

Vrednotenje je locENO od ucenja: pozresna politika (epsilon = 0), brez posodobitev utezi,
brez porabe proracuna korakov, na fiksni mnozici zacetnih stanj.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .env import EnvOutcome
from .initial_states import sample_train_state


@dataclass
class EpisodeRecord:
    episode: int
    global_step: int
    s0: tuple
    outcome: str
    steps: int
    t_event: float
    reward_sum: float
    reward_sum_shaped: float
    discounted_return: float
    max_abs_theta: float
    oob_steps: int
    exit_side: str | None

    def as_row(self) -> dict:
        d = dict(self.__dict__)
        d["s0"] = None
        for i, k in enumerate(("x0", "xdot0", "theta0", "thetadot0")):
            d[k] = self.s0[i]
        del d["s0"]
        return d


def run_episode(env, agent, obs0, greedy: bool, learn: bool, gamma: float, track_reach: bool = False,
                shaping=None):
    """Ena epizoda. Vrne (izid, koraki, t_event, vsota nagrad, diskontiran donos, max|theta|, oob, stran)
    in z track_reach=True se doda se reach = max |x| / x_goal (kako blizu cilja je vozicek prisel).

    shaping: None ali EnergyPotential. Agent se uci na OBLIKOVANI nagradi R' = R + F, metrike pa
    se vedno racunajo na NEOBLIKOVANI nagradi R (vsota in diskontiran donos)."""
    obs = env.reset(obs0)
    if learn:
        agent.begin_episode()
    a = agent.act(obs, greedy=greedy)
    idx = agent.coder.features(obs, a)
    total_r, total_rs, disc, disc_k, max_theta = 0.0, 0.0, 0.0, 1.0, abs(obs[2])
    phi = shaping.phi(env) if shaping is not None else 0.0
    goal = env.track.x_goal
    reach = abs(obs[0]) / goal
    while True:
        obs2, r, term, trunc, info = env.step(a)
        if shaping is not None:
            phi_next = 0.0 if term else shaping.phi(env)
            r_learn = r + shaping.shaping_reward(phi, phi_next, gamma, term)
            phi = phi_next
        else:
            r_learn = r
        total_r += r
        total_rs += r_learn
        disc += disc_k * r
        disc_k *= gamma
        max_theta = max(max_theta, abs(obs2[2]))
        reach = max(reach, abs(obs2[0]) / goal)
        if term:
            idx2, a2 = None, None
        else:
            a2 = agent.act(obs2, greedy=greedy)
            idx2 = agent.coder.features(obs2, a2)
        if learn:
            agent.update(idx, r_learn, idx2, term)
        if term or trunc:
            if learn:
                agent.end_episode()
            out = (env.outcome, env.steps, env.t_event, total_r, disc, max_theta, env.oob_steps,
                   info["exit_side"], total_rs)
            return out + (reach,) if track_reach else out
        idx, a = idx2, a2


def evaluate(env, agent, states, gamma: float) -> dict:
    """Pozresno vrednotenje na fiksnih stanjih (brez ucenja). Utezi se ne spremenijo."""
    w_before = agent.w.copy()
    out = {"success": 0, "fail_pole": 0, "timeout": 0}
    sides = {"left": 0, "right": 0}
    t_succ, steps_succ, t_fail, max_thetas, oob, disc_sum = [], [], [], [], 0, 0.0
    timeout_reach = []
    for s0 in states:
        res = run_episode(env, agent, s0, greedy=True, learn=False, gamma=gamma, track_reach=True)
        outcome, steps, t_event, _, disc, max_theta, oob_steps, side, _rs, reach = res
        out[outcome.value] = out.get(outcome.value, 0) + 1
        if outcome is EnvOutcome.SUCCESS:
            t_succ.append(t_event)
            steps_succ.append(steps)
            if side in sides:
                sides[side] += 1
        elif outcome is EnvOutcome.FAIL_POLE:
            t_fail.append(t_event)
        elif outcome is EnvOutcome.TIMEOUT:
            timeout_reach.append(reach)
        max_thetas.append(max_theta)
        oob += oob_steps
        disc_sum += disc
    assert np.array_equal(w_before, agent.w), "vrednotenje ne sme spreminjati utezi"
    n = len(states)
    return {"n": n, "success_rate": out["success"] / n, "outcomes": out, "exit_sides": sides,
            "t_event_fail_median": float(np.median(t_fail)) if t_fail else None,
            "timeout_reach_median": float(np.median(timeout_reach)) if timeout_reach else None,
            "timeout_reach_max": float(np.max(timeout_reach)) if timeout_reach else None,
            "t_event_median": float(np.median(t_succ)) if t_succ else None,
            "t_event_max": float(np.max(t_succ)) if t_succ else None,
            "steps_median": float(np.median(steps_succ)) if steps_succ else None,
            "max_abs_theta_mean": float(np.mean(max_thetas)),
            "discounted_return_mean": disc_sum / n, "oob_steps_total": int(oob)}


def train(env, agent, *, total_steps: int, eval_every: int, eval_states_quick, eval_states_final,
          rng: np.random.Generator, run=None, gamma: float | None = None, progress=None,
          shaping=None) -> dict:
    """Ucna zanka. Vrne povzetek; epizode se vracajo kot seznam EpisodeRecord."""
    gamma = agent.cfg.gamma if gamma is None else gamma
    records: list[EpisodeRecord] = []
    evals: list[dict] = []
    step, ep, next_eval = 0, 0, eval_every
    first_success_step = None
    while step < total_steps:
        s0 = sample_train_state(rng, env.spec.train_noise)
        outcome, steps, t_event, total_r, disc, max_theta, oob_steps, exit_side, total_rs = run_episode(
            env, agent, s0, greedy=False, learn=True, gamma=gamma, shaping=shaping)
        step += steps
        records.append(EpisodeRecord(ep, step, s0, outcome.value, steps, t_event, total_r, total_rs, disc,
                                     max_theta, oob_steps, exit_side))
        if outcome is EnvOutcome.SUCCESS and first_success_step is None:
            first_success_step = step
        ep += 1
        if run is not None:
            run.scalar("train/episode_reward", total_r, step)
            run.scalar("train/episode_reward_shaped", total_rs, step)
            run.scalar("train/episode_steps", steps, step)
            run.scalar("train/max_abs_theta", max_theta, step)
        if step >= next_eval:
            ev = evaluate(env, agent, eval_states_quick, gamma)
            ev["step"] = step
            evals.append(ev)
            if run is not None:
                run.scalar("eval/success_rate", ev["success_rate"], step)
                run.scalar("eval/w_absmax", float(np.abs(agent.w).max()), step)
                run.scalar("eval/active_traces_max", agent.max_active, step)
            if progress is not None:
                progress(step, ev)
            next_eval += eval_every
    final = evaluate(env, agent, eval_states_final, gamma)
    final["step"] = step
    n_out = {}
    for r in records:
        n_out[r.outcome] = n_out.get(r.outcome, 0) + 1
    return {"episodes": ep, "steps": step, "first_success_step": first_success_step,
            "train_outcomes": n_out, "evals": evals, "final_eval": final,
            "w_absmax": float(np.abs(agent.w).max()), "max_active_traces": agent.max_active,
            "records": records,
            "train_oob_steps": int(sum(r.oob_steps for r in records)),
            "train_oob_fraction": float(sum(r.oob_steps for r in records) / max(step, 1))}
