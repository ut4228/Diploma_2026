"""Fizikalni validacijski testi T1-T12.

Vsak test navaja, kaj preverja. Tolerance so izbrane z rezervo nad izmerjeno napako
(izmerjene vrednosti zapiše scripts/run_validation.py).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from hillcart.dynamics import (PoleCartParams, accelerations, energy, from_observation, mass_matrix_det,
                               normal_force, rhs, to_observation)
from hillcart.experiment_grid import unique_valleys
from hillcart.integrators import integrate, rk4_step
from hillcart.simulator import HillCartSimulator, Outcome, SimConfig
from hillcart.tracks import FlatTrack, LinearTrack, PowerValley

PRM = PoleCartParams()
BASE = PowerValley(H=1.0, tan_phi_max=0.5, p=2.0)
H_INT = 0.005  # RK4 podkorak (dt = 0.02 / 4)


def run_raw(track, prm, obs0, forces, dt=0.02, n_sub=4, method="rk4"):
    """Integracija brez dogodkov; forces = sila za vsak korak odločanja."""
    y = from_observation(obs0, track)
    out = [y]
    for F in forces:
        f = lambda s, F=F: rhs(s, F, track, prm)
        y = integrate(f, y, dt / n_sub, n_sub, method)[-1]
        out.append(y)
    return out


def random_states(rng, n, x_range):
    return [(rng.uniform(*x_range), rng.uniform(-3, 3), rng.uniform(-1.0, 1.0), rng.uniform(-5, 5)) for _ in range(n)]


# ---------------------------------------------------------------- T1
def _symbolic_accels(p: int, H: float, W: float, prm: PoleCartParams):
    sp = pytest.importorskip("sympy")
    t = sp.symbols("t")
    x, psi = sp.Function("x")(t), sp.Function("psi")(t)
    h = H * (x / W) ** p  # velja za x > 0 (brez abs)
    mc, mp, l, g, F = prm.m_c, prm.m_p, prm.l, prm.g, sp.Symbol("F")
    rc = sp.Matrix([x, h])
    rp = sp.Matrix([x + l * sp.sin(psi), h + l * sp.cos(psi)])
    vc, vp = rc.diff(t), rp.diff(t)
    T = sp.Rational(1, 2) * mc * vc.dot(vc) + sp.Rational(1, 2) * mp * vp.dot(vp) \
        + sp.Rational(1, 2) * (mp * l**2 / 3) * psi.diff(t) ** 2
    V = mc * g * h + mp * g * (h + l * sp.cos(psi))
    L = T - V
    EL = lambda q: sp.diff(sp.diff(L, q.diff(t)), t) - sp.diff(L, q)
    X, Vv, P, O, XDD, PDD = sp.symbols("X V P O XDD PDD")
    sub2 = {x.diff(t, 2): XDD, psi.diff(t, 2): PDD}
    sub1 = {x.diff(t): Vv, psi.diff(t): O}
    sub0 = {x: X, psi: P}
    e1 = (EL(x) - F).subs(sub2).subs(sub1).subs(sub0)  # Q_x = F (vodoravna sila)
    e2 = EL(psi).subs(sub2).subs(sub1).subs(sub0)  # Q_psi = 0
    sol = sp.solve([e1, e2], [XDD, PDD], dict=True)[0]
    return sp.lambdify((X, Vv, P, O, F), (sol[XDD], sol[PDD]), "math")


@pytest.mark.parametrize("p", [2, 3])
def test_T01_lagrange_symbolic_matches_implementation(p):
    """T1: neodvisna simbolna izpeljava (Lagrange 2. vrste) == implementacija E1/E2."""
    valley = PowerValley(H=1.0, tan_phi_max=0.5, p=float(p))
    f_sym = _symbolic_accels(p, valley.H, valley.W, PRM)
    rng = np.random.default_rng(1)
    for x, v, psi, om in random_states(rng, 40, (0.05 * valley.W, 0.95 * valley.W)):
        F = rng.uniform(-10, 10)
        assert accelerations(x, v, psi, om, F, valley, PRM) == pytest.approx(f_sym(x, v, psi, om, F), rel=1e-9, abs=1e-9)


def test_T01b_newton_euler_residuals():
    """T1b: Newton-Euler (tangentna komponenta za cel sistem + vrtenje palice okoli zgiba) == 0."""
    rng = np.random.default_rng(2)
    for x, v, psi, om in random_states(rng, 40, (-BASE.W, BASE.W)):
        F = rng.uniform(-10, 10)
        xdd, psidd = accelerations(x, v, psi, om, F, BASE, PRM)
        h1, h2 = BASE.dh(x), BASE.d2h(x)
        s, c = math.sin(psi), math.cos(psi)
        acx, acy = xdd, h1 * xdd + h2 * v * v
        apx = acx + PRM.l * (psidd * c - om * om * s)
        apy = acy - PRM.l * (psidd * s + om * om * c)
        tang = (PRM.m_c * acx + PRM.m_p * apx) + h1 * (PRM.m_c * acy + PRM.m_p * apy) - (F - PRM.M * PRM.g * h1)
        rot = (4 / 3) * PRM.l * psidd + c * acx - s * (PRM.g + acy)  # Florian (2007) en. 13, posplošeno
        assert abs(tang) < 1e-9 and abs(rot) < 1e-9


# ---------------------------------------------------------------- T2
def florian_frictionless(theta, theta_dot, F, prm):
    """Florian (2007), en. (23) in (24), brez trenja."""
    M = prm.M
    s, c = math.sin(theta), math.cos(theta)
    thdd = (prm.g * s + c * (-F - prm.m_p * prm.l * theta_dot**2 * s) / M) / (prm.l * (4 / 3 - prm.m_p * c * c / M))
    xdd = (F + prm.m_p * prm.l * (theta_dot**2 * s - thdd * c)) / M
    return xdd, thdd


def test_T02_flat_track_equals_florian():
    """T2: pri h = 0 se E1/E2 reducirata natanko v Florianovi en. (23)-(24)."""
    flat = FlatTrack()
    rng = np.random.default_rng(3)
    for x, v, psi, om in random_states(rng, 100, (-5, 5)):
        F = rng.uniform(-10, 10)
        assert accelerations(x, v, psi, om, F, flat, PRM) == pytest.approx(florian_frictionless(psi, om, F, PRM),
                                                                           rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------- T3
def test_T03_flat_normal_force_equals_florian_eq20():
    """T3: pri h = 0 je normalna sila enaka Florianovi en. (20)."""
    flat = FlatTrack()
    rng = np.random.default_rng(4)
    for x, v, psi, om in random_states(rng, 100, (-5, 5)):
        F = rng.uniform(-10, 10)
        _, thdd = florian_frictionless(psi, om, F, PRM)
        Nc = PRM.M * PRM.g - PRM.m_p * PRM.l * (thdd * math.sin(psi) + om**2 * math.cos(psi))
        assert normal_force((x, v, psi, om), F, flat, PRM) == pytest.approx(Nc, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------- T4
def energy_drift(method, h, T=2.0, track=BASE):
    y0 = from_observation((0.5 * track.W, 0.0, 0.1, 0.0), track)
    f = lambda y: rhs(y, 0.0, track, PRM)
    ys = integrate(f, y0, h, int(round(T / h)), method)
    E = np.array([energy(y, track, PRM) for y in ys])
    return float(np.max(np.abs(E - E[0])) / (PRM.M * PRM.g * track.H))


def test_T04_energy_conservation_rk4():
    """T4: pri F = 0 se energija ohranja; RK4 je 4. reda, Euler ustvarja energijo."""
    d1 = energy_drift("rk4", H_INT)
    d2 = energy_drift("rk4", H_INT / 2)
    assert d1 < 1e-7, d1
    assert d1 / d2 > 10, (d1, d2)  # pričakovano ~16 za metodo 4. reda
    de = energy_drift("euler", H_INT)
    assert de > 1000 * d1, (de, d1)


# ---------------------------------------------------------------- T5
def work_balance_error(force_model="horizontal"):
    """Integrira [stanje, delo] skupaj; vrne max |E - E0 - W| / (M g H)."""
    track = BASE
    y0 = from_observation((0.5 * track.W, 0.0, 0.02, 0.0), track) + (0.0,)
    forces = [3.0 * (1 if math.sin(2 * math.pi * k * 0.02 / 1.3) >= 0 else -1) for k in range(150)]
    y, err = y0, 0.0
    E0 = energy(y0[:4], track, PRM)
    for F in forces:
        def f(s, F=F):
            x, v, psi, om, _ = s
            if force_model == "horizontal":
                F_eff = F  # Q_x = F
            else:  # negativna kontrola: tangentna sila velikosti F -> Q_x = F*sqrt(1+h'^2)
                F_eff = F * math.sqrt(1 + track.dh(x) ** 2)
            xdd, psidd = accelerations(x, v, psi, om, F_eff, track, PRM)
            return (v, xdd, om, psidd, F * v)  # dW/dt = F * x_dot (moč vodoravne sile)
        for _ in range(4):
            y = rk4_step(f, y, H_INT)
        err = max(err, abs(energy(y[:4], track, PRM) - E0 - y[4]) / (PRM.M * PRM.g * track.H))
    return err


def test_T05_work_energy_horizontal_force():
    """T5: dE/dt = F * x_dot. Negativna kontrola: tangentni model sile bilance NE izpolni."""
    e_h = work_balance_error("horizontal")
    e_t = work_balance_error("tangential")
    assert e_h < 1e-7, e_h
    assert e_t > 1000 * e_h and e_t > 1e-4, (e_t, e_h)


# ---------------------------------------------------------------- T6
def test_T06_small_oscillation_period():
    """T6: m_p -> 0, parabola, majhna amplituda: perioda = 2 pi W / sqrt(2 g H)."""
    prm = PoleCartParams(m_c=1.1, m_p=1e-9)
    track = BASE
    T_exp = 2 * math.pi * track.W / math.sqrt(2 * prm.g * track.H)
    f = lambda s: rhs(s, 0.0, track, prm)
    prev = from_observation((0.02 * track.W, 0.0, 0.0, 0.0), track)
    t, crossings = 0.0, []
    while len(crossings) < 4:
        y = rk4_step(f, prev, 0.001)
        t += 0.001
        if prev[0] < 0 <= y[0]:  # prehod skozi 0 navzgor, linearna interpolacija
            crossings.append(t - 0.001 * y[0] / (y[0] - prev[0]))
        prev = y
    T_num = float(np.mean(np.diff(crossings)))
    assert T_num == pytest.approx(T_exp, rel=1e-3), (T_num, T_exp)


# ---------------------------------------------------------------- T7
@pytest.mark.parametrize("p", [2.0, 3.0])
def test_T07_mirror_symmetry(p):
    """T7: (x, x', theta, theta', F) -> (-x, -x', -theta, -theta', -F) da zrcalno trajektorijo."""
    track = PowerValley(H=1.0, tan_phi_max=0.5, p=p)
    rng = np.random.default_rng(7)
    forces = [float(F) for F in rng.choice([-4.0, 0.0, 4.0], size=100)]
    obs0 = (0.4, 0.3, 0.05, -0.2)
    a = run_raw(track, PRM, obs0, forces)
    b = run_raw(track, PRM, tuple(-v for v in obs0), [-F for F in forces])
    for ya, yb in zip(a, b):
        assert max(abs(u + w) for u, w in zip(ya, yb)) < 1e-9


# ---------------------------------------------------------------- T8
def test_T08_inclined_plane_normal_equilibrium():
    """T8: raven klanec, F = 0, palica vzdolž normale (theta = 0) ostane vzdolž normale;
    voziček pospešuje z x'' = -g sin(phi) cos(phi)."""
    tp = 0.3
    track = LinearTrack(tan_phi=tp)
    phi = math.atan(tp)
    y0 = from_observation((0.0, 0.0, 0.0, 0.0), track)
    xdd, _ = accelerations(*y0, 0.0, track, PRM)
    assert xdd == pytest.approx(-PRM.g * math.sin(phi) * math.cos(phi), rel=1e-12)
    for y in run_raw(track, PRM, (0.0, 0.0, 0.0, 0.0), [0.0] * 25):  # 0.5 s
        assert abs(to_observation(y, track)[2]) < 1e-8


# ---------------------------------------------------------------- T9
@pytest.mark.parametrize("p", [2.0, 3.0, 4.0])
def test_T09_bottom_equilibrium(p):
    """T9: dno kotanje, mirovanje, theta = 0, F = 0 je ravnovesje (vsi odvodi = 0)."""
    track = PowerValley(H=1.0, tan_phi_max=0.5, p=p)
    assert rhs((0.0, 0.0, 0.0, 0.0), 0.0, track, PRM) == (0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------- T10
def test_T10_substep_convergence():
    """T10: rezultat (stanje, izid, čas dogodka) se ne spremeni pri finejšem podkoraku."""
    forces = [4.0 if (k // 30) % 2 == 0 else -4.0 for k in range(100)]
    ref = run_raw(BASE, PRM, (0.1, 0.0, 0.0, 0.0), forces, n_sub=64)[-1]
    for n_sub in (4, 8):
        y = run_raw(BASE, PRM, (0.1, 0.0, 0.0, 0.0), forces, n_sub=n_sub)[-1]
        assert max(abs(a - b) for a, b in zip(y, ref)) < 1e-6
    results = []
    for n_sub in (4, 8, 16):
        sim = HillCartSimulator(BASE, F_max=4.0, cfg=SimConfig(n_substeps=n_sub))
        sim.reset((0.1, 0.0, 0.0, 0.0))
        k = 0
        while sim.outcome is Outcome.RUNNING:
            sim.step(forces[k % len(forces)])
            k += 1
        results.append((sim.outcome, sim.t_event))
    assert len({r[0] for r in results}) == 1
    assert max(r[1] for r in results) - min(r[1] for r in results) <= 0.02 + 1e-12


# ---------------------------------------------------------------- T11
def test_T11_mass_matrix_positive_definite():
    """T11: D = l [4/3 M a - m_p B^2] >= l a (4/3 M - m_p) > 0 za vse kotanje v načrtu."""
    rng = np.random.default_rng(11)
    for track in unique_valleys():
        for _ in range(500):
            x = rng.uniform(-1.2 * track.W, 1.2 * track.W)
            psi = rng.uniform(-math.pi, math.pi)
            D = mass_matrix_det(x, psi, track, PRM)
            bound = PRM.l * (1 + track.dh(x) ** 2) * ((4 / 3) * PRM.M - PRM.m_p)
            assert D >= bound - 1e-12 and bound > 0


# ---------------------------------------------------------------- T12
def test_T12_constant_force_relative_equilibrium():
    """T12: raven tir, konstantna F: palica pri theta_eq = atan(F/(M g)) ostane nagnjena
    (relativno ravnovesje), voziček pospešuje z F/M."""
    flat = FlatTrack()
    F = 2.0
    th = math.atan(F / (PRM.M * PRM.g))
    xdd, psidd = accelerations(0.0, 0.0, th, 0.0, F, flat, PRM)
    assert xdd == pytest.approx(F / PRM.M, rel=1e-12)
    assert abs(psidd) < 1e-12
    for y in run_raw(flat, PRM, (0.0, 0.0, th, 0.0), [F] * 25):
        assert abs(y[2] - th) < 1e-8


# ---------------------------------------------------------------- pretvorba koordinat
def test_observation_roundtrip():
    rng = np.random.default_rng(12)
    for track in unique_valleys():
        for s in random_states(rng, 50, (-track.W, track.W)):
            back = from_observation(to_observation(s, track), track)
            assert back == pytest.approx(s, abs=1e-12)
