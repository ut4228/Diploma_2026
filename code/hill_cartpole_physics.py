import numpy as np


def make_hill(amplitude=1.0, freq=1.0):
    A, k = amplitude, freq
    h = lambda x: -A * np.cos(k * x)
    hprime = lambda x: A * k * np.sin(k * x)
    hprime2 = lambda x: A * k**2 * np.cos(k * x)
    return h, hprime, hprime2


def critical_force(mass_cart, mass_pole, hill_amplitude, hill_freq, g=9.81):
    M = mass_cart + mass_pole
    slope_angle = np.arctan(hill_amplitude * hill_freq)
    return M * g * np.sin(slope_angle)


def accelerations(x, theta, xdot, thetadot, F, hprime, hprime2,
                   mass_cart, mass_pole, l, I, g=9.81):
    M = mass_cart + mass_pole
    hp = hprime(x)
    hpp = hprime2(x)
    c, s = np.cos(theta), np.sin(theta)

    A_mat = np.array([
        [M * (1 + hp**2),              mass_pole * l * (c - hp * s)],
        [mass_pole * l * (c - hp * s), mass_pole * l**2 + I],
    ])
    b_vec = np.array([
        F - M * hp * hpp * xdot**2 + mass_pole * l * (s + hp * c) * thetadot**2 - M * g * hp,
        mass_pole * g * l * s + mass_pole * l * hpp * xdot**2 * s,
    ])
    xacc, thetaacc = np.linalg.solve(A_mat, b_vec)
    return xacc, thetaacc


def rk4_step(state, F, hprime, hprime2, params, dt):
    """En integracijski korak (Runge-Kutta 4. reda).

    state = [x, theta, xdot, thetadot]
    """
    def deriv(s):
        x, theta, xdot, thetadot = s
        xacc, thetaacc = accelerations(x, theta, xdot, thetadot, F,
                                        hprime, hprime2, **params)
        return np.array([xdot, thetadot, xacc, thetaacc])

    k1 = deriv(state)
    k2 = deriv(state + dt / 2 * k1)
    k3 = deriv(state + dt / 2 * k2)
    k4 = deriv(state + dt * k3)
    return state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
