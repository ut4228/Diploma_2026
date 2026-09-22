import numpy as np

def accelerations(x, theta, xdot, thetadot, F, hprime, hprime2,
                   m_c, m_p, l, I, g=9.81):
    M = m_c + m_p
    hp = hprime(x)
    hpp = hprime2(x)
    c, s = np.cos(theta), np.sin(theta)

    A = np.array([
        [M*(1+hp**2),              m_p*l*(c - hp*s)],
        [m_p*l*(c - hp*s),         m_p*l**2 + I]
    ])
    b = np.array([
        F - M*hp*hpp*xdot**2 + m_p*l*(s + hp*c)*thetadot**2 - M*g*hp,
        m_p*g*l*s + m_p*l*hpp*xdot**2*s
    ])
    xacc, thetaacc = np.linalg.solve(A, b)
    return xacc, thetaacc

def rk4_step(state, F, h, hprime, hprime2, params, dt):
    def deriv(s):
        x, theta, xdot, thetadot = s
        xacc, thetaacc = accelerations(x, theta, xdot, thetadot, F,
                                         hprime, hprime2, **params)
        return np.array([xdot, thetadot, xacc, thetaacc])
    k1 = deriv(state)
    k2 = deriv(state + dt/2*k1)
    k3 = deriv(state + dt/2*k2)
    k4 = deriv(state + dt*k3)
    return state + dt/6*(k1 + 2*k2 + 2*k3 + k4)

def total_energy(state, h, hprime, m_c, m_p, l, I, g=9.81):
    x, theta, xdot, thetadot = state
    hp = hprime(x)
    M = m_c + m_p
    c, s = np.cos(theta), np.sin(theta)
    T = 0.5*M*xdot**2*(1+hp**2) + m_p*l*xdot*thetadot*(c - hp*s) \
        + 0.5*(m_p*l**2 + I)*thetadot**2
    V = M*g*h(x) + m_p*g*l*c
    return T + V

params = dict(m_c=1.0, m_p=0.1, l=0.5, I=(1/3)*0.1*0.5**2)

A_hill, k_hill = 0.5, 1.0
h        = lambda x: A_hill*np.sin(k_hill*x)
hprime   = lambda x: A_hill*k_hill*np.cos(k_hill*x)
hprime2  = lambda x: -A_hill*k_hill**2*np.sin(k_hill*x)

dt = 0.001
steps = 5000

state = np.array([0.3, 0.15, 0.4, -0.2])
E0 = total_energy(state, h, hprime, **params)
energies = [E0]

for _ in range(steps):
    state = rk4_step(state, 0.0, h, hprime, hprime2, params, dt)
    energies.append(total_energy(state, h, hprime, **params))

energies = np.array(energies)
print(f"Zacetna energija:   {E0:.10f}")
print(f"Koncna energija:    {energies[-1]:.10f}")
print(f"Max odstopanje:     {np.max(np.abs(energies - E0)):.2e}")
print(f"Relativno odstop.:  {np.max(np.abs(energies - E0))/abs(E0):.2e}")
