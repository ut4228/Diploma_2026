import sympy as sp

t = sp.symbols('t')
g, m_c, m_p, l, I, F = sp.symbols('g m_c m_p l I F', positive=True)

x = sp.Function('x')(t)
theta = sp.Function('theta')(t)

# arbitrary track height profile h(x)
h = sp.Function('h')(x)

xdot = sp.diff(x, t)
thetadot = sp.diff(theta, t)

M = m_c + m_p

hprime = sp.diff(h, x)          # h'(x)
# Cart position on the track
Xc, Yc = x, h
# Pole COM position (pivot at cart, rod half-length l)
Xp = x + l*sp.sin(theta)
Yp = h + l*sp.cos(theta)

Xc_dot = sp.diff(Xc, t)
Yc_dot = sp.diff(Yc, t)
Xp_dot = sp.diff(Xp, t)
Yp_dot = sp.diff(Yp, t)

T = sp.Rational(1,2)*m_c*(Xc_dot**2 + Yc_dot**2) \
    + sp.Rational(1,2)*m_p*(Xp_dot**2 + Yp_dot**2) \
    + sp.Rational(1,2)*I*thetadot**2

V = m_c*g*h + m_p*g*Yp

L = sp.simplify(T - V)

def euler_lagrange(L, q):
    qdot = sp.diff(q, t)
    dL_dqdot = sp.diff(L, qdot)
    ddt = sp.diff(dL_dqdot, t)
    dL_dq = sp.diff(L, q)
    return sp.simplify(ddt - dL_dq)

eq_x = sp.simplify(euler_lagrange(L, x) - F)
eq_theta = sp.simplify(euler_lagrange(L, theta) - 0)

print("=== Equation for x (=F) ===")
print(sp.simplify(eq_x))
print()
print("=== Equation for theta (=0) ===")
print(sp.simplify(eq_theta))

print()
print("=== Sanity check: flat ground h(x)=0 ===")
subs_flat = {sp.Derivative(h, x): 0, sp.diff(h, (x,2)): 0, h: 0}
eq_x_flat = eq_x.subs(subs_flat)
eq_theta_flat = eq_theta.subs(subs_flat)
print("x-eq flat:", sp.simplify(eq_x_flat))
print("theta-eq flat:", sp.simplify(eq_theta_flat))
