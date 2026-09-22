# here we compare derived equations of motion(from derive.py) with Florian's equations (23-24) for the frictionless cart-pole system
import sympy as sp

g, m_c, m_p, l, F, theta, thetadot = sp.symbols('g m_c m_p l F theta thetadot', real=True)
xdd, thetadd = sp.symbols('xdd thetadd', real=True)

M = m_c + m_p
c, s = sp.cos(theta), sp.sin(theta)
I = sp.Rational(1,3)*m_p*l**2

# --- flat-ground equations (from sympy Euler-Lagrange derivation) ---
eq1 = sp.Eq(F, M*xdd + m_p*l*(thetadd*c - thetadot**2*s))
eq2 = sp.Eq(0, (m_p*l**2+I)*thetadd + m_p*l*c*xdd - m_p*g*l*s)

sol = sp.solve([eq1, eq2], [xdd, thetadd], dict=True)[0]
my_thetadd = sp.simplify(sol[thetadd])
my_xdd = sp.simplify(sol[xdd])

# --- Florian eq. 23-24 (frictionless), theta-double-dot solved explicitly ---
florian_thetadd = (g*s + c*((-F - m_p*l*thetadot**2*s)/M)) / (l*(sp.Rational(4,3) - m_p*c**2/M))
florian_thetadd = sp.simplify(florian_thetadd)

# Florian's xdd uses thetadd on RHS (eq 24) -> substitute florian_thetadd in
florian_xdd = (F + m_p*l*(thetadot**2*s - florian_thetadd*c)) / M
florian_xdd = sp.simplify(florian_xdd)

diff_theta = sp.simplify(my_thetadd - florian_thetadd)
diff_x = sp.simplify(my_xdd - florian_xdd)

print("Razlika v theta_ddot (mora biti 0):", diff_theta)
print("Razlika v x_ddot (mora biti 0):    ", diff_x)

# numeric cross-check with random values as extra confidence
import random
subs = {g:9.81, m_c:1.3, m_p:0.27, l:0.42, F:1.7, theta:0.35, thetadot:-0.9}
print()
print("Numericno, moj theta_ddot:    ", float(my_thetadd.subs(subs)))
print("Numericno, Florian theta_ddot:", float(florian_thetadd.subs(subs)))
print("Numericno, moj x_ddot:        ", float(my_xdd.subs(subs)))
print("Numericno, Florian x_ddot:    ", float(florian_xdd.subs(subs)))
