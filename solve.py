import sympy as sp
import numpy as np

# Parameters
un = 30.0
R_val = 19.1
L_val = 1.9e-3
km_val = 40.1e-3
Jr_val = 12.6e-7
N_val = 181.0
B_val = 2.0e-6
Jt_val = 200e-7
dt = 0.040

s = sp.Symbol('s')

# Symbols to calculate symbolic first
R, L, km, Jr, N, B, Jt = sp.symbols('R L k_m J_r N B J_t')

Z1 = R + L*s
Z2 = 1 / (Jr*s + B)
Zt = 1 / (Jt*s)

Z12 = Z1 / (km**2 * N**2)
Z22 = Z2 / N**2
Ze = sp.simplify((Z22 * Zt) / (Z22 + Zt))

W1 = sp.simplify(Ze / ((Z12 + Ze) * km * N))
W2 = sp.simplify(1 / ((Z12 + Ze) * km**2 * N**2)) 
W3 = sp.simplify(-Ze * Z12 / (Ze + Z12))
W4 = sp.simplify(Ze / ((Ze + Z12) * km * N))

# Substitute values
subs_dict = {R: R_val, L: L_val, km: km_val, Jr: Jr_val, N: N_val, B: B_val, Jt: Jt_val}
W1_val = sp.simplify(W1.subs(subs_dict))
W2_val = sp.simplify(W2.subs(subs_dict))
W3_val = sp.simplify(W3.subs(subs_dict))
W4_val = sp.simplify(W4.subs(subs_dict))

print("W1(s) =", W1_val)
print("W2(s) =", W2_val)
print("W3(s) =", W3_val)
print("W4(s) =", W4_val)

# Time constants of W1
num, den = sp.fraction(W1_val)
coeffs = sp.Poly(den, s).all_coeffs()
roots = np.roots([float(c) for c in coeffs])
print("Roots of den:", roots)
T = [-1/rt for rt in roots]
print("Time constants:", T)
