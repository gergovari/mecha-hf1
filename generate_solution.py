import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# Rendszer paraméterek definíciója
un = 30.0
R_val = 19.1
L_val = 1.9e-3
km_val = 40.1e-3
Jr_val = 12.6e-7
N_val = 181.0
B_val = 2.0e-6
Jt_val = 200e-7
dt_val = 0.040

s = sp.Symbol('s')

# Szimbolikus változók deklarálása
R, L, km, Jr, N, B, Jt = sp.symbols('R L k_m J_r N B J_t')

# Impedanciák felírása
Z1 = R + L*s
Z2 = 1 / (Jr*s + B)
Zt = 1 / (Jt*s)

# Impedanciák redukálása a terhelés oldalára
Z12 = Z1 / (km**2 * N**2)
Z22 = Z2 / N**2
Ze = sp.simplify((Z22 * Zt) / (Z22 + Zt))

# Átviteli függvények kiszámítása
W1 = sp.simplify(Ze / ((Z12 + Ze) * km * N))
W2 = sp.simplify(1 / ((Z12 + Ze) * km**2 * N**2)) 
W3 = sp.simplify(-Ze * Z12 / (Ze + Z12))
W4 = sp.simplify(Ze / ((Ze + Z12) * km * N))

# Behelyettesítés a megadott numerikus értékekkel
subs_dict = {R: R_val, L: L_val, km: km_val, Jr: Jr_val, N: N_val, B: B_val, Jt: Jt_val}
W1_val = sp.simplify(W1.subs(subs_dict))
W2_val = sp.simplify(W2.subs(subs_dict))
W3_val = sp.simplify(W3.subs(subs_dict))
W4_val = sp.simplify(W4.subs(subs_dict))

print("W1(s) =")
print(sp.latex(W1_val))
print("W2(s) =")
print(sp.latex(W2_val))
print("W3(s) =")
print(sp.latex(W3_val))
print("W4(s) =")
print(sp.latex(W4_val))

# Időállandók meghatározása a nevező gyökeiből
num, den = sp.fraction(W1_val)
coeffs = sp.Poly(den, s).all_coeffs()
roots = np.roots([float(c) for c in coeffs])
T = [-1/rt for rt in roots]
print("Idoallandok:", T)

# Állandósult értékek (végérték tétel s -> 0)
W1_0 = W1_val.subs(s, 0)
W2_0 = W2_val.subs(s, 0)
W3_0 = W3_val.subs(s, 0)
W4_0 = W4_val.subs(s, 0)

print(f"lim W1(s) = {W1_0}")
print(f"lim W2(s) = {W2_0}")
print(f"lim W3(s) = {W3_0}")
print(f"lim W4(s) = {W4_0}")

u_b = sp.Symbol('u_{b}')
Mt = sp.Symbol('M_t')

# Maximális terhelőnyomaték számítása a megengedett legnagyobb áram (0.47A) alapján
i_max = 0.47
Mt_max = (i_max - float(W2_0) * un) / float(W4_0)
print(f"Mt_max = {Mt_max}")

# W1 átmeneti függvénye
Y1 = sp.apart(W1_val / s)
print("Ugrasvalasz Y1(s):")
print(sp.latex(Y1))
t_sym = sp.Symbol('t')
y1_t = sp.inverse_laplace_transform(W1_val / s, s, t_sym)
print("Ugrasvalasz y1(t):")
print(sp.latex(y1_t))

# Folytonos idejű állapottér (State Space) mátrixok
A = np.array([[-R_val/L_val, -km_val/L_val], [km_val/Jr_val, -B_val/Jr_val]])
B_mat = np.array([[1/L_val], [0]])
C = np.array([[0, 1]])
D = np.array([[0]])

print("A =", A)
print("B_mat =", B_mat)

# Előretartó (Forward) Euler diszkretizáció mátrixai
I = np.eye(2)
Ade = I + A * dt_val
Bde = B_mat * dt_val
Cde = C
Dde = D
print("Ade =", Ade)
print("Bde =", Bde)

# Hátratartó (Backward) Euler diszkretizáció mátrixai
Adh = np.linalg.inv(I - A * dt_val)
Bdh = Adh @ B_mat * dt_val
Cdh = C @ Adh
Ddh = D + C @ Bdh
print("Adh =", Adh)
print("Bdh =", Bdh)

# Szimuláció előkészítése
n_steps = 100
t = np.arange(0, n_steps*dt_val, dt_val)
u = np.ones(n_steps) * un

# Előretartó Euler Szimulációs ciklus
x_de = np.zeros((2, n_steps))
y_de = np.zeros(n_steps)
for k in range(n_steps-1):
    x_de[:, k+1] = Ade @ x_de[:, k] + Bde[:, 0] * u[k]
    y_de[k] = (Cde @ x_de[:, k])[0]
y_de[-1] = (Cde @ x_de[:, -1])[0]

# Hátratartó Euler Szimulációs ciklus
x_dh = np.zeros((2, n_steps))
y_dh = np.zeros(n_steps)
for k in range(n_steps-1):
    x_dh[:, k+1] = Adh @ x_dh[:, k] + Bdh[:, 0] * u[k+1]
    y_dh[k] = (Cdh @ x_dh[:, k])[0]
y_dh[-1] = (Cdh @ x_dh[:, -1])[0]

# Folytonos idejű referencia szimuláció (scipy)
import scipy.signal as signal
sys_c = signal.StateSpace(A, B_mat, C, D)
t_c, y_c = signal.step(sys_c, T=np.linspace(0, (n_steps-1)*dt_val, 1000))
y_c = y_c * un

plt.figure()
plt.plot(t_c, y_c, 'k-', lw=1, alpha=0.5, label="Folytonos referencia")
plt.step(t, y_de, 'r-', where='post', lw=2, label="Előretartó (Forward) Euler")
plt.xlabel("t [s]")
plt.ylabel(r"Szögsebesség $\omega$ [rad/s]")
plt.title("Előretartó Euler ugrásválasz")
plt.grid(True)
plt.legend()
plt.savefig("res/imgs/fig_EulerElore.png")

plt.figure()
plt.plot(t_c, y_c, 'k-', lw=1, alpha=0.5, label="Folytonos referencia")
plt.plot(t, y_dh, 'b-', lw=2, drawstyle='steps-post', label="Hátratartó (Backward) Euler")
plt.xlabel("t [s]")
plt.ylabel(r"Szögsebesség $\omega$ [rad/s]")
plt.title("Hátratartó Euler ugrásválasz")
plt.grid(True)
plt.legend()
plt.savefig("res/imgs/fig_EulerHatra.png")

# Szorgalmi feladat: Konstans terhelőnyomaték szimulációja
B_mat_full = np.array([[1/L_val, 0], [0, -1/Jr_val]])
sys_full = signal.StateSpace(A, B_mat_full, C, np.zeros((1,2)))
t_full = np.linspace(0, 0.5, 1000)
# A gerjesztés mátrixa: első oszlop feszültség, második oszlop lassító nyomaték (0.01 Nm)
u_full = np.vstack([np.ones(len(t_full)) * un, np.ones(len(t_full)) * 0.01]).T
t_sim, y_sim, x_sim = signal.lsim(sys_full, U=u_full, T=t_full)

plt.figure()
plt.plot(t_sim, y_sim, 'g-', lw=2)
plt.xlabel("t [s]")
plt.ylabel(r"Szögsebesség $\omega$ [rad/s]")
plt.title(r"Ugrásválasz terhelőnyomaték ($M_t = 0.01$ Nm) mellett")
plt.grid(True)
plt.savefig("res/imgs/fig_Szorgalmi.png")

# Hajtáslánc ugrásválasza 1.f feladatra (csak a tárcsa figyelembevételével)
t_w1, y_w1 = signal.step(signal.TransferFunction(
    [float(c) for c in sp.Poly(sp.fraction(W1_val)[0], s).all_coeffs()],
    [float(c) for c in sp.Poly(sp.fraction(W1_val)[1], s).all_coeffs()]
), T=t_full)
y_w1 = y_w1 * un
plt.figure()
plt.plot(t_w1, y_w1, 'purple', lw=2)
plt.xlabel("t [s]")
plt.ylabel(r"Tárcsa Szögsebesség $\omega$ [rad/s]")
plt.title(r"A hajtáslánc ugrásválasza (1.f és 2.d feladat)")
plt.grid(True)
plt.savefig("res/imgs/fig_HajtaslancUgras.png")
