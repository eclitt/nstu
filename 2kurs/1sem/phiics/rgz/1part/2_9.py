import numpy as np
import matplotlib.pyplot as plt

L = 0.005
C = 0.2e-6
N = 3
omega0 = 1/np.sqrt(L*C)
ln10 = np.log(10)

beta_sq = (ln10**2 * omega0**2)/(16*np.pi**2*N**2 + ln10**2)
beta = np.sqrt(beta_sq)
R = 2*L*beta
omega = np.sqrt(omega0**2 - beta**2)
T = 2*np.pi/omega
lam = beta*T

print(f"β = {beta:.1f} с⁻¹")
print(f"R = {R:.2f} Ом")
print(f"λ = βT = {lam:.3f}")

t = np.linspace(0, 3*T, 1000)
q_ampl = np.exp(-beta*t)
q = q_ampl*np.cos(omega*t)
W = q_ampl**2

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(t*1000, q, 'b-', label='q(t)')
ax1.plot(t*1000, q_ampl, 'r--', label='огибающая')
ax1.plot(t*1000, -q_ampl, 'r--')
ax1.set_xlabel('Время t, мс')
ax1.set_ylabel('Заряд q')
ax1.grid(True)
ax1.legend()

for n in range(4):
    tn = n*T
    ax1.axvline(x=tn*1000, color='gray', linestyle=':', alpha=0.5)

ax2.semilogy(t*1000, W, 'g-', label='Энергия W(t)')
ax2.set_xlabel('Время t, мс')
ax2.set_ylabel('Энергия (лог. шкала)')
ax2.grid(True, which='both')
ax2.legend()

for n in range(4):
    tn = n*T
    Wn = np.exp(-2*beta*tn)
    ax2.scatter(tn*1000, Wn, color='red', s=50)
    ax2.axvline(x=tn*1000, color='gray', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()