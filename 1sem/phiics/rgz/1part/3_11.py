import numpy as np
import matplotlib.pyplot as plt

# Данные из задачи
U = 220.0      # В
R = 2.0        # Ом
L = 0.16       # Гн
C = 64e-6      # Ф

# 1. Расчет для f = 200 Гц
f1 = 200.0
XL1 = 2*np.pi*f1*L
XC1 = 1/(2*np.pi*f1*C)
Z1 = np.sqrt(R**2 + (XL1 - XC1)**2)
I1 = U/Z1

print(f"При f = {f1} Гц:")
print(f"  X_L = {XL1:.2f} Ом")
print(f"  X_C = {XC1:.2f} Ом")
print(f"  Z = {Z1:.2f} Ом")
print(f"  I = {I1:.3f} А")

# 2. Резонанс
f0 = 1/(2*np.pi*np.sqrt(L*C))
XL0 = 2*np.pi*f0*L
XC0 = 1/(2*np.pi*f0*C)
I0 = U/R
UL0 = I0*XL0
UC0 = I0*XC0

print(f"\nРезонанс:")
print(f"  f₀ = {f0:.2f} Гц")
print(f"  X_L = X_C = {XL0:.2f} Ом")
print(f"  I₀ = {I0:.1f} А")
print(f"  U_L = U_C = {UL0:.1f} В = {UL0/1000:.1f} кВ")

# 3. Построение графиков
freqs = np.linspace(10, 400, 1000)  # Частоты от 10 до 400 Гц

# Реактивные сопротивления
XL = 2*np.pi*freqs*L
XC = 1/(2*np.pi*freqs*C)

# Полное сопротивление и ток
Z = np.sqrt(R**2 + (XL - XC)**2)
I = U/Z

# Напряжения на элементах
UR = I*R
UL = I*XL
UC = I*XC

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# График 1: Ток и сопротивление
ax1.plot(freqs, Z, 'r-', linewidth=2, label='Полное сопротивление Z')
ax1.plot(freqs, I, 'b-', linewidth=2, label='Ток I')
ax1.axvline(x=f0, color='green', linestyle='--', label=f'Резонанс f₀={f0:.1f} Гц')
ax1.scatter([f1], [Z1], color='red', s=100, zorder=5)
ax1.scatter([f1], [I1], color='blue', s=100, zorder=5)
ax1.set_xlabel('Частота f, Гц', fontsize=12)
ax1.set_ylabel('Z (Ом), I (А)', fontsize=12)
ax1.set_title('Зависимость полного сопротивления и тока от частоты', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_xlim(10, 400)

# График 2: Напряжения
ax2.plot(freqs, UR, 'g-', linewidth=2, label='U_R')
ax2.plot(freqs, UL, 'b-', linewidth=2, label='U_L')
ax2.plot(freqs, UC, 'r-', linewidth=2, label='U_C')
ax2.axvline(x=f0, color='green', linestyle='--', label=f'Резонанс f₀={f0:.1f} Гц')
ax2.axhline(y=U, color='black', linestyle=':', label='U_ист = 220 В')
ax2.scatter([f0], [UL0], color='blue', s=100, zorder=5)
ax2.scatter([f0], [UC0], color='red', s=100, zorder=5)
ax2.set_xlabel('Частота f, Гц', fontsize=12)
ax2.set_ylabel('Напряжение, В', fontsize=12)
ax2.set_title('Напряжения на элементах цепи', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xlim(10, 400)
ax2.set_ylim(0, 6000)

plt.tight_layout()
plt.show()

# 4. Векторная диаграмма для f = 200 Гц
fig2, ax3 = plt.subplots(figsize=(8, 8))

# Фазовый сдвиг
phi = np.arctan((XL1 - XC1)/R)

# Векторы
ax3.arrow(0, 0, I1*R, 0, head_width=0.5, head_length=1, fc='g', ec='g', label=f'U_R = {I1*R:.1f} В')
ax3.arrow(0, 0, 0, I1*XL1, head_width=0.5, head_length=1, fc='b', ec='b', label=f'U_L = {I1*XL1:.1f} В')
ax3.arrow(0, 0, 0, -I1*XC1, head_width=0.5, head_length=1, fc='r', ec='r', label=f'U_C = {I1*XC1:.1f} В')
ax3.arrow(0, 0, U*np.cos(phi), U*np.sin(phi), head_width=0.5, head_length=1, fc='k', ec='k', linewidth=2, label=f'U_ист = {U:.0f} В')

ax3.set_xlim(-50, 250)
ax3.set_ylim(-50, 250)
ax3.set_xlabel('Действительная ось', fontsize=12)
ax3.set_ylabel('Мнимая ось', fontsize=12)
ax3.set_title(f'Векторная диаграмма при f = {f1} Гц', fontsize=14)
ax3.grid(True, alpha=0.3)
ax3.legend()
ax3.set_aspect('equal')

plt.tight_layout()
plt.show()