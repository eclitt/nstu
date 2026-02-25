import numpy as np
import matplotlib.pyplot as plt

# Параметры
lambda_m = 640e-9          # м
a = 2.0                    # м
r = 1.131e-3               # м (найденный радиус)

# Диапазон b от 0.5 до 5 м
b = np.linspace(0.5, 5, 1000)

# Число зон Френеля
m = (r**2 / lambda_m) * (1/a + 1/b)

# Интенсивность (нормированная)
I_norm = 4 * (np.sin(np.pi * m / 2))**2

# Построение
plt.figure(figsize=(10, 6))
plt.plot(b, I_norm, 'b-', linewidth=2)
plt.xlabel('Расстояние от диафрагмы до экрана b (м)')
plt.ylabel('Нормированная интенсивность в центре I / I0')
plt.title('Интенсивность в центре дифракционной картины\n(отверстие радиусом 1.13 мм, a = 2 м, λ = 640 нм)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=2, color='r', linestyle='--', label='b = 2 м (заданное)')
plt.legend()
plt.ylim(0, 4.2)
plt.show()