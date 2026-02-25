import numpy as np
import matplotlib.pyplot as plt

# Параметры
lambda_nm = 589  # нм
I0 = 1           # нормируем интенсивность

# Разность хода от -2λ до 2λ
delta_x_lambda = np.linspace(-2, 2, 1000)
delta_x = delta_x_lambda * lambda_nm  # в нм

# Интенсивность
delta_phi = 2 * np.pi * delta_x_lambda  # разность фаз
I = 2 * I0 * (1 + np.cos(delta_phi))

# Построение графика
plt.figure(figsize=(10, 5))
plt.plot(delta_x_lambda, I, 'b-', linewidth=2)
plt.xlabel(r'Разность хода $\Delta x$ (в длинах волн $\lambda$)')
plt.ylabel(r'Интенсивность $I$ (отн. ед.)')
plt.title('Распределение интенсивности в интерференционной картине')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axhline(y=0, color='k', linewidth=0.5)
plt.axvline(x=0, color='k', linewidth=0.5)
plt.xlim(-2, 2)
plt.ylim(0, 4)
plt.show()