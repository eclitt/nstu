import numpy as np
import matplotlib.pyplot as plt

# Константы
C1 = 3.742e8      # Вт·нм⁴/м²
C2 = 1.439e7      # нм·К
T = 2000          # К

# Диапазон длин волн (от 200 до 5000 нм)
lambda_nm = np.linspace(200, 5000, 1000)

# Формула Планка
r_lambda = C1 / (lambda_nm**5 * (np.exp(C2 / (lambda_nm * T)) - 1))

# Находим максимум
max_idx = np.argmax(r_lambda)
lambda_max = lambda_nm[max_idx]
r_max = r_lambda[max_idx]

# Построение
plt.figure(figsize=(10, 6))
plt.plot(lambda_nm, r_lambda, 'b-', linewidth=2)
plt.plot(lambda_max, r_max, 'ro', label=f'Максимум при λ = {lambda_max:.0f} нм')
plt.xlabel('Длина волны λ (нм)')
plt.ylabel('Спектральная плотность r(λ,T) (Вт/(м²·нм))')
plt.title('Спектр излучения абсолютно чёрного тела при T = 2000 K')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()
plt.xlim(200, 5000)
plt.ylim(0, r_max * 1.1)
plt.show()

print(f"λ_max = {lambda_max:.1f} нм")
print(f"r_max = {r_max:.3e} Вт/(м²·нм)")