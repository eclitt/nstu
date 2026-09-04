import numpy as np
import matplotlib.pyplot as plt

# Константы
epsilon0 = 8.854187817e-12  # Ф/м
mu0 = 4 * np.pi * 1e-7      # Гн/м
c = 299792458               # м/с

# Данные из задачи
t = 60.0                    # с
r = 0.05                    # м
E0 = 0.001                  # В/м

# 1. Площадь площадки
S = np.pi * r**2

# 2. Интенсивность (среднее значение вектора Пойнтинга)
I = 0.5 * epsilon0 * c * E0**2

# 3. Энергия за время t
W = I * S * t

print("РАСЧЕТ ЭНЕРГИИ ЭЛЕКТРОМАГНИТНОЙ ВОЛНЫ В ВАКУУМЕ")
print("="*50)
print(f"Площадь площадки: S = π×r² = {S:.6f} м²")
print(f"Амплитуда электрического поля: E₀ = {E0} В/м")
print(f"Интенсивность: I = ½ε₀cE₀² = {I:.4e} Вт/м²")
print(f"Энергия за {t} с: W = I·S·t = {W:.4e} Дж")
print(f"В наноджоулях: {W/1e-9:.3f} нДж")

# 4. Связь с магнитным полем
B0 = E0 / c
print(f"\nАмплитуда магнитного поля: B₀ = E₀/c = {B0:.4e} Тл")
print(f"Проверка: I = ½(c/μ₀)B₀² = {0.5*(c/mu0)*B0**2:.4e} Вт/м²")

# 5. График зависимости энергии от амплитуды E₀
E0_range = np.logspace(-4, 1, 100)  # От 0.1 мВ/м до 10 В/м
W_range = 0.5 * epsilon0 * c * (E0_range**2) * S * t

plt.figure(figsize=(10, 6))
plt.loglog(E0_range, W_range, 'b-', linewidth=2)
plt.axvline(x=E0, color='r', linestyle='--', label=f'Заданная E₀ = {E0} В/м')
plt.axhline(y=W, color='r', linestyle='--', label=f'W = {W:.2e} Дж')
plt.xlabel('Амплитуда E₀, В/м', fontsize=12)
plt.ylabel('Энергия W, Дж', fontsize=12)
plt.title('Зависимость переносимой энергии от амплитуды электрического поля', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()

# 6. Дополнительный график: временная зависимость вектора Пойнтинга
time = np.linspace(0, 2e-8, 1000)  # 20 нс
f = 1e6  # частота 1 МГц (произвольная, для наглядности)
omega = 2 * np.pi * f

# Мгновенные значения полей
E_inst = E0 * np.cos(omega * time)
B_inst = B0 * np.cos(omega * time)
S_inst = (1/mu0) * E_inst * B_inst  # мгновенный вектор Пойнтинга
S_avg = I * np.ones_like(time)      # среднее значение

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time*1e9, E_inst, 'b-', linewidth=1.5)
plt.ylabel('E(t), В/м', fontsize=12)
plt.title('Электрическое поле', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 2)
plt.plot(time*1e9, B_inst*1e12, 'r-', linewidth=1.5)
plt.ylabel('B(t), пТл', fontsize=12)
plt.title('Магнитное поле', fontsize=12)
plt.grid(True, alpha=0.3)

plt.subplot(3, 1, 3)
plt.plot(time*1e9, S_inst, 'g-', linewidth=1.5, label='Мгновенное значение')
plt.plot(time*1e9, S_avg, 'k--', linewidth=2, label='Среднее значение')
plt.xlabel('Время, нс', fontsize=12)
plt.ylabel('S(t), Вт/м²', fontsize=12)
plt.title('Вектор Пойнтинга', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()