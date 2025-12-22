import numpy as np

# Данные
t = 60.0          # с
r = 0.05          # м
A = 0.001         # м
v = 5000.0        # м/с
nu = 1e5          # Гц
rho = 2500.0      # кг/м³

# Вычисления
S = np.pi * r**2
omega = 2 * np.pi * nu

# Интенсивность
I = 0.5 * rho * v * (omega**2) * (A**2)

# Энергия
W = I * S * t

print("РАСЧЕТ ЭНЕРГИИ ЗВУКОВОЙ ВОЛНЫ")
print("="*40)
print(f"Площадь площадки: S = π×r² = {S:.6f} м²")
print(f"Циклическая частота: ω = 2πν = {omega:.2e} рад/с")
print(f"ω² = {omega**2:.2e} рад²/с²")
print(f"A² = {A**2:.2e} м²")
print(f"ρv = {rho*v:.2e} кг/(м²·с)")
print(f"Интенсивность: I = ½ρvω²A² = {I:.2e} Вт/м²")
print(f"Энергия за {t} с: W = I·S·t = {W:.2e} Дж")
print(f"В тераджоулях: {W/1e12:.2f} ТДж")

# Для сравнения: энергия 1 кг тротила ~ 4.18 МДж
TNT_equivalent = W / 4.18e6
print(f"Эквивалент тротила: {TNT_equivalent:.0f} кг = {TNT_equivalent/1000:.1f} тонн")

# График зависимости энергии от амплитуды
import matplotlib.pyplot as plt

A_range = np.logspace(-6, -2, 100)  # Амплитуды от 1 мкм до 1 см
W_range = 0.5 * rho * v * (omega**2) * (A_range**2) * S * t

plt.figure(figsize=(10, 6))
plt.loglog(A_range, W_range, 'b-', linewidth=2)
plt.axvline(x=A, color='r', linestyle='--', label=f'Заданная A = {A} м')
plt.axhline(y=W, color='r', linestyle='--', label=f'W = {W:.2e} Дж')
plt.xlabel('Амплитуда A, м', fontsize=12)
plt.ylabel('Энергия W, Дж', fontsize=12)
plt.title('Зависимость переносимой энергии от амплитуды волны', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()     # м/с
nu = 1e5          # Гц
rho = 2500.0      # кг/м³

# Вычисления
S = np.pi * r**2
omega = 2 * np.pi * nu

# Интенсивность
I = 0.5 * rho * v * (omega**2) * (A**2)

# Энергия
W = I * S * t

print("РАСЧЕТ ЭНЕРГИИ ЗВУКОВОЙ ВОЛНЫ")
print("="*40)
print(f"Площадь площадки: S = π×r² = {S:.6f} м²")
print(f"Циклическая частота: ω = 2πν = {omega:.2e} рад/с")
print(f"ω² = {omega**2:.2e} рад²/с²")
print(f"A² = {A**2:.2e} м²")
print(f"ρv = {rho*v:.2e} кг/(м²·с)")
print(f"Интенсивность: I = ½ρvω²A² = {I:.2e} Вт/м²")
print(f"Энергия за {t} с: W = I·S·t = {W:.2e} Дж")
print(f"В тераджоулях: {W/1e12:.2f} ТДж")

# Для сравнения: энергия 1 кг тротила ~ 4.18 МДж
TNT_equivalent = W / 4.18e6
print(f"Эквивалент тротила: {TNT_equivalent:.0f} кг = {TNT_equivalent/1000:.1f} тонн")

# График зависимости энергии от амплитуды
import matplotlib.pyplot as plt

A_range = np.logspace(-6, -2, 100)  # Амплитуды от 1 мкм до 1 см
W_range = 0.5 * rho * v * (omega**2) * (A_range**2) * S * t

plt.figure(figsize=(10, 6))
plt.loglog(A_range, W_range, 'b-', linewidth=2)
plt.axvline(x=A, color='r', linestyle='--', label=f'Заданная A = {A} м')
plt.axhline(y=W, color='r', linestyle='--', label=f'W = {W:.2e} Дж')
plt.xlabel('Амплитуда A, м', fontsize=12)
plt.ylabel('Энергия W, Дж', fontsize=12)
plt.title('Зависимость переносимой энергии от амплитуды волны', fontsize=14)
plt.grid(True, which='both', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()