import math

# Дано
E_MeV = 5
U0_MeV = 10
a_fm = 5          # фм

# Константы
m_alpha_kg = 6.644e-27          # кг
hbar = 1.055e-34                # Дж·с
eV_to_J = 1.602e-19
fm_to_m = 1e-15

# Перевод в СИ
a_m = a_fm * fm_to_m
U0_minus_E_J = (U0_MeV - E_MeV) * 1e6 * eV_to_J

# Вычисление
sqrt_term = math.sqrt(2 * m_alpha_kg * U0_minus_E_J)
exponent = (2 * a_m / hbar) * sqrt_term
D = math.exp(-exponent)

print(f"Показатель экспоненты: {exponent:.3f}")
print(f"Коэффициент прозрачности D = {D:.3e}")