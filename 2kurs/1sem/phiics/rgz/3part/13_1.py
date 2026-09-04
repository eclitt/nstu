import math

# Дано
a = 0.2e-9          # м
E_eV = 37.8         # эВ

# Константы
m_e = 9.109e-31     # кг
h = 6.626e-34       # Дж·с
hbar = 1.055e-34    # Дж·с
eV_to_J = 1.602e-19

# Энергия в Дж
E = E_eV * eV_to_J

# Номер уровня
n_squared = (8 * m_e * a**2 * E) / h**2
n = round(math.sqrt(n_squared))
print(f"n = {n}")

# Волновой вектор
k1 = n * math.pi / a
print(f"k = {k1:.3e} м^-1 (через n)")

# Проверка через энергию
k2 = math.sqrt(2 * m_e * E) / hbar
print(f"k = {k2:.3e} м^-1 (через энергию)")