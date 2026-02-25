import math

# Дано
A_eV = 2.3
rho = 534            # кг/м³
mu = 0.0069          # кг/моль
NA = 6.022e23        # моль⁻¹

# Константы
hbar = 1.055e-34     # Дж·с
m_e = 9.109e-31      # кг
eV_to_J = 1.602e-19

# Концентрация атомов (и электронов)
n_atoms = (rho / mu) * NA
n = n_atoms

# Энергия Ферми
term = (3 * math.pi**2 * n)**(2/3)
EF_J = (hbar**2 / (2 * m_e)) * term
EF_eV = EF_J / eV_to_J

# Глубина ямы
E0_eV = EF_eV + A_eV

print(f"Концентрация электронов n = {n:.3e} м⁻³")
print(f"Энергия Ферми E_F = {EF_eV:.3f} эВ")
print(f"Глубина потенциальной ямы E0 = {E0_eV:.3f} эВ")