import math

# Дано
t = 4e9                     # лет
T_half = 1.405e10           # лет (период полураспада Th-232)
m_Th0 = 1.0                 # кг
A_Th = 232
A_Pb = 208

# Постоянная распада
lambda_decay = math.log(2) / T_half

# Доля оставшегося тория
fraction_remaining = math.exp(-lambda_decay * t)
fraction_decayed = 1 - fraction_remaining

# Масса распавшегося тория
m_Th_decayed = m_Th0 * fraction_decayed

# Масса образовавшегося свинца
m_Pb = m_Th_decayed * (A_Pb / A_Th)

print(f"Доля распавшегося тория: {fraction_decayed:.3f}")
print(f"Масса распавшегося тория: {m_Th_decayed:.3f} кг")
print(f"Масса образовавшегося свинца: {m_Pb:.3f} кг")