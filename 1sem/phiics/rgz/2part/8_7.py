import math

# Данные
k = 0.05
phi_deg = 60
phi_rad = math.radians(phi_deg)

# Коэффициенты пропускания
tau = 1 - k

# 1) После первого николя
I1_rel = 0.5 * tau
decrease1 = 1 / I1_rel

# 2) После двух николей
I2_rel = 0.5 * (tau**2) * (math.cos(phi_rad))**2
decrease2 = 1 / I2_rel

print(f"После первого николя: I/I0 = {I1_rel:.4f}, уменьшение в {decrease1:.3f} раза")
print(f"После двух николей:   I/I0 = {I2_rel:.6f}, уменьшение в {decrease2:.3f} раза")