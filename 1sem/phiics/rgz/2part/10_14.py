# Дано
I_cal_cm2_min = 1.5  # кал/(см^2·мин)

# Перевод в Вт/м²
cal_to_J = 4.1868
cm2_to_m2 = 1e-4
min_to_s = 60

I_SI = I_cal_cm2_min * cal_to_J / (cm2_to_m2 * min_to_s)
print(f"Интенсивность: {I_SI:.1f} Вт/м²")

# Скорость света
c = 3e8  # м/с

# Давление на зачерненную
P_black = I_SI / c
print(f"Давление на зачерненную: {P_black:.3e} Па")

# Давление на зеркальную
P_mirror = 2 * I_SI / c
print(f"Давление на зеркальную:   {P_mirror:.3e} Па")