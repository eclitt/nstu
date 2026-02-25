# Дано
T_alpha_MeV = 5.3
m_alpha = 4.0026      # а.е.м.
M = 206               # а.е.м. (приближённо)

# Отношение масс
ratio = m_alpha / M

# Энергия отдачи
T_recoil_MeV = T_alpha_MeV * ratio

# Полная энергия
Q_MeV = T_alpha_MeV * (1 + ratio)

print(f"Энергия ядра отдачи: {T_recoil_MeV:.3f} МэВ")
print(f"Полная энергия распада Q: {Q_MeV:.3f} МэВ")