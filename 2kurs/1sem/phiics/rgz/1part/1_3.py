import numpy as np
import matplotlib.pyplot as plt

# Дано
A = 0.1          # Амплитуда, м
nu = 2.0         # Частота, Гц
omega = 2 * np.pi * nu  # Угловая частота, рад/с
phi0 = np.pi / 6 # Начальная фаза, рад
W = 0.077        # Полная энергия, Дж

# Вычисляем коэффициент k из полной энергии: W = (k * A^2) / 2
k = 2 * W / (A**2)
print(f"Коэффициент k = {k:.3f} Н/м")

# Функции энергий
def E_kin(t):
    # E_kin = (k*A^2/2) * cos^2(omega*t + phi0)
    return (k * A**2 / 2) * (np.cos(omega * t + phi0)**2)

def E_pot(t):
    # E_pot = (k*A^2/2) * sin^2(omega*t + phi0)
    return (k * A**2 / 2) * (np.sin(omega * t + phi0)**2)

# Создаем массив времени от 0 до 1.5 периодов
T = 1.0 / nu  # Период
t = np.linspace(0, 1.5*T, 500)

# Рассчитываем энергии
Ek = E_kin(t)
Ep = E_pot(t)

# Находим моменты времени, когда E_kin = E_pot (первые два после t=0)
# Решаем уравнение tan(omega*t + phi0) = +/-1
t1 = (np.pi/4 - phi0) / omega
t2 = (3*np.pi/4 - phi0) / omega  # следующее решение для tan = -1

print(f"Первый момент t1 = {t1:.6f} с ({t1*1000:.2f} мс)")
print(f"Второй момент t2 = {t2:.6f} с ({t2*1000:.2f} мс)")

# Строим график
plt.figure(figsize=(12, 7))

plt.plot(t, Ek, 'b-', linewidth=2, label=r'$E_{кин}(t)$')
plt.plot(t, Ep, 'r-', linewidth=2, label=r'$E_{пот}(t)$')
plt.axhline(y=W, color='green', linestyle='--', linewidth=2, label=r'$W_{полн} = const$')

# Отмечаем найденные точки
plt.scatter([t1, t2], [E_kin(t1), E_kin(t2)], color='black', s=80, zorder=5, label=r'$E_{кин}=E_{пот}$')
plt.axvline(x=t1, color='gray', linestyle=':', alpha=0.7)
plt.axvline(x=t2, color='gray', linestyle=':', alpha=0.7)

plt.xlabel('Время t, с', fontsize=14)
plt.ylabel('Энергия, Дж', fontsize=14)
plt.title('Кинетическая и потенциальная энергия гармонического осциллятора', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='best', fontsize=12)

# Добавляем небольшие оси для фазы
ax_phase = plt.gca().inset_axes([0.65, 0.15, 0.25, 0.2])
phase = omega * t + phi0
ax_phase.plot(t, phase, 'purple')
ax_phase.axhline(y=np.pi/4, color='gray', linestyle=':', alpha=0.7)
ax_phase.axhline(y=3*np.pi/4, color='gray', linestyle=':', alpha=0.7)
ax_phase.set_xlabel('t, с', fontsize=9)
ax_phase.set_ylabel('Фаза, рад', fontsize=9)
ax_phase.set_title('Фаза колебаний', fontsize=10)
ax_phase.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()