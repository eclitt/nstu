import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# ==========================================
# 1. ГЕНЕРАЦИЯ ВЫБОРКИ МЕТОДОМ НЕЙМАНА
# ==========================================

def f(z):
    """Функция плотности распределения"""
    return 0.5 * (1 - 0.25 * z)

def neumann_sample(N, z_min=0, z_max=4):
    """Генерация выборки методом Неймана"""
    f_max = 0.5  # максимум f(z) при z=0
    
    sample = []
    total_generated = 0
    
    while len(sample) < N:
        z = np.random.uniform(z_min, z_max)
        u = np.random.uniform(0, f_max)
        total_generated += 1
        
        if u <= f(z):
            sample.append(z)
    
    return np.array(sample), total_generated

# Параметры
N = 200
z_min = 0
z_max = 4

# Генерируем выборку
np.random.seed(42)  # для воспроизводимости
sample, total_generated = neumann_sample(N)

print("=" * 60)
print("МЕТОД НЕЙМАНА - РЕЗУЛЬТАТЫ ГЕНЕРАЦИИ")
print("=" * 60)
print(f"Размер выборки: {N}")
print(f"Всего сгенерировано пар: {total_generated}")
print(f"Эффективность: {N/total_generated*100:.2f}%")
print(f"Среднее выборочное: {np.mean(sample):.6f}")
print(f"Дисперсия выборочная: {np.var(sample, ddof=1):.6f}")

# ==========================================
# 2. КРИТЕРИЙ ХИ-КВАДРАТ
# ==========================================

bins = [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
observed_freq, _ = np.histogram(sample, bins=bins)

def theoretical_prob(a, b):
    return 0.5*(b - a) - 0.0625*(b**2 - a**2)

theoretical_freq = []
for i in range(len(bins)-1):
    p = theoretical_prob(bins[i], bins[i+1])
    theoretical_freq.append(p * N)

chi2_sum = 0
print("\n" + "=" * 60)
print("КРИТЕРИЙ ХИ-КВАДРАТ")
print("=" * 60)
print(f"{'Интервал':<12} {'n_i':<10} {'np_i':<10} {'(n_i-np_i)²/np_i':<15}")
print("-" * 50)

for i in range(len(bins)-1):
    ni = observed_freq[i]
    npi = theoretical_freq[i]
    term = (ni - npi)**2 / npi if npi > 0 else 0
    chi2_sum += term
    print(f"{bins[i]:.1f}-{bins[i+1]:.1f}   {ni:<10} {npi:<10.2f} {term:<15.6f}")

print("-" * 50)
print(f"Сумма χ² = {chi2_sum:.6f}")

df = len(bins) - 1 - 2  # k - 1 - r
chi2_crit = stats.chi2.ppf(0.95, df)

print(f"\nСтепени свободы: ν = {df}")
print(f"Критическое значение χ²(0.05, {df}) = {chi2_crit:.4f}")

if chi2_sum < chi2_crit:
    print("\n✅ ГИПОТЕЗА ПРИНИМАЕТСЯ: распределение соответствует теоретическому")
else:
    print("\n❌ ГИПОТЕЗА ОТВЕРГАЕТСЯ: распределение НЕ соответствует теоретическому")

# ==========================================
# 3. ПОСТРОЕНИЕ ГРАФИКА
# ==========================================

plt.figure(figsize=(10, 6))

# Гистограмма эмпирической плотности (нормированная)
plt.hist(sample, bins=bins, density=True, alpha=0.6, 
         color='skyblue', edgecolor='black', label='Эмпирическая гистограмма')

# Теоретическая кривая плотности
z_theor = np.linspace(0, 4, 200)
f_theor = f(z_theor)
plt.plot(z_theor, f_theor, 'r-', linewidth=2, label='Теоретическая плотность f(z)')

# Настройки графика
plt.xlabel('z', fontsize=12)
plt.ylabel('Плотность вероятности f(z)', fontsize=12)
plt.title('Эмпирическая и теоретическая функции плотности\n(метод Неймана, N=200)', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 4)
plt.ylim(0, 0.55)

# Добавление подписи с результатами хи-квадрат
plt.text(2.5, 0.48, f'χ² = {chi2_sum:.4f}\nχ²крит = {chi2_crit:.4f}\nν = {df}', 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

# Сохранение графика
plt.savefig('histogram_neumann.png', dpi=150, bbox_inches='tight')
print("\nГрафик сохранён в файл 'histogram_neumann.png'")

# Показать график
plt.show()

print("\n" + "=" * 60)
print("Работа завершена!")
print("=" * 60)