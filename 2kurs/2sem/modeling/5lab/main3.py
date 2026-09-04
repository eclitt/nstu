import numpy as np
import scipy.stats as stats

# ==========================================
# 1. ГЕНЕРАЦИЯ ВЫБОРКИ МЕТОДОМ НЕЙМАНА
# ==========================================

def f(z):
    return 0.5 * (1 - 0.25 * z)

def neumann_sample(N, z_min=0, z_max=4):
    f_max = 0.5
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
np.random.seed(42)  # для воспроизводимости

# Генерируем выборку
sample, total_generated = neumann_sample(N)

# Выборочные характеристики
m_hat = np.mean(sample)
D_hat = np.var(sample, ddof=1)  # несмещённая оценка дисперсии
sigma_hat = np.sqrt(D_hat)

print("=" * 60)
print("ИСХОДНЫЕ ДАННЫЕ (метод Неймана, N=200)")
print("=" * 60)
print(f"Выборочное среднее m̂ = {m_hat:.6f}")
print(f"Выборочная дисперсия D̂ = {D_hat:.6f}")
print(f"Выборочное СКО σ̂ = {sigma_hat:.6f}")
print(f"Эффективность: {N/total_generated*100:.2f}%")
print()

# ==========================================
# 2. ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ
# ==========================================

alpha = 0.05  # уровень значимости (100% - 95% = 5%)
t_value = stats.t.ppf(1 - alpha/2, df=N-1)  # квантиль Стьюдента

margin_m = t_value * sigma_hat / np.sqrt(N)

ci_m_lower = m_hat - margin_m
ci_m_upper = m_hat + margin_m

print("=" * 60)
print("ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ МАТЕМАТИЧЕСКОГО ОЖИДАНИЯ (95%)")
print("=" * 60)
print(f"Квантиль Стьюдента t(0.025, {N-1}) = {t_value:.4f}")
print(f"Стандартная ошибка среднего = {sigma_hat/np.sqrt(N):.6f}")
print(f"Предельная ошибка Δ = {margin_m:.6f}")
print()
print(f"Доверительный интервал:")
print(f"  [{ci_m_lower:.6f}, {ci_m_upper:.6f}]")
print()

# Теоретическое значение M = 1.33333
M_theor = 4/3
if ci_m_lower <= M_theor <= ci_m_upper:
    print(f"✅ Теоретическое значение M = {M_theor:.4f} ПОПАДАЕТ в доверительный интервал.")
else:
    print(f"❌ Теоретическое значение M = {M_theor:.4f} НЕ ПОПАДАЕТ в доверительный интервал.")

# ==========================================
# 3. ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ ДИСПЕРСИИ
# ==========================================

# Квантили хи-квадрат
chi2_lower = stats.chi2.ppf(alpha/2, df=N-1)    # левая граница (0.025)
chi2_upper = stats.chi2.ppf(1 - alpha/2, df=N-1) # правая граница (0.975)

ci_D_lower = (N-1) * D_hat / chi2_upper
ci_D_upper = (N-1) * D_hat / chi2_lower

print("\n" + "=" * 60)
print("ДОВЕРИТЕЛЬНЫЙ ИНТЕРВАЛ ДЛЯ ДИСПЕРСИИ (95%)")
print("=" * 60)
print(f"Квантили χ²: χ²(0.025, {N-1}) = {chi2_lower:.4f}, χ²(0.975, {N-1}) = {chi2_upper:.4f}")
print()
print(f"Доверительный интервал:")
print(f"  [{ci_D_lower:.6f}, {ci_D_upper:.6f}]")
print()

# Теоретическое значение D = 0.88889
D_theor = 8/9
if ci_D_lower <= D_theor <= ci_D_upper:
    print(f"✅ Теоретическое значение D = {D_theor:.4f} ПОПАДАЕТ в доверительный интервал.")
else:
    print(f"❌ Теоретическое значение D = {D_theor:.4f} НЕ ПОПАДАЕТ в доверительный интервал.")

# ==========================================
# 4. ДЛЯ СРАВНЕНИЯ: ОЦЕНКИ ПО ВЫБОРКЕ
# ==========================================

print("\n" + "=" * 60)
print("СРАВНЕНИЕ С ТЕОРЕТИЧЕСКИМИ ЗНАЧЕНИЯМИ")
print("=" * 60)
print(f"Эмпирическое среднее:     {m_hat:.6f}")
print(f"Теоретическое среднее:    1.333333")
print(f"Ошибка εm:                {abs(m_hat - M_theor):.6f}")
print()
print(f"Эмпирическая дисперсия:   {D_hat:.6f}")
print(f"Теоретическая дисперсия:  0.888889")
print(f"Ошибка εD:                {abs(D_hat - D_theor):.6f}")