import random
import math

# Функция плотности распределения для варианта 1
# f(z) = 0.5 * (1 - 0.25 * z), z in [0, 4]
def f(z):
    return 0.5 * (1 - 0.25 * z)

# Параметры
N = 200                    # Нужный объем выборки
z_min = 0                  # Минимальное значение z
z_max = 4                  # Максимальное значение z
f_max = 0.5                # Максимум f(z) (достигается при z=0)

# Генерация выборки методом Неймана
sample = []
total_generated = 0        # Общее количество сгенерированных пар (включая отброшенные)

while len(sample) < N:
    # Генерируем кандидата
    z = random.uniform(z_min, z_max)      # z = 4 * u1
    u = random.uniform(0, f_max)          # u = 0.5 * u2
    
    total_generated += 1
    
    # Проверка условия принятия: u <= f(z)
    if u <= f(z):
        sample.append(z)

# Вывод результатов
print(f"Сгенерировано {N} значений Z")
print(f"Общее число сгенерированных пар (включая отброшенные): {total_generated}")
print(f"Эффективность метода: {N/total_generated*100:.2f}%")
print()

# Вывод первых 10 значений
print("Первые 10 значений Z:")
for i in range(10):
    print(f"{i+1}: {sample[i]:.8f}")

# Сохранение в текстовый файл
with open("neumann_results.txt", "w", encoding="utf-8") as f_out:
    f_out.write("Метод Неймана для варианта 1\n")
    f_out.write(f"f(z) = 0.5 * (1 - 0.25 * z), z in [0, 4]\n")
    f_out.write(f"Всего сгенерировано пар (включая отброшенные): {total_generated}\n")
    f_out.write(f"Эффективность: {N/total_generated*100:.2f}%\n\n")
    
    f_out.write("Все 200 значений Z:\n")
    for i, val in enumerate(sample, 1):
        f_out.write(f"{i}: {val:.8f}\n")

print("\nРезультаты сохранены в файл 'neumann_results.txt'")