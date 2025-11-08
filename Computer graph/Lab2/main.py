import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Triangle3D:
    def __init__(self, points, name=""):
        self.points = np.array(points)
        self.A, self.B, self.C = points
        self.name = name
        
    def get_plane_equation(self):
        """Найти уравнение плоскости треугольника"""
        AB = self.B - self.A
        AC = self.C - self.A
        normal = np.cross(AB, AC)
        D = -np.dot(normal, self.A)
        return normal, D
    
    def plot(self, ax, color='blue', alpha=0.5):
        """Отрисовать треугольник"""
        vertices = np.vstack([self.points, self.points[0]])
        ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                color=color, linewidth=2, label=self.name)
        ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                  color=color, s=50)
        
        # Заполнение треугольника
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        triangle = Poly3DCollection([self.points], alpha=alpha, color=color)
        ax.add_collection3d(triangle)

def find_intersection_line(n1, D1, n2, D2):
    """Найти линию пересечения двух плоскостей"""
    # Направляющий вектор линии пересечения
    direction = np.cross(n1, n2)
    
    if np.linalg.norm(direction) < 1e-10:
        print("Плоскости параллельны!")
        return None, None
    
    # Нормализуем направляющий вектор
    direction = direction / np.linalg.norm(direction)
    
    # Находим точку на линии пересечения
    # Решаем систему из 2 уравнений с 3 неизвестными
    A = np.array([n1, n2])
    b = np.array([-D1, -D2])
    
    # Пробуем разные значения z для нахождения точки
    for z_val in [0, 1, 2]:
        try:
            # Фиксируем z и решаем для x, y
            A_2d = A[:, :2]  # Берем только x,y коэффициенты
            b_2d = b - A[:, 2] * z_val
            xy_solution = np.linalg.solve(A_2d, b_2d)
            point = np.array([xy_solution[0], xy_solution[1], z_val])
            
            # Проверяем, что точка удовлетворяет обоим уравнениям
            if abs(np.dot(n1, point) + D1) < 1e-10 and abs(np.dot(n2, point) + D2) < 1e-10:
                return point, direction
        except np.linalg.LinAlgError:
            continue
    
    # Если не нашли точку, используем метод наименьших квадратов
    try:
        # Расширенная матрица с дополнительным условием
        A_extended = np.vstack([A, [0, 0, 1]])
        b_extended = np.append(b, 0)
        point = np.linalg.lstsq(A_extended, b_extended, rcond=None)[0]
        return point, direction
    except:
        return None, None

# Задаем точки треугольников
a1 = np.array([1, 1, 1])
b1 = np.array([1, 3, 4])
c1 = np.array([5, 1, 0])

a2 = np.array([1, 1, 4])
b2 = np.array([3, 4, 1])
c2 = np.array([4, 2, 1])

# Создаем треугольники
triangle1 = Triangle3D([a1, b1, c1], "Треугольник 1")
triangle2 = Triangle3D([a2, b2, c2], "Треугольник 2")

# Находим уравнения плоскостей
n1, D1 = triangle1.get_plane_equation()
n2, D2 = triangle2.get_plane_equation()

print("Расчет уравнений плоскостей:")
print("=" * 50)

# Выводим подробные расчеты
print("\nДля треугольника 1:")
print(f"Точки: a1{a1}, b1{b1}, c1{c1}")
AB1 = b1 - a1
AC1 = c1 - a1
print(f"Вектор AB: {AB1}")
print(f"Вектор AC: {AC1}")
print(f"Нормальный вектор N1 = AB × AC = {n1}")
print(f"Уравнение плоскости: {n1[0]:.1f}x + {n1[1]:.1f}y + {n1[2]:.1f}z + {D1:.1f} = 0")

print("\nДля треугольника 2:")
print(f"Точки: a2{a2}, b2{b2}, c2{c2}")
AB2 = b2 - a2
AC2 = c2 - a2
print(f"Вектор AB: {AB2}")
print(f"Вектор AC: {AC2}")
print(f"Нормальный вектор N2 = AB × AC = {n2}")
print(f"Уравнение плоскости: {n2[0]:.1f}x + {n2[1]:.1f}y + {n2[2]:.1f}z + {D2:.1f} = 0")

# Проверка принадлежности точек плоскостям
print("\nПроверка уравнений:")
print("Треугольник 1:")
for point, name in zip([a1, b1, c1], ['a1', 'b1', 'c1']):
    result = np.dot(n1, point) + D1
    print(f"  {name}: {n1[0]:.1f}*{point[0]} + {n1[1]:.1f}*{point[1]} + {n1[2]:.1f}*{point[2]} + {D1:.1f} = {result:.1f}")

print("Треугольник 2:")
for point, name in zip([a2, b2, c2], ['a2', 'b2', 'c2']):
    result = np.dot(n2, point) + D2
    print(f"  {name}: {n2[0]:.1f}*{point[0]} + {n2[1]:.1f}*{point[1]} + {n2[2]:.1f}*{point[2]} + {D2:.1f} = {result:.1f}")

# Находим линию пересечения
intersection_point, direction = find_intersection_line(n1, D1, n2, D2)

# Визуализация
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Рисуем треугольники
triangle1.plot(ax, color='blue', alpha=0.3)
triangle2.plot(ax, color='red', alpha=0.3)

# Рисуем линию пересечения
if intersection_point is not None:
    t = np.linspace(-5, 5, 100)
    line_points = np.array([intersection_point + t_i * direction for t_i in t])
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
            'g-', linewidth=3, label='Линия пересечения плоскостей')
    
    ax.scatter(*intersection_point, color='green', s=100, marker='o', 
               label='Точка на линии пересечения')

# Настройка графика
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Пересечение двух треугольников')

# Устанавливаем равные масштабы
all_points = np.vstack([triangle1.points, triangle2.points])
max_range = np.max(np.ptp(all_points, axis=0))
mid_point = np.mean(all_points, axis=0)

ax.set_xlim(mid_point[0] - max_range/2, mid_point[0] + max_range/2)
ax.set_ylim(mid_point[1] - max_range/2, mid_point[1] + max_range/2)
ax.set_zlim(mid_point[2] - max_range/2, mid_point[2] + max_range/2)

plt.tight_layout()
plt.show()

# Вывод результатов
print("\n" + "=" * 50)
print("РЕЗУЛЬТАТЫ:")
print("=" * 50)

print("\nУравнение плоскости треугольника 1:")
print(f"{n1[0]:.4f}x + {n1[1]:.4f}y + {n1[2]:.4f}z + {D1:.4f} = 0")

print("\nУравнение плоскости треугольника 2:")
print(f"{n2[0]:.4f}x + {n2[1]:.4f}y + {n2[2]:.4f}z + {D2:.4f} = 0")

if intersection_point is not None:
    print(f"\nЛиния пересечения плоскостей:")
    print(f"Точка на линии: ({intersection_point[0]:.4f}, {intersection_point[1]:.4f}, {intersection_point[2]:.4f})")
    print(f"Направляющий вектор: ({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f})")
    print(f"\nПараметрическое уравнение линии:")
    print(f"x = {intersection_point[0]:.4f} + {direction[0]:.4f}t")
    print(f"y = {intersection_point[1]:.4f} + {direction[1]:.4f}t")
    print(f"z = {intersection_point[2]:.4f} + {direction[2]:.4f}t")
else:
    print("\nПлоскости параллельны и не пересекаются!")