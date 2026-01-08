import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

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
        A_extended = np.vstack([A, [0, 0, 1]])
        b_extended = np.append(b, 0)
        point = np.linalg.lstsq(A_extended, b_extended, rcond=None)[0]
        return point, direction
    except:
        return None, None

def line_triangle_intersection(line_point, line_dir, triangle):
    """Найти точку пересечения линии с треугольником"""
    A, B, C = triangle.points
    
    # Нормаль треугольника
    normal, D = triangle.get_plane_equation()
    
    # Проверка параллельности линии и плоскости
    if abs(np.dot(line_dir, normal)) < 1e-10:
        return None
    
    # Находим параметр t пересечения линии и плоскости
    t = -(np.dot(normal, line_point) + D) / np.dot(normal, line_dir)
    intersection_point = line_point + t * line_dir
    
    # Проверяем, лежит ли точка внутри треугольника
    def point_in_triangle(P, A, B, C):
        # Векторы сторон
        v0 = C - A
        v1 = B - A
        v2 = P - A
        
        # Вычисляем dot products
        dot00 = np.dot(v0, v0)
        dot01 = np.dot(v0, v1)
        dot02 = np.dot(v0, v2)
        dot11 = np.dot(v1, v1)
        dot12 = np.dot(v1, v2)
        
        # Вычисляем барицентрические координаты
        inv_denom = 1 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        
        # Проверяем, лежит ли точка внутри треугольника
        return (u >= 0) and (v >= 0) and (u + v <= 1)
    
    if point_in_triangle(intersection_point, A, B, C):
        return intersection_point
    else:
        return None

# Задаем координаты вершин согласно отчету
a1 = np.array([1, 1, 1])
b1 = np.array([1, 5, 4])
c1 = np.array([5, 1, 4])
d1 = np.array([1, 1, 5])  # Для пирамиды

a2 = np.array([1, 1, 4])
b2 = np.array([3, 4, 1])
c2 = np.array([4, 2, 1])
d2 = np.array([5, 3, 2])  # Для пирамиды

# Создаем треугольники для оснований пирамид
triangle1_base = Triangle3D([a1, b1, c1], "Треугольник 1 (основание)")
triangle2_base = Triangle3D([a2, b2, c2], "Треугольник 2 (основание)")

# Находим уравнения плоскостей
n1, D1 = triangle1_base.get_plane_equation()
n2, D2 = triangle2_base.get_plane_equation()

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

# Находим линию пересечения плоскостей
intersection_point, direction = find_intersection_line(n1, D1, n2, D2)

# Визуализация
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# Рисуем основания треугольников
triangle1_base.plot(ax, color='blue', alpha=0.3)
triangle2_base.plot(ax, color='red', alpha=0.3)

# Рисуем пирамиды
def plot_pyramid(ax, base_points, apex, color, alpha=0.3):
    """Отрисовать пирамиду"""
    # Основание
    base = Poly3DCollection([base_points], alpha=alpha, color=color)
    ax.add_collection3d(base)
    
    # Боковые грани
    for i in range(len(base_points)):
        next_idx = (i + 1) % len(base_points)
        side = Poly3DCollection([[base_points[i], base_points[next_idx], apex]], 
                               alpha=alpha*0.7, color=color)
        ax.add_collection3d(side)
    
    # Ребра
    vertices = np.vstack([base_points, base_points[0]])
    ax.plot(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            color=color, linewidth=2)
    
    for point in base_points:
        ax.plot([point[0], apex[0]], [point[1], apex[1]], [point[2], apex[2]], 
                color=color, linewidth=2)
    
    # Вершины
    ax.scatter(*apex, color=color, s=50)

# Рисуем пирамиды
plot_pyramid(ax, [a1, b1, c1], d1, 'blue', 0.2)
plot_pyramid(ax, [a2, b2, c2], d2, 'red', 0.2)

# Рисуем линию пересечения плоскостей
if intersection_point is not None:
    t = np.linspace(-5, 5, 100)
    line_points = np.array([intersection_point + t_i * direction for t_i in t])
    ax.plot(line_points[:, 0], line_points[:, 1], line_points[:, 2], 
            'g-', linewidth=3, label='Линия пересечения плоскостей')
    
    ax.scatter(*intersection_point, color='green', s=100, marker='o',
              label='Точка на линии пересечения')

# Находим и рисуем точки пересечения линии с треугольниками
if intersection_point is not None:
    # Пересечение с треугольником 1
    intersection_t1 = line_triangle_intersection(intersection_point, direction, triangle1_base)
    if intersection_t1 is not None:
        ax.scatter(*intersection_t1, color='yellow', s=100, marker='s',
                  label='Пересечение с треугольником 1')
        print(f"\nПересечение с треугольником 1: {intersection_t1}")
    
    # Пересечение с треугольником 2
    intersection_t2 = line_triangle_intersection(intersection_point, direction, triangle2_base)
    if intersection_t2 is not None:
        ax.scatter(*intersection_t2, color='orange', s=100, marker='^',
                  label='Пересечение с треугольником 2')
        print(f"Пересечение с треугольником 2: {intersection_t2}")

# Настройка графика
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
ax.set_title('Пересечение двух пирамид (треугольников)')

# Устанавливаем равные масштабы
all_points = np.vstack([triangle1_base.points, triangle2_base.points, [d1], [d2]])
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
    print(f"\nЛиния пересечения:")
    print(f"Точка на линии: ({intersection_point[0]:.4f}, {intersection_point[1]:.4f}, {intersection_point[2]:.4f})")
    print(f"Направляющий вектор: ({direction[0]:.4f}, {direction[1]:.4f}, {direction[2]:.4f})")
    print(f"Параметрическое уравнение линии:")
    print(f"x = {intersection_point[0]:.4f} + t*{direction[0]:.4f}")
    print(f"y = {intersection_point[1]:.4f} + t*{direction[1]:.4f}")
    print(f"z = {intersection_point[2]:.4f} + t*{direction[2]:.4f}")
else:
    print("\nПлоскости параллельны, пересечения нет!")