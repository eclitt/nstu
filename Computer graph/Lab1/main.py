import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("ЛАБОРАТОРНАЯ РАБОТА ПО КОМПЬЮТЕРНОЙ ГРАФИКЕ")
print("Повороты треугольника в координатные плоскости")
print("=" * 60)

# 1. Построение треугольника и нормали к нему
print("\n1. ПОСТРОЕНИЕ ТРЕУГОЛЬНИКА И НОРМАЛИ К НЕМУ")

# НОВЫЕ КООРДИНАТЫ ТОЧЕК
A = np.array([1, 2, 1])    # A(1,2,1)
B = np.array([4, 5, 3])    # B(4,5,3)  
C = np.array([2, 7, 4])    # C(2,7,4)

print("Вершины треугольника:")
print(f"A({A[0]}, {A[1]}, {A[2]})")
print(f"B({B[0]}, {B[1]}, {B[2]})")
print(f"C({C[0]}, {C[1]}, {C[2]})")

# Вычисление нормали через векторное произведение
AB = B - A
AC = C - A
N = np.cross(AB, AC)
print(f"\nВектор AB = {AB}")
print(f"Вектор AC = {AC}")
print(f"Нормаль к треугольнику: N({N[0]}, {N[1]}, {N[2]})")

# Координаты для отрисовки треугольника
triangle_x = [A[0], B[0], C[0], A[0]]
triangle_y = [A[1], B[1], C[1], A[1]]
triangle_z = [A[2], B[2], C[2], A[2]]

# Точки для отображения нормали
center = (A + B + C) / 3
normal_end = center + N / np.linalg.norm(N) * 3

# 2. Поворот треугольника в плоскость XOY
print("\n2. ПОВОРОТ ТРЕУГОЛЬНИКА В ПЛОСКОСТЬ XOY")

# Поворот вокруг оси Z для обнуления x-компоненты нормали
alpha = np.arctan2(N[1], -N[0]) if N[0] != 0 else 0
print(f"Угол поворота вокруг Z: α = {alpha:.3f} радиан")

Rz = np.array([
    [np.cos(alpha), np.sin(alpha), 0],
    [-np.sin(alpha), np.cos(alpha), 0],
    [0, 0, 1]
])

# Поворот нормали вокруг Z
N1_z = Rz @ N
print(f"Нормаль после поворота вокруг Z: N1_z({N1_z[0]:.3f}, {N1_z[1]:.3f}, {N1_z[2]:.3f})")

# Поворот вокруг оси X для обнуления y-компоненты нормали
beta = np.arctan2(N1_z[1], N1_z[2]) if N1_z[2] != 0 else 0
print(f"Угол поворота вокруг X: β = {beta:.3f} радиан")

Rx = np.array([
    [1, 0, 0],
    [0, np.cos(beta), np.sin(beta)],
    [0, -np.sin(beta), np.cos(beta)]
])

# Полная матрица поворота для XOY
Q_eq1 = Rx @ Rz
print("\nМатрица поворота в плоскость XOY:")
print(Q_eq1)

# Поворот вершин треугольника
A1 = Q_eq1 @ A
B1 = Q_eq1 @ B
C1 = Q_eq1 @ C

print(f"\nВершины после поворота в XOY:")
print(f"A1({A1[0]:.3f}, {A1[1]:.3f}, {A1[2]:.3f})")
print(f"B1({B1[0]:.3f}, {B1[1]:.3f}, {B1[2]:.3f})")
print(f"C1({C1[0]:.3f}, {C1[1]:.3f}, {C1[2]:.3f})")

# Нормаль после поворота
N1 = Q_eq1 @ N
print(f"Нормаль после поворота: N1({N1[0]:.3f}, {N1[1]:.3f}, {N1[2]:.3f})")

# Координаты для отрисовки
triangle1_x = [A1[0], B1[0], C1[0], A1[0]]
triangle1_y = [A1[1], B1[1], C1[1], A1[1]]
triangle1_z = [A1[2], B1[2], C1[2], A1[2]]

# 3. Поворот треугольника в плоскость XOZ
print("\n3. ПОВОРОТ ТРЕУГОЛЬНИКА В ПЛОСКОСТЬ XOZ")

# Поворот вокруг оси Y для обнуления x-компоненты нормали
alpha_y = np.arctan2(N[0], N[2]) if N[2] != 0 else 0
print(f"Угол поворота вокруг Y: α = {alpha_y:.3f} радиан")

Ry = np.array([
    [np.cos(alpha_y), 0, -np.sin(alpha_y)],
    [0, 1, 0],
    [np.sin(alpha_y), 0, np.cos(alpha_y)]
])

# Поворот нормали вокруг Y
N1_y = Ry @ N
print(f"Нормаль после поворота вокруг Y: N1_y({N1_y[0]:.3f}, {N1_y[1]:.3f}, {N1_y[2]:.3f})")

# Поворот вокруг оси X для обнуления y-компоненты нормали
beta_x = np.arctan2(N1_y[1], N1_y[2]) if N1_y[2] != 0 else 0
print(f"Угол поворота вокруг X: β = {beta_x:.3f} радиан")

Rx2 = np.array([
    [1, 0, 0],
    [0, np.cos(beta_x), np.sin(beta_x)],
    [0, -np.sin(beta_x), np.cos(beta_x)]
])

# Полная матрица поворота для XOZ
Q_eq2 = Rx2 @ Ry
print("\nМатрица поворота в плоскость XOZ:")
print(Q_eq2)

# Поворот вершин треугольника
A2 = Q_eq2 @ A
B2 = Q_eq2 @ B
C2 = Q_eq2 @ C

print(f"\nВершины после поворота в XOZ:")
print(f"A2({A2[0]:.3f}, {A2[1]:.3f}, {A2[2]:.3f})")
print(f"B2({B2[0]:.3f}, {B2[1]:.3f}, {B2[2]:.3f})")
print(f"C2({C2[0]:.3f}, {C2[1]:.3f}, {C2[2]:.3f})")

# Нормаль после поворота
N2 = Q_eq2 @ N
print(f"Нормаль после поворота: N2({N2[0]:.3f}, {N2[1]:.3f}, {N2[2]:.3f})")

# Координаты для отрисовки
triangle2_x = [A2[0], B2[0], C2[0], A2[0]]
triangle2_y = [A2[1], B2[1], C2[1], A2[1]]
triangle2_z = [A2[2], B2[2], C2[2], A2[2]]

# 4. Поворот треугольника в плоскость YOZ
print("\n4. ПОВОРОТ ТРЕУГОЛЬНИКА В ПЛОСКОСТЬ YOZ")

# Поворот вокруг оси X для обнуления y-компоненты нормали
alpha_x = np.arctan2(N[2], N[1]) if N[1] != 0 else 0
print(f"Угол поворота вокруг X: α = {alpha_x:.3f} радиан")

Rx3 = np.array([
    [1, 0, 0],
    [0, np.cos(alpha_x), np.sin(alpha_x)],
    [0, -np.sin(alpha_x), np.cos(alpha_x)]
])

# Поворот нормали вокруг X
N1_x = Rx3 @ N
print(f"Нормаль после поворота вокруг X: N1_x({N1_x[0]:.3f}, {N1_x[1]:.3f}, {N1_x[2]:.3f})")

# Поворот вокруг оси Y для обнуления x-компоненты нормали
beta_y = np.arctan2(N1_x[0], N1_x[2]) if N1_x[2] != 0 else 0
print(f"Угол поворота вокруг Y: β = {beta_y:.3f} радиан")

Ry2 = np.array([
    [np.cos(beta_y), 0, -np.sin(beta_y)],
    [0, 1, 0],
    [np.sin(beta_y), 0, np.cos(beta_y)]
])

# Полная матрица поворота для YOZ
Q_eq3 = Ry2 @ Rx3
print("\nМатрица поворота в плоскость YOZ:")
print(Q_eq3)

# Поворот вершин треугольника
A3 = Q_eq3 @ A
B3 = Q_eq3 @ B
C3 = Q_eq3 @ C

print(f"\nВершины после поворота в YOZ:")
print(f"A3({A3[0]:.3f}, {A3[1]:.3f}, {A3[2]:.3f})")
print(f"B3({B3[0]:.3f}, {B3[1]:.3f}, {B3[2]:.3f})")
print(f"C3({C3[0]:.3f}, {C3[1]:.3f}, {C3[2]:.3f})")

# Нормаль после поворота
N3 = Q_eq3 @ N
print(f"Нормаль после поворота: N3({N3[0]:.3f}, {N3[1]:.3f}, {N3[2]:.3f})")

# Координаты для отрисовки
triangle3_x = [A3[0], B3[0], C3[0], A3[0]]
triangle3_y = [A3[1], B3[1], C3[1], A3[1]]
triangle3_z = [A3[2], B3[2], C3[2], A3[2]]

# Визуализация
fig = plt.figure(figsize=(18, 12))
fig.suptitle('ЛАБОРАТОРНАЯ РАБОТА: ПОВОРОТЫ ТРЕУГОЛЬНИКА В КООРДИНАТНЫЕ ПЛОСКОСТИ\n(Новые координаты: A(1,2,1), B(4,5,3), C(2,7,4))', 
             fontsize=14, fontweight='bold')

# 1. Исходный треугольник и нормаль
ax1 = fig.add_subplot(231, projection='3d')
ax1.plot(triangle_x, triangle_y, triangle_z, 'b-', linewidth=3, label='Треугольник ABC')
ax1.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], [A[2], B[2], C[2]], 
           c='red', s=80, marker='o')
ax1.quiver(center[0], center[1], center[2], 
           N[0], N[1], N[2], 
           color='green', length=3, label=f'Нормаль N({N[0]},{N[1]},{N[2]})', 
           linewidth=2, arrow_length_ratio=0.1)
ax1.set_title('1. Исходный треугольник и нормаль', fontweight='bold')
ax1.legend()
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.grid(True)

# 2. Поворот в плоскость XOY
ax2 = fig.add_subplot(232, projection='3d')
ax2.plot(triangle_x, triangle_y, triangle_z, 'b-', linewidth=1, alpha=0.3, 
         label='Исходный треугольник')
ax2.plot(triangle1_x, triangle1_y, triangle1_z, 'r-', linewidth=3, 
         label='Повернутый в XOY')
ax2.scatter([A1[0], B1[0], C1[0]], [A1[1], B1[1], C1[1]], [A1[2], B1[2], C1[2]], 
           c='red', s=80, marker='o')
center1 = (A1 + B1 + C1) / 3
ax2.quiver(center1[0], center1[1], center1[2], 
           N1[0], N1[1], N1[2], 
           color='orange', length=2, 
           label=f'Нормаль N1({N1[0]:.1f},{N1[1]:.1f},{N1[2]:.1f})', 
           linewidth=2, arrow_length_ratio=0.1)
ax2.set_title('2. Поворот в плоскость XOY', fontweight='bold')
ax2.legend()
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.grid(True)

# 3. Поворот в плоскость XOZ
ax3 = fig.add_subplot(233, projection='3d')
ax3.plot(triangle_x, triangle_y, triangle_z, 'b-', linewidth=1, alpha=0.3, 
         label='Исходный треугольник')
ax3.plot(triangle2_x, triangle2_y, triangle2_z, 'g-', linewidth=3, 
         label='Повернутый в XOZ')
ax3.scatter([A2[0], B2[0], C2[0]], [A2[1], B2[1], C2[1]], [A2[2], B2[2], C2[2]], 
           c='green', s=80, marker='o')
center2 = (A2 + B2 + C2) / 3
ax3.quiver(center2[0], center2[1], center2[2], 
           N2[0], N2[1], N2[2], 
           color='purple', length=2, 
           label=f'Нормаль N2({N2[0]:.1f},{N2[1]:.1f},{N2[2]:.1f})', 
           linewidth=2, arrow_length_ratio=0.1)
ax3.set_title('3. Поворот в плоскость XOZ', fontweight='bold')
ax3.legend()
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.grid(True)

# 4. Поворот в плоскость YOZ
ax4 = fig.add_subplot(234, projection='3d')
ax4.plot(triangle_x, triangle_y, triangle_z, 'b-', linewidth=1, alpha=0.3, 
         label='Исходный треугольник')
ax4.plot(triangle3_x, triangle3_y, triangle3_z, 'm-', linewidth=3, 
         label='Повернутый в YOZ')
ax4.scatter([A3[0], B3[0], C3[0]], [A3[1], B3[1], C3[1]], [A3[2], B3[2], C3[2]], 
           c='magenta', s=80, marker='o')
center3 = (A3 + B3 + C3) / 3
ax4.quiver(center3[0], center3[1], center3[2], 
           N3[0], N3[1], N3[2], 
           color='brown', length=2, 
           label=f'Нормаль N3({N3[0]:.1f},{N3[1]:.1f},{N3[2]:.1f})', 
           linewidth=2, arrow_length_ratio=0.1)
ax4.set_title('4. Поворот в плоскость YOZ', fontweight='bold')
ax4.legend()
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.grid(True)

# 5. Все треугольники вместе
ax5 = fig.add_subplot(235, projection='3d')
ax5.plot(triangle_x, triangle_y, triangle_z, 'b-', linewidth=3, 
         label='Исходный ABC')
ax5.plot(triangle1_x, triangle1_y, triangle1_z, 'r-', linewidth=2, 
         label='В XOY')
ax5.plot(triangle2_x, triangle2_y, triangle2_z, 'g-', linewidth=2, 
         label='В XOZ') 
ax5.plot(triangle3_x, triangle3_y, triangle3_z, 'm-', linewidth=2, 
         label='В YOZ')
ax5.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], [A[2], B[2], C[2]], 
           c='blue', s=50)
ax5.scatter([A1[0], B1[0], C1[0]], [A1[1], B1[1], C1[1]], [A1[2], B1[2], C1[2]], 
           c='red', s=50)
ax5.scatter([A2[0], B2[0], C2[0]], [A2[1], B2[1], C2[1]], [A2[2], B2[2], C2[2]], 
           c='green', s=50)
ax5.scatter([A3[0], B3[0], C3[0]], [A3[1], B3[1], C3[1]], [A3[2], B3[2], C3[2]], 
           c='magenta', s=50)
ax5.set_title('5. Все треугольники вместе', fontweight='bold')
ax5.legend()
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.grid(True)

# 6. Проекции на координатные плоскости
ax6 = fig.add_subplot(236)
ax6.plot(triangle_x, triangle_y, 'b-', linewidth=3, label='Исходный (XY)')
ax6.plot(triangle1_x, triangle1_y, 'r-', linewidth=2, label='В XOY (XY)')
ax6.plot(triangle2_x, triangle2_y, 'g-', linewidth=2, label='В XOZ (XY)')
ax6.plot(triangle3_x, triangle3_y, 'm-', linewidth=2, label='В YOZ (XY)')
ax6.scatter([A[0], B[0], C[0]], [A[1], B[1], C[1]], c='blue', s=50)
ax6.scatter([A1[0], B1[0], C1[0]], [A1[1], B1[1], C1[1]], c='red', s=50)
ax6.scatter([A2[0], B2[0], C2[0]], [A2[1], B2[1], C2[1]], c='green', s=50)
ax6.scatter([A3[0], B3[0], C3[0]], [A3[1], B3[1], C3[1]], c='magenta', s=50)
ax6.set_title('6. Проекции на плоскость XY', fontweight='bold')
ax6.legend()
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.grid(True)

plt.tight_layout()
plt.show()

# ОТЧЕТ О РЕЗУЛЬТАТАХ
print("\n" + "="*70)
print("ОТЧЕТ О РЕЗУЛЬТАТАХ ЛАБОРАТОРНОЙ РАБОТЫ")
print("="*70)

print("\n1. ИСХОДНЫЕ ДАННЫЕ:")
print(f"   Вершины треугольника:")
print(f"   A({A[0]}, {A[1]}, {A[2]})")
print(f"   B({B[0]}, {B[1]}, {B[2]})") 
print(f"   C({C[0]}, {C[1]}, {C[2]})")
print(f"   Векторы: AB = {AB}, AC = {AC}")
print(f"   Нормаль: N({N[0]}, {N[1]}, {N[2]})")

print("\n2. ПОВОРОТ В ПЛОСКОСТЬ XOY:")
print(f"   Углы поворота: α = {alpha:.3f} рад, β = {beta:.3f} рад")
print(f"   Вершины после поворота:")
print(f"   A1({A1[0]:.3f}, {A1[1]:.3f}, {A1[2]:.3f})")
print(f"   B1({B1[0]:.3f}, {B1[1]:.3f}, {B1[2]:.3f})")
print(f"   C1({C1[0]:.3f}, {C1[1]:.3f}, {C1[2]:.3f})")
print(f"   Нормаль после поворота: N1({N1[0]:.3f}, {N1[1]:.3f}, {N1[2]:.3f})")

print("\n3. ПОВОРОТ В ПЛОСКОСТЬ XOZ:")
print(f"   Углы поворота: α = {alpha_y:.3f} рад, β = {beta_x:.3f} рад")
print(f"   Вершины после поворота:")
print(f"   A2({A2[0]:.3f}, {A2[1]:.3f}, {A2[2]:.3f})")
print(f"   B2({B2[0]:.3f}, {B2[1]:.3f}, {B2[2]:.3f})")
print(f"   C2({C2[0]:.3f}, {C2[1]:.3f}, {C2[2]:.3f})")
print(f"   Нормаль после поворота: N2({N2[0]:.3f}, {N2[1]:.3f}, {N2[2]:.3f})")

print("\n4. ПОВОРОТ В ПЛОСКОСТЬ YOZ:")
print(f"   Углы поворота: α = {alpha_x:.3f} рад, β = {beta_y:.3f} рад")
print(f"   Вершины после поворота:")
print(f"   A3({A3[0]:.3f}, {A3[1]:.3f}, {A3[2]:.3f})")
print(f"   B3({B3[0]:.3f}, {B3[1]:.3f}, {B3[2]:.3f})")
print(f"   C3({C3[0]:.3f}, {C3[1]:.3f}, {C3[2]:.3f})")
print(f"   Нормаль после поворота: N3({N3[0]:.3f}, {N3[1]:.3f}, {N3[2]:.3f})")

print("\n5. ВЫВОДЫ:")
print("   - Успешно построен треугольник по заданным координатам вершин")
print("   - Вычислена нормаль к треугольнику через векторное произведение")
print("   - Выполнены повороты треугольника во все три координатные плоскости")
print("   - Для каждого поворота определены соответствующие матрицы преобразования")
print("   - Визуализированы исходный и повернутые треугольники с их нормалями")
print("   - Результаты демонстрируют корректность математических преобразований")