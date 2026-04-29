import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

# Параметры из отчета
R = 2.0  # Большой радиус
r = 1.0  # Малый радиус

# Уравнения поверхностей из отчета
def F1(x, y, z):
    return (x**2 + y**2 + z**2 + R**2 - r**2)**2 - 4 * R**2 * (x**2 + y**2)

def F2(x, y, z):
    return (x - r)**2 + y**2 + z**2 - R**2

# Градиенты из отчета
def gradient_F1(x, y, z):
    dfdx = 4*x*(x**2 + y**2 + z**2 + R**2 - r**2) - 8*R**2*x
    dfdy = 4*y*(x**2 + y**2 + z**2 + R**2 - r**2) - 8*R**2*y
    dfdz = 4*z*(x**2 + y**2 + z**2 + R**2 - r**2)
    return np.array([dfdx, dfdy, dfdz])

def gradient_F2(x, y, z):
    dfdx = 2*(x - r)
    dfdy = 2*y
    dfdz = 2*z
    return np.array([dfdx, dfdy, dfdz])

# Нормализованное векторное произведение нормалей (из отчета)
def normalized_cross_normals(x, y, z):
    n1 = gradient_F1(x, y, z)
    n2 = gradient_F2(x, y, z)
    cross = np.cross(n1, n2)
    norm = np.linalg.norm(cross)
    if norm == 0:
        return cross
    return cross / norm

# Параметризации из отчета
def torus1(u, v):
    # R1(u,v) из отчета
    x = (R + r * np.cos(u)) * np.cos(v)
    y = (R + r * np.cos(u)) * np.sin(v)
    z = r * np.sin(u)
    return x, y, z

def torus2(u, v):
    # R2(u,v) из отчета
    x = r + R * np.cos(u) * np.cos(v)
    y = R * np.cos(u) * np.sin(v)
    z = R * np.sin(u)
    return x, y, z

# Поиск начальной точки как в отчете (x=y)
def find_initial_point():
    def equations(vars):
        x, y, z = vars
        return [F1(x, y, z), F2(x, y, z), x - y]
    
    # Используем начальное приближение из отчета
    initial_guess = [1.5, 1.5, 0.5]
    
    solution = fsolve(equations, initial_guess)
    return solution

# Дифференциальное уравнение для трассировки линии пересечения
def intersection_curve(t, Y):
    x, y, z = Y
    direction = normalized_cross_normals(x, y, z)
    return direction

# Трассировка линии пересечения
def trace_intersection_curve(initial_point, t_span, n_points=300):
    sol = solve_ivp(intersection_curve, t_span, initial_point, 
                   method='RK45', t_eval=np.linspace(t_span[0], t_span[1], n_points),
                   rtol=1e-8, atol=1e-10)
    return sol.y

# Визуализация
def plot_tori_and_intersection():
    fig = plt.figure(figsize=(15, 10))
    
    # 3D график
    ax = fig.add_subplot(111, projection='3d')
    
    # Параметры для построения торов
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    # Построение первого тора (из отчета)
    X1, Y1, Z1 = torus1(U, V)
    ax.plot_surface(X1, Y1, Z1, alpha=0.7, color='blue', label='Тор 1')
    
    # Построение второго тора (из отчета)
    X2, Y2, Z2 = torus2(U, V)
    ax.plot_surface(X2, Y2, Z2, alpha=0.7, color='red', label='Тор 2')
    
    # Поиск и построение линии пересечения
    try:
        initial_point = find_initial_point()
        print(f"Найдена начальная точка: {initial_point}")
        
        # Трассировка в обе стороны от начальной точки
        curve_forward = trace_intersection_curve(initial_point, [0, 15])
        curve_backward = trace_intersection_curve(initial_point, [0, -15])
        
        # Объединение кривых
        full_curve = np.hstack([curve_backward[:, ::-1], curve_forward])
        
        ax.plot(full_curve[0], full_curve[1], full_curve[2], 
                color='green', linewidth=3, label='Линия пересечения')
        
        # Отмечаем начальную точку
        ax.scatter([initial_point[0]], [initial_point[1]], [initial_point[2]], 
                   color='black', s=100, marker='o', label='Начальная точка')
        
    except Exception as e:
        print(f"Ошибка при построении линии пересечения: {e}")
        import traceback
        traceback.print_exc()
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Пересечение двух торов (расчеты из отчета)')
    ax.legend()
    
    # Устанавливаем равные масштабы осей
    max_range = 4
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    
    plt.tight_layout()
    plt.show()

# Дополнительная визуализация с проекциями
def plot_projections():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Параметры для построения торов
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, 2 * np.pi, 50)
    U, V = np.meshgrid(u, v)
    
    # Построение первого тора
    X1, Y1, Z1 = torus1(U, V)
    
    # Построение второго тора
    X2, Y2, Z2 = torus2(U, V)
    
    # Поиск линии пересечения
    try:
        initial_point = find_initial_point()
        curve_forward = trace_intersection_curve(initial_point, [0, 15])
        curve_backward = trace_intersection_curve(initial_point, [0, -15])
        full_curve = np.hstack([curve_backward[:, ::-1], curve_forward])
    except:
        full_curve = None
    
    # Проекция XY
    axes[0].contour(X1, Y1, Z1, levels=[0], colors='blue', linewidths=2)
    axes[0].contour(X2, Y2, Z2, levels=[0], colors='red', linewidths=2)
    if full_curve is not None:
        axes[0].plot(full_curve[0], full_curve[1], 'g-', linewidth=2, label='Пересечение')
    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')
    axes[0].set_title('Проекция XY')
    axes[0].grid(True)
    axes[0].axis('equal')
    axes[0].legend()
    
    # Проекция XZ
    axes[1].contour(X1, Z1, Y1, levels=[0], colors='blue', linewidths=2)
    axes[1].contour(X2, Z2, Y2, levels=[0], colors='red', linewidths=2)
    if full_curve is not None:
        axes[1].plot(full_curve[0], full_curve[2], 'g-', linewidth=2, label='Пересечение')
    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Z')
    axes[1].set_title('Проекция XZ')
    axes[1].grid(True)
    axes[1].axis('equal')
    axes[1].legend()
    
    # Проекция YZ
    axes[2].contour(Y1, Z1, X1, levels=[0], colors='blue', linewidths=2)
    axes[2].contour(Y2, Z2, X2, levels=[0], colors='red', linewidths=2)
    if full_curve is not None:
        axes[2].plot(full_curve[1], full_curve[2], 'g-', linewidth=2, label='Пересечение')
    axes[2].set_xlabel('Y')
    axes[2].set_ylabel('Z')
    axes[2].set_title('Проекция YZ')
    axes[2].grid(True)
    axes[2].axis('equal')
    axes[2].legend()
    
    plt.tight_layout()
    plt.show()

# Визуализация изолиний для анализа пересечений
def plot_contour_analysis():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Различные уровни сечений
    z_levels = [-1.5, 0, 1, 1.5]
    
    for i, z in enumerate(z_levels):
        ax = axes[i//2, i%2]
        
        x = np.linspace(-3, 5, 100)
        y = np.linspace(-3, 3, 100)
        X, Y = np.meshgrid(x, y)
        
        Z1_vals = F1(X, Y, z)
        Z2_vals = F2(X, Y, z)
        
        # Контуры для обоих торов
        cs1 = ax.contour(X, Y, Z1_vals, levels=[0], colors='blue', linewidths=2)
        cs2 = ax.contour(X, Y, Z2_vals, levels=[0], colors='red', linewidths=2)
        
        ax.set_title(f'Сечение Z = {z}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        # Добавляем легенду только на первом графике
        if i == 0:
            ax.legend([cs1.collections[0], cs2.collections[0]], ['Тор 1', 'Тор 2'])
    
    plt.tight_layout()
    plt.show()

# Запуск визуализации
if __name__ == "__main__":
    print("Визуализация пересечения двух торов по расчетам из отчета")
    print("Параметры: R=2, r=1")
    print("Тор 1: классический тор вокруг оси Z")
    print("Тор 2: тор со смещенным центром")
    
    plot_tori_and_intersection()
    plot_projections()
    plot_contour_analysis()