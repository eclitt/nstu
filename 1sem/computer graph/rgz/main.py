import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
class RotatingCube3D:
    def __init__(self):
        # Создаём фигуру
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.patch.set_facecolor('#1a1a2e')
        
        # Основной график для 3D визуализации
        self.ax = self.fig.add_subplot(231, projection='3d')
        self.ax.set_box_aspect([1,1,1])
        
        # График для фронтальной проекции (XZ плоскость)
        self.ax_front = self.fig.add_subplot(232)
        self.ax_front.set_aspect('equal')
        self.ax_front.set_title('Фронтальная проекция (XZ)', color='white')
        self.ax_front.set_facecolor('#0a0a1a')
        
        # График для боковой проекции (YZ плоскость)
        self.ax_side = self.fig.add_subplot(233)
        self.ax_side.set_aspect('equal')
        self.ax_side.set_title('Боковая проекция (YZ)', color='white')
        self.ax_side.set_facecolor('#0a0a1a')
        
        # График для матричных преобразований
        self.ax_matrix = self.fig.add_subplot(234)
        self.ax_matrix.axis('off')
        self.ax_matrix.set_title('Матрицы преобразований', color='white', fontsize=12)
        self.ax_matrix.set_facecolor('#0a0a1a')
        
        # График для информации
        self.ax_info = self.fig.add_subplot(235)
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#0a0a1a')
        
        # График для управления
        self.ax_control = self.fig.add_subplot(236)
        self.ax_control.axis('off')
        self.ax_control.set_facecolor('#0a0a1a')
        
        # Настройка 3D вида
        self.setup_3d_view()
        
        # Исходные вершины куба (локальные координаты)
        self.vertices_local = np.array([
            [-1, -1, -1],  # 0: лево-низ-зад
            [ 1, -1, -1],  # 1: право-низ-зад
            [ 1,  1, -1],  # 2: право-верх-зад
            [-1,  1, -1],  # 3: лево-верх-зад
            [-1, -1,  1],  # 4: лево-низ-перед
            [ 1, -1,  1],  # 5: право-низ-перед
            [ 1,  1,  1],  # 6: право-верх-перед
            [-1,  1,  1]   # 7: лево-верх-перед
        ], dtype=np.float64)
        
        # Грани куба (индексы вершин)
        self.faces = [
            [0, 1, 2, 3],  # задняя грань
            [4, 5, 6, 7],  # передняя грань
            [0, 3, 7, 4],  # левая грань
            [1, 2, 6, 5],  # правая грань
            [0, 1, 5, 4],  # нижняя грань
            [3, 2, 6, 7]   # верхняя грань
        ]
        
        # Названия граней
        self.face_names = ['Задняя', 'Передняя', 'Левая', 'Правая', 'Нижняя', 'Верхняя']
        
        # Базовые цвета граней (без освещения)
        self.base_colors = np.array([
            [0.8, 0.2, 0.2],  # красный
            [0.2, 0.8, 0.2],  # зелёный
            [0.2, 0.2, 0.8],  # синий
            [0.8, 0.8, 0.2],  # жёлтый
            [0.8, 0.2, 0.8],  # фиолетовый
            [0.2, 0.8, 0.8]   # голубой
        ])
        
        # Текущие преобразованные вершины
        self.vertices_transformed = self.vertices_local.copy()
        
        # Параметры преобразований
        self.rotation_angles = np.array([0.0, 0.0, 0.0])  # X, Y, Z в радианах
        self.scale_factors = np.array([1.0, 1.0, 1.0])    # масштаб по осям
        self.translation = np.array([0.0, 0.0, 0.0])      # смещение
        
        # Параметры освещения
        self.light_direction = np.array([1.0, 1.0, 1.0])  # направление света
        self.light_direction = self.light_direction / np.linalg.norm(self.light_direction)
        self.light_intensity = 0.7
        self.ambient_intensity = 0.3
        
        # Создаём UI элементы
        self.create_ui()
        
        # Инициализируем отображение
        self.update_cube()
        
        # Анимация
        self.animation_running = False
        self.ani = None
        
    def setup_3d_view(self):
        """Настраиваем 3D вид"""
        self.ax.set_xlim(-3, 3)
        self.ax.set_ylim(-3, 3)
        self.ax.set_zlim(-3, 3)
        self.ax.set_xlabel('X', color='white')
        self.ax.set_ylabel('Y', color='white')
        self.ax.set_zlabel('Z', color='white')
        self.ax.set_title('3D Куб с матричными преобразованиями', color='white', fontsize=14)
        self.ax.xaxis.label.set_color('white')
        self.ax.yaxis.label.set_color('white')
        self.ax.zaxis.label.set_color('white')
        self.ax.tick_params(colors='white')
        self.ax.set_facecolor('#0a0a1a')
        self.ax.grid(True, alpha=0.3)
        
        # Настраиваем проекции
        for ax_2d in [self.ax_front, self.ax_side]:
            ax_2d.set_xlim(-3, 3)
            ax_2d.set_ylim(-3, 3)
            ax_2d.grid(True, alpha=0.3)
            ax_2d.tick_params(colors='white')
            ax_2d.xaxis.label.set_color('white')
            ax_2d.yaxis.label.set_color('white')
            
        self.ax_front.set_xlabel('X', color='white')
        self.ax_front.set_ylabel('Z', color='white')
        
        self.ax_side.set_xlabel('Y', color='white')
        self.ax_side.set_ylabel('Z', color='white')
    
    def create_rotation_matrix_x(self, angle):
        """Матрица поворота вокруг оси X"""
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)]
        ])
    
    def create_rotation_matrix_y(self, angle):
        """Матрица поворота вокруг оси Y"""
        return np.array([
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)]
        ])
    
    def create_rotation_matrix_z(self, angle):
        """Матрица поворота вокруг оси Z"""
        return np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])
    
    def create_scale_matrix(self, sx, sy, sz):
        """Матрица масштабирования"""
        return np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, sz]
        ])
    
    def create_translation_matrix(self, tx, ty, tz):
        """Матрица смещения (однородные координаты)"""
        return np.array([
            [1, 0, 0, tx],
            [0, 1, 0, ty],
            [0, 0, 1, tz],
            [0, 0, 0, 1]
        ])
    
    def create_composite_matrix(self):
        """Создаём композитную матрицу преобразований"""
        # Порядок: масштаб → поворот → смещение
        R_x = self.create_rotation_matrix_x(self.rotation_angles[0])
        R_y = self.create_rotation_matrix_y(self.rotation_angles[1])
        R_z = self.create_rotation_matrix_z(self.rotation_angles[2])
        
        # Комбинированный поворот (порядок: Z → Y → X)
        R = R_x @ R_y @ R_z
        
        # Матрица масштабирования
        S = self.create_scale_matrix(*self.scale_factors)
        
        # Комбинированная матрица 3x3
        M = R @ S
        
        return M
    
    def apply_transformations(self):
        """Применяем все преобразования к вершинам"""
        M = self.create_composite_matrix()
        
        # Преобразуем вершины: v' = M * v^T
        self.vertices_transformed = (M @ self.vertices_local.T).T
        
        # Добавляем смещение
        self.vertices_transformed += self.translation
        
        # Вычисляем нормали граней
        self.compute_face_normals()
        
        # Вычисляем цвета с учётом освещения
        self.compute_face_colors()
    
    def compute_face_normals(self):
        """Вычисляем нормали граней"""
        self.face_normals = []
        self.face_centers = []
        
        for face in self.faces:
            # Получаем вершины грани
            v0, v1, v2, v3 = [self.vertices_transformed[i] for i in face]
            
            # Вычисляем центр грани
            center = (v0 + v1 + v2 + v3) / 4
            self.face_centers.append(center)
            
            # Вычисляем нормаль через векторное произведение
            vec1 = v1 - v0
            vec2 = v3 - v0
            normal = np.cross(vec1, vec2)
            
            # Нормализуем нормаль
            norm = np.linalg.norm(normal)
            if norm > 0:
                normal = normal / norm
            
            # Направляем нормаль наружу (от центра куба)
            if np.dot(normal, center) < 0:
                normal = -normal
                
            self.face_normals.append(normal)
    
    def compute_face_colors(self):
        """Вычисляем цвета граней с учётом освещения"""
        self.face_colors = []
        
        for i, normal in enumerate(self.face_normals):
            # Косинус угла между нормалью и направлением света
            cos_angle = np.dot(normal, self.light_direction)
            cos_angle = max(0, min(1, cos_angle))  # Ограничиваем [0, 1]
            
            # Модель освещения: ambient + diffuse
            intensity = self.ambient_intensity + self.light_intensity * cos_angle
            
            # Применяем интенсивность к базовому цвету
            color = self.base_colors[i] * intensity
            color = np.clip(color, 0, 1)  # Ограничиваем значения
            
            # Добавляем альфа-канал
            color = np.append(color, 0.8)
            self.face_colors.append(color)
    
    def create_ui(self):
        """Создаём элементы управления"""
        # Слайдеры для вращения
        slider_height = 0.03
        slider_width = 0.25
        slider_start_x = 0.05
        slider_start_y = 0.92
        
        # Слайдер для вращения X
        self.ax_slider_x = plt.axes([slider_start_x, slider_start_y - 0*0.05, 
                                    slider_width, slider_height], 
                                   facecolor='#2a2a4a')
        self.slider_x = Slider(self.ax_slider_x, 'Вращение X', -180, 180, 
                              valinit=0, valstep=1, color='#ff3366')
        self.slider_x.on_changed(self.update_from_slider)
        
        # Слайдер для вращения Y
        self.ax_slider_y = plt.axes([slider_start_x, slider_start_y - 1*0.05, 
                                    slider_width, slider_height],
                                   facecolor='#2a2a4a')
        self.slider_y = Slider(self.ax_slider_y, 'Вращение Y', -180, 180,
                              valinit=0, valstep=1, color='#33ff66')
        self.slider_y.on_changed(self.update_from_slider)
        
        # Слайдер для вращения Z
        self.ax_slider_z = plt.axes([slider_start_x, slider_start_y - 2*0.05,
                                    slider_width, slider_height],
                                   facecolor='#2a2a4a')
        self.slider_z = Slider(self.ax_slider_z, 'Вращение Z', -180, 180,
                              valinit=0, valstep=1, color='#3366ff')
        self.slider_z.on_changed(self.update_from_slider)
        
        # Слайдеры для масштабирования
        slider_start_x2 = 0.35
        
        self.ax_slider_sx = plt.axes([slider_start_x2, slider_start_y - 0*0.05,
                                     slider_width, slider_height],
                                    facecolor='#2a2a4a')
        self.slider_sx = Slider(self.ax_slider_sx, 'Масштаб X', 0.1, 3.0,
                               valinit=1.0, valstep=0.1, color='#ff6633')
        self.slider_sx.on_changed(self.update_from_slider)
        
        self.ax_slider_sy = plt.axes([slider_start_x2, slider_start_y - 1*0.05,
                                     slider_width, slider_height],
                                    facecolor='#2a2a4a')
        self.slider_sy = Slider(self.ax_slider_sy, 'Масштаб Y', 0.1, 3.0,
                               valinit=1.0, valstep=0.1, color='#66ff33')
        self.slider_sy.on_changed(self.update_from_slider)
        
        self.ax_slider_sz = plt.axes([slider_start_x2, slider_start_y - 2*0.05,
                                     slider_width, slider_height],
                                    facecolor='#2a2a4a')
        self.slider_sz = Slider(self.ax_slider_sz, 'Масштаб Z', 0.1, 3.0,
                               valinit=1.0, valstep=0.1, color='#3366ff')
        self.slider_sz.on_changed(self.update_from_slider)
        
        # Кнопки
        button_width = 0.12
        button_height = 0.05
        button_start_x = 0.05
        button_start_y = 0.75
        
        # Кнопка сброса
        self.ax_button_reset = plt.axes([button_start_x, button_start_y,
                                        button_width, button_height])
        self.button_reset = Button(self.ax_button_reset, 'Сброс',
                                  color='#2a2a4a', hovercolor='#3a3a5a')
        self.button_reset.on_clicked(self.reset_transformations)
        
        # Кнопка анимации
        self.ax_button_anim = plt.axes([button_start_x + button_width + 0.02, button_start_y,
                                       button_width, button_height])
        self.button_anim = Button(self.ax_button_anim, 'Старт/Стоп анимации',
                                 color='#2a2a4a', hovercolor='#3a3a5a')
        self.button_anim.on_clicked(self.toggle_animation)
        
        # Настройка слайдера освещения
        self.ax_slider_light = plt.axes([0.65, 0.05, 0.3, 0.03],
                                       facecolor='#2a2a4a')
        self.slider_light = Slider(self.ax_slider_light, 'Интенсивность света',
                                  0.0, 1.0, valinit=0.7, valstep=0.05,
                                  color='#ffff33')
        self.slider_light.on_changed(self.update_lighting)
    
    def update_from_slider(self, val):
        """Обновляем параметры из слайдеров"""
        self.rotation_angles = np.radians([
            self.slider_x.val,
            self.slider_y.val,
            self.slider_z.val
        ])
        
        self.scale_factors = np.array([
            self.slider_sx.val,
            self.slider_sy.val,
            self.slider_sz.val
        ])
        
        self.update_cube()
    
    def update_lighting(self, val):
        """Обновляем освещение"""
        self.light_intensity = self.slider_light.val
        self.update_cube()
    
    def reset_transformations(self, event):
        """Сбрасываем все преобразования"""
        self.slider_x.set_val(0)
        self.slider_y.set_val(0)
        self.slider_z.set_val(0)
        self.slider_sx.set_val(1.0)
        self.slider_sy.set_val(1.0)
        self.slider_sz.set_val(1.0)
        self.slider_light.set_val(0.7)
        
        self.rotation_angles = np.array([0.0, 0.0, 0.0])
        self.scale_factors = np.array([1.0, 1.0, 1.0])
        self.translation = np.array([0.0, 0.0, 0.0])
        
        self.update_cube()
    
    def toggle_animation(self, event):
        """Включаем/выключаем анимацию"""
        if self.animation_running:
            if self.ani:
                self.ani.event_source.stop()
            self.animation_running = False
            self.button_anim.label.set_text('Старт анимации')
        else:
            self.start_animation()
            self.animation_running = True
            self.button_anim.label.set_text('Стоп анимации')
    
    def start_animation(self):
        """Запускаем анимацию вращения"""
        if self.ani:
            self.ani.event_source.stop()
        
        self.ani = FuncAnimation(self.fig, self.animate_rotation,
                                frames=360, interval=20, blit=False)
    
    def animate_rotation(self, frame):
        """Функция анимации"""
        # Плавное вращение по всем осям
        angle = np.radians(frame * 2)
        self.rotation_angles[0] = angle * 0.5
        self.rotation_angles[1] = angle * 0.7
        self.rotation_angles[2] = angle * 0.3
        
        # Обновляем слайдеры (для отображения)
        self.slider_x.set_val(np.degrees(self.rotation_angles[0]))
        self.slider_y.set_val(np.degrees(self.rotation_angles[1]))
        self.slider_z.set_val(np.degrees(self.rotation_angles[2]))
        
        self.update_cube()
        return []
    
    def draw_cube_3d(self):
        """Рисуем куб в 3D"""
        self.ax.clear()
        self.setup_3d_view()
        
        # Применяем преобразования
        self.apply_transformations()
        
        # Рисуем каждую грань
        for i, face in enumerate(self.faces):
            vertices = [self.vertices_transformed[j] for j in face]
            
            # Создаём полигон для грани
            poly = [[vertices[0], vertices[1], vertices[2], vertices[3]]]
            face_collection = Poly3DCollection(poly, alpha=0.8)
            face_collection.set_facecolor(self.face_colors[i])
            face_collection.set_edgecolor('white')
            face_collection.set_linewidth(1)
            
            self.ax.add_collection3d(face_collection)
            
            # Подписываем центр грани
            center = self.face_centers[i]
            self.ax.text(center[0], center[1], center[2], 
                        self.face_names[i], fontsize=8, color='white',
                        ha='center', va='center')
        
        # Рисуем вершины
        for i, vertex in enumerate(self.vertices_transformed):
            self.ax.scatter(vertex[0], vertex[1], vertex[2], 
                          color='white', s=30, alpha=0.8)
            self.ax.text(vertex[0], vertex[1], vertex[2], 
                        str(i), fontsize=10, color='yellow',
                        ha='center', va='center')
        
        # Рисуем направление света
        light_end = self.light_direction * 2
        self.ax.quiver(0, 0, 0, light_end[0], light_end[1], light_end[2],
                      color='yellow', alpha=0.5, arrow_length_ratio=0.1)
        self.ax.text(light_end[0], light_end[1], light_end[2],
                    'Свет', color='yellow', fontsize=10)
    
    def draw_projections(self):
        """Рисуем 2D проекции"""
        # Фронтальная проекция (XZ плоскость)
        self.ax_front.clear()
        self.ax_front.set_title('Фронтальная проекция (XZ)', color='white')
        self.ax_front.set_xlabel('X', color='white')
        self.ax_front.set_ylabel('Z', color='white')
        self.ax_front.set_xlim(-3, 3)
        self.ax_front.set_ylim(-3, 3)
        self.ax_front.grid(True, alpha=0.3)
        self.ax_front.set_facecolor('#0a0a1a')
        
        for i, face in enumerate(self.faces):
            vertices = [self.vertices_transformed[j] for j in face]
            # Берем координаты X и Z
            x_coords = [v[0] for v in vertices] + [vertices[0][0]]
            z_coords = [v[2] for v in vertices] + [vertices[0][2]]
            self.ax_front.fill(x_coords, z_coords, 
                              color=self.face_colors[i], alpha=0.6)
            self.ax_front.plot(x_coords, z_coords, 'white', linewidth=1)
        
        # Боковая проекция (YZ плоскость)
        self.ax_side.clear()
        self.ax_side.set_title('Боковая проекция (YZ)', color='white')
        self.ax_side.set_xlabel('Y', color='white')
        self.ax_side.set_ylabel('Z', color='white')
        self.ax_side.set_xlim(-3, 3)
        self.ax_side.set_ylim(-3, 3)
        self.ax_side.grid(True, alpha=0.3)
        self.ax_side.set_facecolor('#0a0a1a')
        
        for i, face in enumerate(self.faces):
            vertices = [self.vertices_transformed[j] for j in face]
            # Берем координаты Y и Z
            y_coords = [v[1] for v in vertices] + [vertices[0][1]]
            z_coords = [v[2] for v in vertices] + [vertices[0][2]]
            self.ax_side.fill(y_coords, z_coords,
                            color=self.face_colors[i], alpha=0.6)
            self.ax_side.plot(y_coords, z_coords, 'white', linewidth=1)
    
    def draw_matrices_info(self):
        """Отображаем информацию о матрицах"""
        self.ax_matrix.clear()
        self.ax_matrix.axis('off')
        self.ax_matrix.set_title('Матрицы преобразований', color='white', fontsize=12)
        self.ax_matrix.set_facecolor('#0a0a1a')
        
        # Вычисляем матрицы
        R_x = self.create_rotation_matrix_x(self.rotation_angles[0])
        R_y = self.create_rotation_matrix_y(self.rotation_angles[1])
        R_z = self.create_rotation_matrix_z(self.rotation_angles[2])
        S = self.create_scale_matrix(*self.scale_factors)
        M = self.create_composite_matrix()
        
        # Форматируем матрицы для отображения
        matrices_info = []
        matrices_info.append("Матрица поворота X:")
        matrices_info.append(self.format_matrix(R_x))
        matrices_info.append("\nМатрица поворота Y:")
        matrices_info.append(self.format_matrix(R_y))
        matrices_info.append("\nМатрица поворота Z:")
        matrices_info.append(self.format_matrix(R_z))
        matrices_info.append("\nМатрица масштабирования:")
        matrices_info.append(self.format_matrix(S))
        matrices_info.append("\nРезультирующая матрица M = RₓRᵧR₂S:")
        matrices_info.append(self.format_matrix(M))
        
        # Отображаем текст
        text = "\n".join(matrices_info)
        self.ax_matrix.text(0.05, 0.95, text, transform=self.ax_matrix.transAxes,
                          fontsize=8, color='white', verticalalignment='top',
                          fontfamily='monospace')
    
    def format_matrix(self, matrix):
        """Форматируем матрицу для красивого отображения"""
        if matrix.shape == (4, 4):  # Для матрицы смещения
            rows = []
            for i in range(4):
                row = " ".join([f"{val:6.2f}" for val in matrix[i]])
                rows.append(f"[{row}]")
            return "\n".join(rows)
        else:  # Для матриц 3x3
            rows = []
            for i in range(3):
                row = " ".join([f"{val:6.2f}" for val in matrix[i]])
                rows.append(f"[{row}]")
            return "\n".join(rows)
    
    def draw_info_panel(self):
        """Рисуем информационную панель"""
        self.ax_info.clear()
        self.ax_info.axis('off')
        self.ax_info.set_facecolor('#0a0a1a')
        
        info_text = []
        info_text.append("ИНФОРМАЦИЯ:")
        info_text.append("=" * 30)
        info_text.append(f"Углы вращения:")
        info_text.append(f"  X: {np.degrees(self.rotation_angles[0]):.1f}°")
        info_text.append(f"  Y: {np.degrees(self.rotation_angles[1]):.1f}°")
        info_text.append(f"  Z: {np.degrees(self.rotation_angles[2]):.1f}°")
        info_text.append("")
        info_text.append(f"Масштаб:")
        info_text.append(f"  X: {self.scale_factors[0]:.2f}")
        info_text.append(f"  Y: {self.scale_factors[1]:.2f}")
        info_text.append(f"  Z: {self.scale_factors[2]:.2f}")
        info_text.append("")
        info_text.append(f"Освещение:")
        info_text.append(f"  Интенсивность: {self.light_intensity:.2f}")
        info_text.append(f"  Направление: {self.light_direction}")
        info_text.append("")
        info_text.append("Управление:")
        info_text.append("• Слайдеры - изменение параметров")
        info_text.append("• Сброс - вернуть исходное состояние")
        info_text.append("• Анимация - автоматическое вращение")
        
        self.ax_info.text(0.05, 0.95, "\n".join(info_text),
                         transform=self.ax_info.transAxes,
                         fontsize=9, color='white', verticalalignment='top')
    
    def update_cube(self):
        """Обновляем все отображения"""
        self.draw_cube_3d()
        self.draw_projections()
        self.draw_matrices_info()
        self.draw_info_panel()
        self.fig.canvas.draw_idle()
    
    def show(self):
        """Показываем окно"""
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,
                           wspace=0.2, hspace=0.2)
        plt.show()

# Запуск программы
if __name__ == "__main__":
    print("=" * 70)
    print("3D ВРАЩАЮЩИЙСЯ КУБ С МАТРИЧНЫМИ ПРЕОБРАЗОВАНИЯМИ")
    print("=" * 70)
    print("\nДемонстрация работы матриц в компьютерной графике:")
    print("1. Матрицы поворота вокруг осей X, Y, Z")
    print("2. Матрица масштабирования")
    print("3. Композитные преобразования")
    print("4. Проекции на плоскости XZ и YZ")
    print("5. Модель освещения граней")
    print("\nУправление:")
    print("• Слайдеры вращения: изменение углов по осям")
    print("• Слайдеры масштаба: изменение размеров по осям")
    print("• Слайдер освещения: изменение интенсивности света")
    print("• Кнопка 'Сброс': возврат к исходному состоянию")
    print("• Кнопка 'Старт/Стоп анимации': автоматическое вращение")
    print("\nМатематика:")
    print("• v' = M × v, где M = Rₓ × Rᵧ × R₂ × S")
    print("• Освещение: I = ambient + light × (n·l)")
    print("=" * 70)
    
    cube = RotatingCube3D()
    cube.show()