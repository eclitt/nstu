package com.example.ru.nstu._1lab;

import java.util.List;
import java.util.HashMap;
import java.util.Map;

/**
 * AI для менеджеров.
 * Менеджеры двигаются по окружности с радиусом R со скоростью V.
 */
public class ManagerAI extends BaseAI {

    /** Радиус орбиты */
    private final double orbitRadius;

    /** Карта: менеджер -> угол на орбите */
    private final Map<Employee, Double> employeeAngles = new HashMap<>();

    /** Карта: менеджер -> центр орбиты (x, y) */
    private final Map<Employee, double[]> orbitCenters = new HashMap<>();

    /**
     * Конструктор.
     *
     * @param employees    список менеджеров
     * @param worldWidth   ширина рабочей области
     * @param worldHeight  высота рабочей области
     * @param orbitRadius  радиус орбиты
     */
    public ManagerAI(List<Manager> employees, double worldWidth, double worldHeight,
                     double orbitRadius) {
        super(employees, worldWidth, worldHeight);
        this.orbitRadius = orbitRadius;

        // Инициализируем начальные позиции и углы
        synchronized (employees) {
            for (Manager manager : employees) {
                // Центр орбиты — случайная точка, чтобы орбита помещалась в мир
                double centerX = orbitRadius + Math.random() * (worldWidth - 2 * orbitRadius);
                double centerY = orbitRadius + Math.random() * (worldHeight - 2 * orbitRadius);
                orbitCenters.put(manager, new double[]{centerX, centerY});
                employeeAngles.put(manager, Math.random() * 2 * Math.PI);

                // Устанавливаем начальную позицию на орбите
                manager.setX(centerX + orbitRadius);
                manager.setY(centerY);
            }
        }
    }

    /**
     * Инициализирует параметры орбиты для менеджера.
     */
    public synchronized void initManager(Manager manager) {
        double centerX = orbitRadius + Math.random() * (worldWidth - 2 * orbitRadius);
        double centerY = orbitRadius + Math.random() * (worldHeight - 2 * orbitRadius);
        orbitCenters.put(manager, new double[]{centerX, centerY});
        employeeAngles.put(manager, Math.random() * 2 * Math.PI);
    }

    @Override
    protected void updatePositions() {
        // Вычисляем угловую скорость: ω = V / R (рад/с)
        double angularVelocity = velocity / orbitRadius;
        double deltaTime = 0.016; // ~60 FPS в секундах
        double deltaAngle = angularVelocity * deltaTime;

        synchronized (employees) {
            for (Employee emp : employees) {
                // Инициализация, если менеджер новый
                if (!orbitCenters.containsKey(emp)) {
                    initManager((Manager) emp);
                }

                double[] center = orbitCenters.get(emp);
                if (center == null) {
                    continue;
                }

                double angle = employeeAngles.getOrDefault(emp, 0.0);
                angle += deltaAngle;

                // Полное вращение — нормализуем угол
                if (angle > 2 * Math.PI) {
                    angle -= 2 * Math.PI;
                }

                // Обновляем позицию по формуле окружности
                double newX = center[0] + orbitRadius * Math.cos(angle);
                double newY = center[1] + orbitRadius * Math.sin(angle);

                // Проверка границ — если орбита выходит за пределы, сдвигаем центр
                if (newX < 0 || newX > worldWidth - emp.getWidth() ||
                    newY < 0 || newY > worldHeight - emp.getHeight()) {
                    // Рассчитываем допустимые границы для центра орбиты
                    double minX = orbitRadius;
                    double maxX = Math.max(minX + 1, worldWidth - orbitRadius);
                    double minY = orbitRadius;
                    double maxY = Math.max(minY + 1, worldHeight - orbitRadius);
                    // Пересчитываем центр
                    center[0] = minX + Math.random() * (maxX - minX);
                    center[1] = minY + Math.random() * (maxY - minY);
                }

                newX = center[0] + orbitRadius * Math.cos(angle);
                newY = center[1] + orbitRadius * Math.sin(angle);

                // Финальная проверка границ
                newX = Math.max(0, Math.min(worldWidth - emp.getWidth(), newX));
                newY = Math.max(0, Math.min(worldHeight - emp.getHeight(), newY));

                emp.setX(newX);
                emp.setY(newY);

                employeeAngles.put(emp, angle);
            }
        }
    }

    @Override
    protected String getThreadName() {
        return "ManagerAI-Thread";
    }

    /**
     * Возвращает радиус орбиты.
     */
    public double getOrbitRadius() {
        return orbitRadius;
    }
}
