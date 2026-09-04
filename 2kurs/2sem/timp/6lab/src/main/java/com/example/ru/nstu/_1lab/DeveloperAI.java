package com.example.ru.nstu._1lab;

import java.util.List;
import java.util.Random;
import java.util.HashMap;
import java.util.Map;

/**
 * AI для разработчиков.
 * Разработчики двигаются хаотично со скоростью V.
 * Хаотичность достигается случайной сменой направления движения раз в N секунд.
 * Каждый разработчик имеет своё собственное случайное направление.
 */
public class DeveloperAI extends BaseAI {

    /** Интервал смены направления (в секундах) */
    private final int directionChangeInterval;

    /** Генератор случайных чисел */
    private final Random random = new Random();

    /** Карта: разработчик -> угол направления движения (в радианах) */
    private final Map<Employee, Double> employeeAngles = new HashMap<>();

    /** Карта: разработчик -> время последней смены направления (мс) */
    private final Map<Employee, Long> lastChangeTimes = new HashMap<>();

    /**
     * Конструктор.
     *
     * @param employees                  список разработчиков
     * @param worldWidth                 ширина рабочей области
     * @param worldHeight                высота рабочей области
     * @param directionChangeIntervalSec интервал смены направления (сек)
     */
    public DeveloperAI(List<Developer> employees, double worldWidth, double worldHeight,
                       int directionChangeIntervalSec) {
        super(employees, worldWidth, worldHeight);
        this.directionChangeInterval = directionChangeIntervalSec;
    }

    /**
     * Инициализирует случайное направление для разработчика.
     * Вызывается при добавлении нового разработчика.
     */
    public synchronized void initDeveloper(Developer developer) {
        employeeAngles.put(developer, random.nextDouble() * 2 * Math.PI);
        lastChangeTimes.put(developer, System.currentTimeMillis());
    }

    @Override
    protected void updatePositions() {
        long currentTime = System.currentTimeMillis();

        // Вычисляем смещение за прошедшее время (16 мс — примерно один кадр)
        double deltaTime = 0.016; // секунды

        // Обновляем позицию каждого разработчика
        synchronized (employees) {
            for (Employee emp : employees) {
                // Инициализация, если ещё не сделана
                if (!employeeAngles.containsKey(emp)) {
                    initDeveloper((Developer) emp);
                }

                double angle = employeeAngles.get(emp);
                long lastChange = lastChangeTimes.get(emp);

                // Проверяем, пора ли сменить направление
                long timeSinceLastChange = currentTime - lastChange;
                if (timeSinceLastChange >= directionChangeInterval * 1000L) {
                    angle = random.nextDouble() * 2 * Math.PI;
                    employeeAngles.put(emp, angle);
                    lastChangeTimes.put(emp, currentTime);
                }

                double dx = Math.cos(angle) * velocity * deltaTime;
                double dy = Math.sin(angle) * velocity * deltaTime;

                double newX = emp.getX() + dx;
                double newY = emp.getY() + dy;

                // Отскок от границ — случайная смена направления при ударе
                if (newX < 0) {
                    newX = 0;
                    angle = Math.PI - angle + (random.nextDouble() - 0.5) * 0.5;
                    employeeAngles.put(emp, angle);
                    lastChangeTimes.put(emp, currentTime);
                } else if (newX > worldWidth - emp.getWidth()) {
                    newX = worldWidth - emp.getWidth();
                    angle = Math.PI - angle + (random.nextDouble() - 0.5) * 0.5;
                    employeeAngles.put(emp, angle);
                    lastChangeTimes.put(emp, currentTime);
                }

                if (newY < 0) {
                    newY = 0;
                    angle = -angle + (random.nextDouble() - 0.5) * 0.5;
                    employeeAngles.put(emp, angle);
                    lastChangeTimes.put(emp, currentTime);
                } else if (newY > worldHeight - emp.getHeight()) {
                    newY = worldHeight - emp.getHeight();
                    angle = -angle + (random.nextDouble() - 0.5) * 0.5;
                    employeeAngles.put(emp, angle);
                    lastChangeTimes.put(emp, currentTime);
                }

                emp.setX(newX);
                emp.setY(newY);
            }
        }
    }

    @Override
    protected String getThreadName() {
        return "DeveloperAI-Thread";
    }

    /**
     * Возвращает интервал смены направления в секундах.
     */
    public int getDirectionChangeInterval() {
        return directionChangeInterval;
    }
}
