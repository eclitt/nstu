package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;

/**
 * Абстрактный класс сотрудника компании.
 * Содержит базовые свойства для визуализации и позиционирования объекта.
 */
public abstract class Employee implements IBehaviour {
    // Счётчик для последовательных ID (используется для проверки уникальности)
    private static int counter = 0;
    
    // Уникальный случайный идентификатор
    protected int id;
    
    // Время рождения от начала симуляции (мс)
    protected long birthTime;
    
    // Время жизни объекта (мс) - через сколько объект исчезнет
    protected long lifetime;

    // Позиция и размер объекта
    protected double x;
    protected double y;
    protected double width = 40;
    protected double height = 40;

    // Цвет для визуализации
    protected Color color;

    public Employee() {
        this.id = generateUniqueId();
        this.birthTime = 0;
        this.lifetime = 0;
    }
    
    /**
     * Генерирует уникальный случайный идентификатор.
     * @return уникальный ID
     */
    private static synchronized int generateUniqueId() {
        int newId;
        do {
            newId = (int) (Math.random() * 1000000);
        } while (!Employee.idSet.add(newId));
        return newId;
    }
    
    /**
     * Устанавливает время рождения объекта от начала симуляции.
     * @param birthTime время рождения (мс от начала симуляции)
     */
    public void setBirthTime(long birthTime) {
        this.birthTime = birthTime;
    }
    
    /**
     * Устанавливает время жизни объекта.
     * @param lifetime время жизни (мс)
     */
    public void setLifetime(long lifetime) {
        this.lifetime = lifetime;
    }

    public int getId() {
        return id;
    }

    public long getBirthTime() {
        return birthTime;
    }

    public long getLifetime() {
        return lifetime;
    }
    
    /**
     * Проверяет, истекло ли время жизни объекта.
     * @param currentTime текущее время симуляции (мс)
     * @return true, если время жизни истекло
     */
    public boolean isExpired(long currentTime) {
        return lifetime > 0 && (currentTime - birthTime) >= lifetime;
    }

    public double getX() {
        return x;
    }

    public void setX(double x) {
        this.x = x;
    }

    public double getY() {
        return y;
    }

    public void setY(double y) {
        this.y = y;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }

    public Color getColor() {
        return color;
    }

    /**
     * Сбрасывает счётчик ID и коллекцию использованных ID. Вызывается при перезапуске симуляции.
     */
    public static void resetCounter() {
        counter = 0;
        idSet.clear();
    }
    
    /**
     * TreeSet для хранения уникальных идентификаторов.
     */
    public static java.util.Set<Integer> idSet = new java.util.TreeSet<>();

    /**
     * Возвращает тип сотрудника в виде строки.
     * @return название типа (Developer, Manager)
     */
    public abstract String getType();

    @Override
    public String toString() {
        return getType() + "{id=" + id + "}";
    }
}
