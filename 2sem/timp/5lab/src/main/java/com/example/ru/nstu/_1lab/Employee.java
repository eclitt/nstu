package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;
import java.io.Serializable;
import java.util.HashSet;
import java.util.Set;

/**
 * Абстрактный класс сотрудника компании.
 * Содержит базовые свойства для визуализации и позиционирования объекта.
 */
public abstract class Employee implements IBehaviour, Serializable {
    // Счётчик для последовательных ID (используется для проверки уникальности)
    protected static int counter = 0;
    public static Set<Integer> idSet = new HashSet<>();
    
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
    protected transient Color color;
    protected double red, green, blue, opacity;

    public Employee() {
        this.id = generateUniqueId();
        this.birthTime = 0;
        this.lifetime = 0;
    }

    protected void setColor(Color color) {
        this.color = color;
        this.red = color.getRed();
        this.green = color.getGreen();
        this.blue = color.getBlue();
        this.opacity = color.getOpacity();
    }

    public Color getColor() {
        if (color == null) {
            color = new Color(red, green, blue, opacity);
        }
        return color;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Employee employee = (Employee) obj;
        return id == employee.id;
    }

    @Override
    public int hashCode() {
        return Integer.hashCode(id);
    }

    /**
     * Сбрасывает счетчик ID и множество идентификаторов.
     */
    public static synchronized void resetCounter() {
        counter = 0;
        idSet.clear();
    }

    /**
     * Сбрасывает множество идентификаторов.
     */
    public static synchronized void clearIdSet() {
        idSet.clear();
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
