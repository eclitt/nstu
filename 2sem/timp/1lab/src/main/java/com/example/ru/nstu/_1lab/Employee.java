package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;


public abstract class Employee implements IBehaviour {
    private static int counter = 0;
    protected int id;
    protected long creationTime;
    
    // Позиция и размер объекта
    protected double x;
    protected double y;
    protected double width = 40;
    protected double height = 40;
    
    // Цвет для визуализации
    protected Color color;
    public Employee() {
        this.id = ++counter;
        this.creationTime = System.currentTimeMillis();
    }

    public int getId() {
        return id;
    }

    public long getCreationTime() {
        return creationTime;
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
     * Возвращает тип сотрудника в виде строки.
     * @return название типа (Developer, Manager)
     */
    public abstract String getType();

    @Override
    public String toString() {
        return getType() + "{id=" + id + "}";
    }
}
