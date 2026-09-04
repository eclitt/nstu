package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;

/**
 * Класс менеджера.
 * Визуализируется красным цветом в виде прямоугольника с "документами".
 */
public class Manager extends Employee {
    
    public Manager() {
        super();
        this.color = Color.RED;
        this.width = 40;
        this.height = 40;
    }

    @Override
    public String getType() {
        return "Manager";
    }

    @Override
    public void update(long elapsedTime) {
        // Менеджер "управляет" - небольшая анимация
        // В данной реализации просто остаётся на месте
    }
}
