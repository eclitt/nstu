package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;

/**
 * Класс разработчика.
 * Визуализируется синим цветом в виде прямоугольника с "кодом".
 */
public class Developer extends Employee {
    
    public Developer() {
        super();
        this.color = Color.BLUE;
        this.width = 40;
        this.height = 40;
    }

    @Override
    public String getType() {
        return "Developer";
    }

    @Override
    public void update(long elapsedTime) {
        // Разработчик "пишет код" - небольшая анимация пульсации
        // В данной реализации просто остаётся на месте
    }
}
