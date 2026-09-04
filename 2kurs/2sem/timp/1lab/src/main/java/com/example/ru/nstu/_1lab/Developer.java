package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;


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
        // мб анимация

    }
}
