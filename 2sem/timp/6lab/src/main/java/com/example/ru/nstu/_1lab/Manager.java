package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;

/**
 * Класс менеджера.
 * Визуализируется красным цветом в виде прямоугольника с "документами".
 * Движением управляет ManagerAI.
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
        // Движением занимается ManagerAI в отдельном потоке.
        // Этот метод вызывается из основного потока для синхронизации,
        // но фактическое обновление позиции происходит в AI-потоке.
    }
}
