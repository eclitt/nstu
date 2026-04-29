package com.example.ru.nstu._1lab;

import javafx.scene.paint.Color;

/**
 * Класс разработчика.
 * Визуализируется синим цветом в виде прямоугольника с "кодом".
 * Движением управляет DeveloperAI.
 */
public class Developer extends Employee {

    public Developer() {
        super();
        setColor(Color.BLUE);
        this.width = 40;
        this.height = 40;
    }

    @Override
    public String getType() {
        return "Developer";
    }

    @Override
    public void update(long elapsedTime) {
        // Движением занимается DeveloperAI в отдельном потоке.
        // Этот метод вызывается из основного потока для синхронизации,
        // но фактическое обновление позиции происходит в AI-потоке.
    }
}
