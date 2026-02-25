package com.example.ru.nstu._1lab;

import javafx.animation.AnimationTimer;
import javafx.fxml.FXML;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.input.KeyCode;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;

/**
 * Контроллер для симуляции компании.
 * Управление:
 * - B - запустить симуляцию
 * - E - остановить симуляцию
 * - T - показать/скрыть время
 */
public class SimulationController {

    @FXML
    private Canvas canvas;

    private Habitat habitat;
    private AnimationTimer timer;
    private boolean showTime = true;
    private boolean simulationEnded = false;

    // Параметры для статистики
    private int totalDevelopers = 0;
    private int totalManagers = 0;
    private long finalTime = 0;

    @FXML
    public void initialize() {
        // Инициализация среды с параметрами из задания
        habitat = new Habitat(canvas.getWidth(), canvas.getHeight(), 2, 3, 0.7, 50);

        // Обработчик нажатий клавиш
        canvas.setOnKeyPressed(event -> {
            KeyCode code = event.getCode();
            if (code == KeyCode.B) {
                startSimulation();
            } else if (code == KeyCode.E) {
                stopSimulation();
            } else if (code == KeyCode.T) {
                toggleTime();
            }
        });

        // Таймер анимации
        timer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                updateAndRender();
            }
        };

        timer.start();
    }

    private void startSimulation() {
        if (!habitat.isRunning()) {
            habitat.start();
            simulationEnded = false;
            totalDevelopers = 0;
            totalManagers = 0;
        }
    }

    private void stopSimulation() {
        if (habitat.isRunning()) {
            // Сохраняем статистику перед остановкой
            totalDevelopers = habitat.getDeveloperCount();
            totalManagers = habitat.getManagerCount();
            finalTime = habitat.getElapsedTime();
            
            habitat.stop();
            simulationEnded = true;
        }
    }

    private void toggleTime() {
        showTime = !showTime;
    }

    private void updateAndRender() {
        GraphicsContext gc = canvas.getGraphicsContext2D();
        
        // Очистка холста
        gc.setFill(Color.LIGHTGRAY);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());

        if (habitat.isRunning()) {
            // Обновление состояния среды
            habitat.update(habitat.getElapsedTime());
            
            // Отрисовка времени
            if (showTime) {
                drawTime(gc);
            }
            
            // Отрисовка сотрудников
            drawEmployees(gc);
        } else if (simulationEnded) {
            // Отрисовка статистики после остановки
            drawStatistics(gc);
        } else {
            // Начальный экран
            drawStartScreen(gc);
        }
    }

    private void drawTime(GraphicsContext gc) {
        gc.setFill(Color.BLACK);
        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 14));
        long time = habitat.getElapsedTime();
        String timeText = String.format("Время: %d.%03d сек", time / 1000, time % 1000);
        gc.fillText(timeText, 10, 20);
    }

    private void drawEmployees(GraphicsContext gc) {
        for (Employee employee : habitat.getEmployees()) {
            drawEmployee(gc, employee);
        }
    }

    private void drawEmployee(GraphicsContext gc, Employee employee) {
        double x = employee.getX();
        double y = employee.getY();
        double w = employee.getWidth();
        double h = employee.getHeight();

        // Тень
        gc.setFill(Color.rgb(0, 0, 0, 0.3));
        gc.fillRoundRect(x + 3, y + 3, w, h, 10, 10);

        // Основной цвет
        gc.setFill(employee.getColor());
        gc.fillRoundRect(x, y, w, h, 10, 10);

        // Обводка
        gc.setStroke(Color.DARKGRAY);
        gc.setLineWidth(2);
        gc.strokeRoundRect(x, y, w, h, 10, 10);

        // Иконка внутри
        gc.setFill(Color.WHITE);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 12));
        
        if (employee instanceof Developer) {
            // Разработчик - символ кода "</>"
            gc.fillText("</>", x + 8, y + 25);
        } else if (employee instanceof Manager) {
            // Менеджер - символ документа
            gc.fillText("M", x + 15, y + 25);
        }

        // ID сотрудника (мелким шрифтом)
        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 8));
        gc.setFill(Color.WHITE);
        gc.fillText("#" + employee.getId(), x + 2, y + 12);
    }

    private void drawStatistics(GraphicsContext gc) {
        // Центрирование текста
        double centerX = canvas.getWidth() / 2;
        double centerY = canvas.getHeight() / 2;

        // Заголовок
        gc.setFill(Color.DARKBLUE);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 24));
        gc.fillText("Симуляция завершена", centerX - 120, centerY - 80);

        // Количество разработчиков
        gc.setFill(Color.BLUE);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 18));
        gc.fillText("Разработчиков: " + totalDevelopers, centerX - 80, centerY - 30);

        // Количество менеджеров
        gc.setFill(Color.RED);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 18));
        gc.fillText("Менеджеров: " + totalManagers, centerX - 80, centerY + 10);

        // Общее количество
        gc.setFill(Color.BLACK);
        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 16));
        gc.fillText("Всего объектов: " + (totalDevelopers + totalManagers), centerX - 70, centerY + 50);

        // Время симуляции
        gc.setFill(Color.DARKGREEN);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 16));
        long seconds = finalTime / 1000;
        long millis = finalTime % 1000;
        gc.fillText(String.format("Время симуляции: %d.%03d сек", seconds, millis), centerX - 90, centerY + 90);

        // Подсказка
        gc.setFill(Color.GRAY);
        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 12));
        gc.fillText("Нажмите B для нового запуска", centerX - 90, centerY + 130);
    }

    private void drawStartScreen(GraphicsContext gc) {
        double centerX = canvas.getWidth() / 2;
        double centerY = canvas.getHeight() / 2;

        gc.setFill(Color.DARKGRAY);
        gc.setFont(Font.font("Arial", FontWeight.BOLD, 18));
        gc.fillText("Симуляция компании", centerX - 85, centerY - 30);

        gc.setFill(Color.BLACK);
        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 14));
        gc.fillText("Нажмите B для запуска", centerX - 80, centerY + 10);

        gc.setFill(Color.GRAY);
        gc.setFont(Font.font("Arial", FontWeight.NORMAL, 11));
        gc.fillText("B - старт, E - стоп, T - время", centerX - 90, centerY + 50);
    }
}
