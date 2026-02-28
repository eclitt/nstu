package com.example.ru.nstu._1lab;

import javafx.animation.AnimationTimer;
import javafx.fxml.FXML;
import javafx.geometry.Insets;
import javafx.geometry.Pos;
import javafx.scene.Scene;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.control.*;
import javafx.scene.input.KeyCode;
import javafx.scene.layout.HBox;
import javafx.scene.layout.VBox;
import javafx.scene.paint.Color;
import javafx.scene.text.Font;
import javafx.scene.text.FontWeight;
import javafx.stage.Modality;
import javafx.stage.Stage;

/**
 * Контроллер для симуляции компании.
 * Управление:
 * - B - запустить симуляцию
 * - E - остановить симуляцию
 * - T - показать/скрыть время
 */
public class SimulationController {

    @FXML
    private SplitPane splitPane;

    @FXML
    private Canvas canvas;

    @FXML
    private Label timeLabel;

    @FXML
    private Label developersLabel;

    @FXML
    private Label managersLabel;

    @FXML
    private Label totalLabel;

    @FXML
    private Button startbtn;

    @FXML
    private Button stopbtn;

    @FXML
    private CheckBox timeon;

    @FXML
    private CheckBox timeoff;

    // Элементы меню
    @FXML
    private CheckMenuItem menuTimeOn;

    @FXML
    private CheckMenuItem menuTimeOff;


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
        // Настройка пропорций SplitPane (60% слева, 40% справа)
        splitPane.setDividerPositions(0.6);

        // Инициализация среды будет выполнена при старте симуляции
        // для получения корректных размеров Canvas после рендеринга
        habitat = null;
        // Кнопки правой панели
        startbtn.setDisable(false);
        stopbtn.setDisable(true);
        timeon.setSelected(true);
        timeoff.setSelected(false);
        // Элементы меню
        menuTimeOn.setSelected(true);
        menuTimeOff.setSelected(false);

        // Обработчик нажатий клавиш
        canvas.setOnKeyPressed(event -> {
            KeyCode code = event.getCode();
            if (code == KeyCode.B) {
                startSimulation();
                startbtn.setDisable(true);
                stopbtn.setDisable(false);
            } else if (code == KeyCode.E) {
                stopSimulation();
                startbtn.setDisable(false);
                stopbtn.setDisable(true);
            } else if (code == KeyCode.T) {
                if (showTime) { showTime = false; timeon.setSelected(false); timeoff.setSelected(true);}
                else { showTime = true; timeon.setSelected(true); timeoff.setSelected(false);}

            } else if (code == KeyCode.I) {
                showInfoDialog();
            }
        });

        // Таймер анимации
        timer = new AnimationTimer() {
            @Override
            public void handle(long now) {
                updateAndRender();
                updateStatistics();
            }
        };
    }

    // Методы для кнопок в правой панели
    @FXML
    private void startSimulationButton() {
        startSimulation();
        // Синхронизация состояния кнопок
        startbtn.setDisable(true);
        stopbtn.setDisable(false);
        canvas.requestFocus();
    }

    @FXML
    private void stopSimulationButton() {
        stopSimulation();
        // Синхронизация состояния кнопок
        stopbtn.setDisable(true);
        startbtn.setDisable(false);
        canvas.requestFocus();
    }

    @FXML
    private void chkShowTime() {
        onTime();
        timeon.setSelected(true);
        menuTimeOn.setSelected(true);
        timeoff.setSelected(false);
        menuTimeOff.setSelected(false);
        // Принудительная перерисовка
        updateAndRender();
        canvas.requestFocus();
    }

    @FXML
    private void chkHideTime() {
        offTime();
        timeoff.setSelected(true);
        menuTimeOff.setSelected(true);
        timeon.setSelected(false);
        menuTimeOn.setSelected(false);
        // Принудительная перерисовка
        updateAndRender();
        canvas.requestFocus();
    }

    @FXML
    private void exitApplication() {
        Stage stage = (Stage) canvas.getScene().getWindow();
        stage.close();
    }

    @FXML
    private void showAboutDialog() {
        Stage dialogStage = new Stage();
        dialogStage.setTitle("О программе");
        dialogStage.initModality(Modality.WINDOW_MODAL);
        dialogStage.initOwner(canvas.getScene().getWindow());

        VBox vbox = new VBox(15);
        vbox.setPadding(new Insets(20));
        vbox.setStyle("-fx-background-color: white;");

        Label titleLabel = new Label("Симуляция компании");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 18));
        titleLabel.setTextFill(Color.DARKBLUE);

        Label versionLabel = new Label("Версия 1.0");
        versionLabel.setFont(Font.font("Arial", FontWeight.NORMAL, 14));

        Label descriptionLabel = new Label("Лабораторная работа №2 по ТИМП");
        descriptionLabel.setFont(Font.font("Arial", FontWeight.NORMAL, 12));
        descriptionLabel.setTextFill(Color.GRAY);

        Button closeBtn = new Button("Закрыть");
        closeBtn.setPrefWidth(100);
        closeBtn.setOnAction(e -> dialogStage.close());

        HBox buttonBox = new HBox(closeBtn);
        buttonBox.setAlignment(Pos.CENTER);

        vbox.getChildren().addAll(titleLabel, versionLabel, descriptionLabel, new Separator(), buttonBox);

        Scene scene = new Scene(vbox);
        dialogStage.setScene(scene);
        dialogStage.setResizable(false);
        dialogStage.showAndWait();
    }

    @FXML
    private void showInfoButton() {
        showInfoDialog();
        canvas.requestFocus();
    }

    private void startSimulation() {
        if (habitat == null) {
            // Инициализируем среду с актуальными размерами Canvas (Singleton)
            habitat = Habitat.getInstance(canvas.getWidth(), canvas.getHeight(), 2, 3, 0.7, 50);
        } else if (!habitat.isRunning()) {
            // Сбрасываем Singleton при новом запуске после остановки
            Habitat.resetInstance();
            habitat = Habitat.getInstance(canvas.getWidth(), canvas.getHeight(), 2, 3, 0.7, 50);
        }
        if (!habitat.isRunning()) {
            Employee.resetCounter();
            habitat.start();
            simulationEnded = false;
            totalDevelopers = 0;
            totalManagers = 0;
            timer.start();
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
            timer.stop();
            // Принудительная перерисовка для отображения статистики
            updateAndRender();
        }
    }

    private void offTime() {
        showTime = false;
    }
    private void onTime() {
        showTime = true;
    }


    /**
     * Показывает модальное окно с информацией о симуляции.
     */
    private void showInfoDialog() {
        // Получаем актуальные значения
        int devs = habitat != null ? habitat.getDeveloperCount() : 0;
        int mgrs = habitat != null ? habitat.getManagerCount() : 0;
        long time = habitat != null ? habitat.getElapsedTime() : finalTime;

        if (simulationEnded) {
            devs = totalDevelopers;
            mgrs = totalManagers;
            time = finalTime;
        }

        // Создаём модальное окно
        Stage dialogStage = new Stage();
        dialogStage.setTitle("Информация о симуляции");
        dialogStage.initModality(Modality.WINDOW_MODAL);
        dialogStage.initOwner(canvas.getScene().getWindow());

        VBox vbox = new VBox(15);
        vbox.setPadding(new Insets(20));
        vbox.setStyle("-fx-background-color: white;");

        Label titleLabel = new Label("Статистика симуляции");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 18));
        titleLabel.setTextFill(Color.DARKBLUE);

        Label devLabel = new Label("Разработчиков: " + devs);
        devLabel.setFont(Font.font("Arial", FontWeight.NORMAL, 14));
        devLabel.setTextFill(Color.BLUE);

        Label mgrLabel = new Label("Менеджеров: " + mgrs);
        mgrLabel.setFont(Font.font("Arial", FontWeight.NORMAL, 14));
        mgrLabel.setTextFill(Color.RED);

        Label totalLabelInfo = new Label("Всего объектов: " + (devs + mgrs));
        totalLabelInfo.setFont(Font.font("Arial", FontWeight.NORMAL, 14));

        long seconds = time / 1000;
        long millis = time % 1000;
        Label timeLabelInfo = new Label(String.format("Время симуляции: %d.%03d сек", seconds, millis));
        timeLabelInfo.setFont(Font.font("Arial", FontWeight.BOLD, 14));
        timeLabelInfo.setTextFill(Color.DARKGREEN);

        Button closeBtn = new Button("Закрыть");
        closeBtn.setPrefWidth(100);
        closeBtn.setOnAction(e -> dialogStage.close());

        HBox buttonBox = new HBox(closeBtn);
        buttonBox.setAlignment(Pos.CENTER);

        vbox.getChildren().addAll(titleLabel, devLabel, mgrLabel, totalLabelInfo, new Separator(), buttonBox);

        Scene scene = new Scene(vbox);
        dialogStage.setScene(scene);
        dialogStage.setResizable(false);
        dialogStage.showAndWait();
    }

    /**
     * Обновление статистики в правой панели
     */
    private void updateStatistics() {
        if (habitat.isRunning()) {
            long time = habitat.getElapsedTime();
            timeLabel.setText(String.format("Время: %d.%03d сек", time / 1000, time % 1000));
            developersLabel.setText("Разработчиков: " + habitat.getDeveloperCount());
            managersLabel.setText("Менеджеров: " + habitat.getManagerCount());
            totalLabel.setText("Всего: " + (habitat.getDeveloperCount() + habitat.getManagerCount()));
        } else if (simulationEnded) {
            timeLabel.setText(String.format("Время: %d.%03d сек", finalTime / 1000, finalTime % 1000));
            developersLabel.setText("Разработчиков: " + totalDevelopers);
            managersLabel.setText("Менеджеров: " + totalManagers);
            totalLabel.setText("Всего: " + (totalDevelopers + totalManagers));
        }
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
