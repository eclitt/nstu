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

import java.util.ArrayList;
import java.util.List;

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

    @FXML
    private CheckBox showInfoCheckBox;

    @FXML
    private TextField n1Field;

    @FXML
    private TextField n2Field;

    @FXML
    private TextField n1FieldRight;

    @FXML
    private TextField n2FieldRight;

    @FXML
    private ComboBox<String> probabilityComboBox;

    @FXML
    private ComboBox<String> probabilityComboBoxRight;

    @FXML
    private ListView<String> probabilityListView;

    // Элементы меню
    @FXML
    private CheckMenuItem menuTimeOn;

    @FXML
    private CheckMenuItem menuTimeOff;

    @FXML
    private CheckMenuItem menuShowInfo;


    private Habitat habitat;
    private AnimationTimer timer;
    private boolean showTime = true;
    private boolean showInfoDialog = true;
    private boolean simulationEnded = false;
    private boolean waitForOkCancel = false;

    // Параметры для статистики
    private int totalDevelopers = 0;
    private int totalManagers = 0;
    private long finalTime = 0;

    // Значения вероятностей
    private final List<Double> probabilityValues = List.of(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
    private final List<String> probabilityStrings = List.of("10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%");

    // Значения по умолчанию
    private static final int DEFAULT_N1 = 2;
    private static final int DEFAULT_N2 = 3;
    private static final double DEFAULT_P1 = 0.7;

    @FXML
    public void initialize() {
        // Настройка пропорций SplitPane (60% слева, 40% справа)
        splitPane.setDividerPositions(0.6);

        // Инициализация ComboBox и ListView вероятностей
        initializeProbabilityControls();

        // Инициализация среды будет выполнена при старте симуляции
        habitat = null;
        
        // Кнопки правой панели
        startbtn.setDisable(false);
        stopbtn.setDisable(true);
        timeon.setSelected(true);
        timeoff.setSelected(false);
        showInfoCheckBox.setSelected(true);
        
        // Элементы меню
        menuTimeOn.setSelected(true);
        menuTimeOff.setSelected(false);
        menuShowInfo.setSelected(true);

        // Обработчик нажатий клавиш
        canvas.setOnKeyPressed(event -> {
            KeyCode code = event.getCode();
            if (code == KeyCode.B) {
                startSimulation();
                startbtn.setDisable(true);
                stopbtn.setDisable(false);
            } else if (code == KeyCode.E) {
                stopSimulationWithDialog();
            } else if (code == KeyCode.T) {
                toggleTimeDisplay();
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

    /**
     * Инициализация ComboBox и ListView для выбора вероятности
     */
    private void initializeProbabilityControls() {
        // Заполнение ComboBox
        probabilityComboBox.getItems().addAll(probabilityStrings);
        probabilityComboBox.setValue("70%");
        probabilityComboBox.setOnAction(e -> {
            int index = probabilityComboBox.getSelectionModel().getSelectedIndex();
            if (index >= 0 && index < probabilityListView.getItems().size()) {
                probabilityListView.getSelectionModel().select(index);
            }
            probabilityComboBoxRight.setValue(probabilityComboBox.getValue());
        });

        // Заполнение ComboBoxRight
        probabilityComboBoxRight.getItems().addAll(probabilityStrings);
        probabilityComboBoxRight.setValue("70%");
        probabilityComboBoxRight.setOnAction(e -> {
            int index = probabilityComboBoxRight.getSelectionModel().getSelectedIndex();
            if (index >= 0) {
                probabilityComboBox.setValue(probabilityStrings.get(index));
                probabilityListView.getSelectionModel().select(index);
            }
        });

        // Заполнение ListView
        probabilityListView.getItems().addAll(probabilityStrings);
        probabilityListView.getSelectionModel().select(6); // 70% по умолчанию
        probabilityListView.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                int index = probabilityStrings.indexOf(newVal);
                if (index >= 0) {
                    probabilityComboBox.setValue(newVal);
                    probabilityComboBoxRight.setValue(newVal);
                }
            }
        });
    }

    /**
     * Получение текущих параметров симуляции из полей ввода с валидацией
     */
    private SimulationSettings getSimulationSettings() {
        // Используем поля из правой панели как основные
        int n1 = parsePeriod(n1FieldRight, "Период рождения разработчиков");
        int n2 = parsePeriod(n2FieldRight, "Период рождения менеджеров");
        double p1 = getProbabilityFromControls();
        
        return new SimulationSettings(n1, n2, p1, 50);
    }

    /**
     * Парсинг периода из TextField с валидацией
     */
    private int parsePeriod(TextField field, String fieldName) {
        try {
            String text = field.getText().trim();
            if (text.isEmpty()) {
                throw new NumberFormatException("Поле пустое");
            }
            int value = Integer.parseInt(text);
            if (value <= 0) {
                throw new NumberFormatException("Значение должно быть больше 0");
            }
            if (value > 60) {
                throw new NumberFormatException("Значение не должно превышать 60");
            }
            return value;
        } catch (NumberFormatException e) {
            showErrorDialog(fieldName, "Должно быть целое число от 1 до 60");
            field.setText(String.valueOf(fieldName.contains("разработчиков") ? DEFAULT_N1 : DEFAULT_N2));
            return fieldName.contains("разработчиков") ? DEFAULT_N1 : DEFAULT_N2;
        }
    }

    /**
     * Получение вероятности из ComboBox/ListView
     */
    private double getProbabilityFromControls() {
        String selected = probabilityComboBoxRight.getValue();
        if (selected == null) {
            selected = probabilityComboBox.getValue();
        }
        if (selected == null && !probabilityListView.getSelectionModel().getSelectedItems().isEmpty()) {
            selected = probabilityListView.getSelectionModel().getSelectedItem();
        }
        if (selected != null) {
            int index = probabilityStrings.indexOf(selected);
            if (index >= 0) {
                return probabilityValues.get(index);
            }
        }
        return DEFAULT_P1;
    }

    /**
     * Показывает диалог ошибки валидации
     */
    private void showErrorDialog(String fieldName, String message) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Ошибка ввода");
        alert.setHeaderText("Неверное значение в поле \"" + fieldName + "\"");
        alert.setContentText(message + "\n\nБудет установлено значение по умолчанию.");
        alert.initOwner(canvas.getScene().getWindow());
        alert.showAndWait();
    }

    // Методы для кнопок в правой панели
    @FXML
    private void startSimulationButton() {
        startSimulation();
        startbtn.setDisable(true);
        stopbtn.setDisable(false);
        canvas.requestFocus();
    }

    @FXML
    private void stopSimulationButton() {
        stopSimulationWithDialog();
        canvas.requestFocus();
    }

    @FXML
    private void chkShowTime() {
        showTime = true;
        timeon.setSelected(true);
        menuTimeOn.setSelected(true);
        timeoff.setSelected(false);
        menuTimeOff.setSelected(false);
        updateAndRender();
        canvas.requestFocus();
    }

    @FXML
    private void chkHideTime() {
        showTime = false;
        timeoff.setSelected(true);
        menuTimeOff.setSelected(true);
        timeon.setSelected(false);
        menuTimeOn.setSelected(false);
        updateAndRender();
        canvas.requestFocus();
    }

    @FXML
    private void chkShowInfo() {
        showInfoDialog = menuShowInfo.isSelected();
        showInfoCheckBox.setSelected(showInfoDialog);
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

        Label versionLabel = new Label("Версия 2.0");
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

    private void startSimulation() {
        // Получаем и применяем настройки
        SimulationSettings settings = getSimulationSettings();
        
        if (habitat == null) {
            habitat = Habitat.getInstance(canvas.getWidth(), canvas.getHeight(), 
                    settings.n1, settings.n2, settings.p1, settings.kPercent);
        } else if (!habitat.isRunning()) {
            Habitat.resetInstance();
            habitat = Habitat.getInstance(canvas.getWidth(), canvas.getHeight(),
                    settings.n1, settings.n2, settings.p1, settings.kPercent);
        }
        
        if (!habitat.isRunning()) {
            Employee.resetCounter();
            habitat.start();
            simulationEnded = false;
            waitForOkCancel = false;
            totalDevelopers = 0;
            totalManagers = 0;
            timer.start();
        }
    }

    /**
     * Остановка симуляции с диалогом подтверждения (если включено)
     */
    private void stopSimulationWithDialog() {
        if (habitat != null && habitat.isRunning()) {
            // Сохраняем статистику
            totalDevelopers = habitat.getDeveloperCount();
            totalManagers = habitat.getManagerCount();
            finalTime = habitat.getElapsedTime();

            if (showInfoDialog) {
                // Показываем модальное окно с TextArea
                showStopConfirmationDialog();
            } else {
                // Останавливаем сразу
                stopSimulation();
                startbtn.setDisable(false);
                stopbtn.setDisable(true);
            }
        }
    }

    /**
     * Показывает модальное окно подтверждения остановки с TextArea
     */
    private void showStopConfirmationDialog() {
        Stage dialogStage = new Stage();
        dialogStage.setTitle("Остановка симуляции");
        dialogStage.initModality(Modality.WINDOW_MODAL);
        dialogStage.initOwner(canvas.getScene().getWindow());

        VBox vbox = new VBox(10);
        vbox.setPadding(new Insets(15));

        Label titleLabel = new Label("Статистика симуляции");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 16));

        TextArea textArea = new TextArea();
        textArea.setPrefRowCount(10);
        textArea.setPrefColumnCount(40);
        textArea.setEditable(false);
        textArea.setWrapText(true);

        // Формируем текст статистики
        StringBuilder stats = new StringBuilder();
        stats.append("=== РЕЗУЛЬТАТЫ СИМУЛЯЦИИ ===\n\n");
        stats.append("Время симуляции: ").append(finalTime / 1000).append(".")
             .append(String.format("%03d", finalTime % 1000)).append(" сек\n\n");
        stats.append("Сгенерировано объектов:\n");
        stats.append("  - Разработчиков (Developer): ").append(totalDevelopers).append(" шт.\n");
        stats.append("  - Менеджеров (Manager): ").append(totalManagers).append(" шт.\n");
        stats.append("  - Всего: ").append(totalDevelopers + totalManagers).append(" шт.\n\n");
        
        if (totalDevelopers + totalManagers > 0) {
            double devPercent = (totalDevelopers * 100.0) / (totalDevelopers + totalManagers);
            double mgrPercent = (totalManagers * 100.0) / (totalDevelopers + totalManagers);
            stats.append(String.format("Процент разработчиков: %.1f%%\n", devPercent));
            stats.append(String.format("Процент менеджеров: %.1f%%\n", mgrPercent));
        }

        textArea.setText(stats.toString());

        HBox buttonBox = new HBox(10);
        buttonBox.setAlignment(Pos.CENTER_RIGHT);

        Button okButton = new Button("OK");
        okButton.setPrefWidth(80);
        okButton.setOnAction(e -> {
            stopSimulation();
            startbtn.setDisable(false);
            stopbtn.setDisable(true);
            simulationEnded = true;
            dialogStage.close();
        });

        Button cancelButton = new Button("Отмена");
        cancelButton.setPrefWidth(80);
        cancelButton.setOnAction(e -> {
            // Продолжаем симуляцию
            if (habitat != null && !habitat.isRunning()) {
                habitat.start();
                timer.start();
            }
            dialogStage.close();
        });

        buttonBox.getChildren().addAll(okButton, cancelButton);
        vbox.getChildren().addAll(titleLabel, textArea, buttonBox);

        Scene scene = new Scene(vbox);
        dialogStage.setScene(scene);
        dialogStage.setResizable(false);
        
        // Блокируем симуляцию на время диалога
        if (timer != null) {
            timer.stop();
        }
        waitForOkCancel = true;
        
        dialogStage.showAndWait();
    }

    private void stopSimulation() {
        if (habitat != null && habitat.isRunning()) {
            habitat.stop();
            simulationEnded = true;
            timer.stop();
            updateAndRender();
        }
    }

    private void toggleTimeDisplay() {
        if (showTime) {
            chkHideTime();
        } else {
            chkShowTime();
        }
    }

    /**
     * Обновление статистики в правой панели
     */
    private void updateStatistics() {
        if (habitat != null && habitat.isRunning()) {
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
        if (waitForOkCancel) {
            return; // Не обновляем во время ожидания OK/Cancel
        }

        GraphicsContext gc = canvas.getGraphicsContext2D();

        // Очистка холста
        gc.setFill(Color.LIGHTGRAY);
        gc.fillRect(0, 0, canvas.getWidth(), canvas.getHeight());

        if (habitat != null && habitat.isRunning()) {
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
        if (habitat != null && habitat.isRunning()) {
            long time = habitat.getElapsedTime();
            String timeText = String.format("Время: %d.%03d сек", time / 1000, time % 1000);
            gc.fillText(timeText, 10, 20);
        }
    }

    private void drawEmployees(GraphicsContext gc) {
        if (habitat == null) return;
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

    /**
     * Внутренний класс для хранения настроек симуляции
     */
    private static class SimulationSettings {
        int n1, n2, kPercent;
        double p1;

        SimulationSettings(int n1, int n2, double p1, int kPercent) {
            this.n1 = n1;
            this.n2 = n2;
            this.p1 = p1;
            this.kPercent = kPercent;
        }
    }
}
