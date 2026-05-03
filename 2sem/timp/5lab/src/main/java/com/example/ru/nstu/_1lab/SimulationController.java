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
import javafx.application.Platform;
import javafx.stage.FileChooser;
import javafx.stage.Modality;
import javafx.stage.Stage;
import java.io.*;
import java.util.*;

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
    private RadioButton timeon;

    @FXML
    private RadioButton timeoff;

    @FXML
    private CheckBox showInfoCheckBox;

    @FXML
    private TextField n1FieldRight;

    @FXML
    private TextField n2FieldRight;

    @FXML
    private TextField developerLifetimeField;

    @FXML
    private TextField managerLifetimeField;

    @FXML
    private ComboBox<String> probabilityComboBoxRight;

    @FXML
    private ListView<String> probabilityListView;

    // Элементы управления AI потоками
    @FXML
    private Button pauseDeveloperAI;

    @FXML
    private Button resumeDeveloperAI;

    @FXML
    private Button pauseManagerAI;

    @FXML
    private Button resumeManagerAI;

    @FXML
    private ComboBox<String> developerPriorityComboBox;

    @FXML
    private ComboBox<String> managerPriorityComboBox;

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
    ToggleGroup toggleGroup = new ToggleGroup();
    // Параметры для статистики
    private int totalDevelopers = 0;
    private int totalManagers = 0;
    private long finalTime = 0;

    private ConsoleWindow consoleWindow;

    // Значения вероятностей
    private final List<Double> probabilityValues = List.of(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0);
    private final List<String> probabilityStrings = List.of("10%", "20%", "30%", "40%", "50%", "60%", "70%", "80%", "90%", "100%");

    // Значения приоритетов потоков (1-10)
    private final List<String> priorityStrings = List.of("1", "2", "3", "4", "5", "6", "7", "8", "9", "10");

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

        // Инициализация приоритетов AI потоков
        initializePriorityControls();

        // Инициализация среды будет выполнена при старте симуляции
        habitat = null;
        
        // Кнопки правой панели
        startbtn.setDisable(false);
        stopbtn.setDisable(true);

        // Группа для переключателей времени

        timeon.setToggleGroup(toggleGroup);
        timeoff.setToggleGroup(toggleGroup);
        timeon.setSelected(true);

        toggleGroup.selectedToggleProperty().addListener((obs, old, newVal) -> {
            boolean newShowTime = (newVal == timeon);
            if (this.showTime != newShowTime) {
                this.showTime = newShowTime;
                updateAndRender();
                canvas.requestFocus();
            }
        });





        showInfoCheckBox.setSelected(true);

        // Элементы меню
        menuTimeOn.setSelected(true);
        menuTimeOff.setSelected(false);
        menuShowInfo.setSelected(true);

        // Глобальный обработчик клавиш на сцене (работает независимо от фокуса)
        canvas.sceneProperty().addListener((obs, oldScene, newScene) -> {
            if (newScene != null) {
                newScene.addEventFilter(javafx.scene.input.KeyEvent.KEY_PRESSED, event -> {
                    KeyCode code = event.getCode();
                    if (code == KeyCode.B) {
                        startSimulation();
                        startbtn.setDisable(true);
                        stopbtn.setDisable(false);
                        event.consume();
                    } else if (code == KeyCode.E) {
                        stopSimulationWithDialog();
                        event.consume();
                    } else if (code == KeyCode.T) {
                        toggleTimeDisplay();
                        event.consume();
                    } else if (code == KeyCode.I) {
                        showInfoDialog();
                        event.consume();
                    } else if (code == KeyCode.S) {
                        saveSimulation();
                        event.consume();
                    } else if (code == KeyCode.L) {
                        loadSimulation();
                        event.consume();
                    }
                });
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

        // Загрузка конфигурации при запуске
        loadConfig();
    }

    /**
     * Инициализация ComboBox и ListView для выбора вероятности
     */
    private void initializeProbabilityControls() {
        // Заполнение ComboBoxRight
        probabilityComboBoxRight.getItems().addAll(probabilityStrings);
        probabilityComboBoxRight.setValue("70%");
        probabilityComboBoxRight.setOnAction(e -> {
            int index = probabilityComboBoxRight.getSelectionModel().getSelectedIndex();
            if (index >= 0 && index < probabilityListView.getItems().size()) {
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
                    probabilityComboBoxRight.setValue(newVal);
                }
            }
        });
    }

    /**
     * Инициализация ComboBox для выбора приоритета AI потоков
     */
    private void initializePriorityControls() {
        // Заполнение ComboBox для приоритетов
        developerPriorityComboBox.getItems().addAll(priorityStrings);
        developerPriorityComboBox.setValue("5"); // Нормальный приоритет

        managerPriorityComboBox.getItems().addAll(priorityStrings);
        managerPriorityComboBox.setValue("5");

        // Обработчики изменения приоритета
        developerPriorityComboBox.setOnAction(e -> {
            if (habitat != null && habitat.getDeveloperAI() != null) {
                String selected = developerPriorityComboBox.getValue();
                if (selected != null) {
                    int priority = Integer.parseInt(selected);
                    habitat.getDeveloperAI().setThreadPriority(priority);
                }
            }
        });

        managerPriorityComboBox.setOnAction(e -> {
            if (habitat != null && habitat.getManagerAI() != null) {
                String selected = managerPriorityComboBox.getValue();
                if (selected != null) {
                    int priority = Integer.parseInt(selected);
                    habitat.getManagerAI().setThreadPriority(priority);
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
        long developerLifetime = parseLifetime(developerLifetimeField, "Время жизни разработчиков");
        long managerLifetime = parseLifetime(managerLifetimeField, "Время жизни менеджеров");

        return new SimulationSettings(n1, n2, p1, 50, developerLifetime, managerLifetime);
    }
    
    /**
     * Парсинг времени жизни из TextField с валидацией
     */
    private long parseLifetime(TextField field, String fieldName) {
        try {
            String text = field.getText().trim();
            if (text.isEmpty()) {
                throw new NumberFormatException("Поле пустое");
            }
            int value = Integer.parseInt(text);
            if (value <= 0) {
                throw new NumberFormatException("Значение должно быть больше 0");
            }
            if (value > 300) {
                throw new NumberFormatException("Значение не должно превышать 300 секунд");
            }
            return value * 1000L; // Конвертируем секунды в миллисекунды
        } catch (NumberFormatException e) {
            showErrorDialog(fieldName, "Должно быть целое число от 1 до 300 секунд");
            field.setText(String.valueOf(fieldName.contains("разработчиков") ? "30" : "25"));
            return fieldName.contains("разработчиков") ? 30000L : 25000L;
        }
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
    private void chkShowInfo() {
        showInfoDialog = menuShowInfo.isSelected();
        showInfoCheckBox.setSelected(showInfoDialog);
        canvas.requestFocus();
    }

    @FXML
    public void showConsole() {
        try {
            if (consoleWindow == null) {
                consoleWindow = new ConsoleWindow(this);
            }
            consoleWindow.show();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public int fireAllManagers() {
        if (habitat != null) {
            int count = habitat.fireAllManagers();
            updateStatistics();
            return count;
        }
        return 0;
    }

    public void hireManagers(int n) {
        if (habitat != null) {
            habitat.hireManagers(n);
            updateStatistics();
        }
    }

    @FXML
    public void saveSimulation() {
        if (habitat == null) return;

        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Сохранить симуляцию");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Simulation Files", "*.sim"));
        File file = fileChooser.showSaveDialog(canvas.getScene().getWindow());

        if (file != null) {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(file))) {
                List<Employee> employees = CollectionsStorage.getInstance().getEmployees();
                oos.writeObject(employees);
                oos.writeLong(habitat.getElapsedTime());
                showAlert("Успех", "Симуляция сохранена в " + file.getName());
            } catch (IOException e) {
                showAlert("Ошибка", "Не удалось сохранить симуляцию: " + e.getMessage());
            }
        }
    }

    @FXML
    @SuppressWarnings("unchecked")
    public void loadSimulation() {
        FileChooser fileChooser = new FileChooser();
        fileChooser.setTitle("Загрузить симуляцию");
        fileChooser.getExtensionFilters().add(new FileChooser.ExtensionFilter("Simulation Files", "*.sim"));
        File file = fileChooser.showOpenDialog(canvas.getScene().getWindow());

        if (file != null) {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(file))) {
                // Останавливаем текущую симуляцию
                if (habitat != null && habitat.isRunning()) {
                    stopSimulation();
                }

                List<Employee> employees = (List<Employee>) ois.readObject();
                long savedElapsedTime = ois.readLong();

                // Сброс и инициализация среды
                Habitat.resetInstance();
                startSimulation();
                
                // Очищаем текущие объекты (созданные при старте) и добавляем загруженные
                CollectionsStorage.getInstance().clear();
                habitat.resetCounts();

                for (Employee emp : employees) {
                    // Корректировка времени рождения: 
                    // прожитое_время = savedElapsedTime - старое_birthTime
                    // новое_birthTime = текущее_время_симуляции - прожитое_время
                    // Поскольку мы только что начали симуляцию, текущее время = 0
                    emp.setBirthTime(emp.getBirthTime() - savedElapsedTime);
                    habitat.addLoadedEmployee(emp);
                }

                startbtn.setDisable(true);
                stopbtn.setDisable(false);
                updateStatistics();
                showAlert("Успех", "Симуляция загружена из " + file.getName());
            } catch (IOException | ClassNotFoundException e) {
                showAlert("Ошибка", "Не удалось загрузить симуляцию: " + e.getMessage());
                e.printStackTrace();
            }
        }
    }

    private void saveConfig() {
        File configFile = new File("config.txt");
        try (PrintWriter writer = new PrintWriter(new FileWriter(configFile))) {
            writer.println("n1=" + n1FieldRight.getText());
            writer.println("n2=" + n2FieldRight.getText());
            writer.println("p1=" + probabilityComboBoxRight.getValue());
            writer.println("devLifetime=" + developerLifetimeField.getText());
            writer.println("mgrLifetime=" + managerLifetimeField.getText());
            writer.println("showTime=" + (timeon.isSelected() ? "true" : "false"));
            writer.println("showInfo=" + showInfoCheckBox.isSelected());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void loadConfig() {
        File configFile = new File("config.txt");
        if (!configFile.exists()) return;

        try (BufferedReader reader = new BufferedReader(new FileReader(configFile))) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split("=");
                if (parts.length != 2) continue;
                String key = parts[0];
                String value = parts[1];

                switch (key) {
                    case "n1": n1FieldRight.setText(value); break;
                    case "n2": n2FieldRight.setText(value); break;
                    case "p1": probabilityComboBoxRight.setValue(value); break;
                    case "devLifetime": developerLifetimeField.setText(value); break;
                    case "mgrLifetime": managerLifetimeField.setText(value); break;
                    case "showTime": 
                        if (value.equals("true")) timeon.setSelected(true);
                        else timeoff.setSelected(true);
                        break;
                    case "showInfo": showInfoCheckBox.setSelected(Boolean.parseBoolean(value)); break;
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    @FXML
    public void exitApplication() {
        saveConfig();
        if (consoleWindow != null) {
            consoleWindow.close();
        }
        Platform.exit();
    }

    private void showAlert(String title, String content) {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle(title);
        alert.setHeaderText(null);
        alert.setContentText(content);
        alert.showAndWait();
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
    
    @FXML
    private void showCurrentObjectsButton() {
        showCurrentObjectsDialog();
        canvas.requestFocus();
    }

    /**
     * Приостанавливает AI разработчиков.
     */
    @FXML
    private void pauseDeveloperAI() {
        if (habitat != null && habitat.getDeveloperAI() != null) {
            habitat.getDeveloperAI().pause();
        }
        canvas.requestFocus();
    }

    /**
     * Возобновляет AI разработчиков.
     */
    @FXML
    private void resumeDeveloperAI() {
        if (habitat != null && habitat.getDeveloperAI() != null) {
            habitat.getDeveloperAI().resume();
        }
        canvas.requestFocus();
    }

    /**
     * Приостанавливает AI менеджеров.
     */
    @FXML
    private void pauseManagerAI() {
        if (habitat != null && habitat.getManagerAI() != null) {
            habitat.getManagerAI().pause();
        }
        canvas.requestFocus();
    }

    /**
     * Возобновляет AI менеджеров.
     */
    @FXML
    private void resumeManagerAI() {
        if (habitat != null && habitat.getManagerAI() != null) {
            habitat.getManagerAI().resume();
        }
        canvas.requestFocus();
    }

    /**
     * Показывает модальное окно со списком текущих живых объектов.
     * В диалоговое окно передаётся LinkedList с объектами и HashMap с временем рождения.
     */
    private void showCurrentObjectsDialog() {
        if (habitat == null || !habitat.isRunning()) {
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Текущие объекты");
            alert.setHeaderText("Симуляция не запущена");
            alert.setContentText("Запустите симуляцию для просмотра текущих объектов.");
            alert.initOwner(canvas.getScene().getWindow());
            alert.showAndWait();
            return;
        }
        
        // Получаем коллекции из Habitat
        LinkedList<Employee> employees = habitat.getEmployeesLinkedList();
        
        // Создаём модальное окно
        Stage dialogStage = new Stage();
        dialogStage.setTitle("Текущие объекты");
        dialogStage.initModality(Modality.WINDOW_MODAL);
        dialogStage.initOwner(canvas.getScene().getWindow());

        VBox vbox = new VBox(10);
        vbox.setPadding(new Insets(15));

        Label titleLabel = new Label("Список живых объектов");
        titleLabel.setFont(Font.font("Arial", FontWeight.BOLD, 16));

        // Создаём TextArea для отображения списка
        TextArea textArea = new TextArea();
        textArea.setPrefRowCount(15);
        textArea.setPrefColumnCount(50);
        textArea.setEditable(false);
        textArea.setWrapText(false);

        // Формируем текст списка объектов
        StringBuilder sb = new StringBuilder();
        sb.append("ID\t\tТип\t\t\tВремя рождения (мс)\tВремя жизни (сек)\n");
        sb.append("=".repeat(70)).append("\n");
        
        // Сортируем по времени рождения (ключу)
        employees.sort(Comparator.comparingLong(Employee::getBirthTime));
        
        for (Employee emp : employees) {
            String type = emp.getType();
            long birthTime = emp.getBirthTime();
            long lifetimeSec = emp.getLifetime() / 1000;
            sb.append(emp.getId()).append("\t\t")
              .append(type).append("\t\t\t")
              .append(birthTime).append("\t\t\t\t")
              .append(lifetimeSec).append("\n");
        }
        
        if (employees.isEmpty()) {
            sb.append("Нет активных объектов\n");
        }
        
        sb.append("=".repeat(70)).append("\n");
        sb.append("Всего объектов: ").append(employees.size());

        textArea.setText(sb.toString());

        Button closeBtn = new Button("Закрыть");
        closeBtn.setPrefWidth(100);
        closeBtn.setOnAction(e -> dialogStage.close());

        HBox buttonBox = new HBox(closeBtn);
        buttonBox.setAlignment(Pos.CENTER);

        vbox.getChildren().addAll(titleLabel, textArea, buttonBox);

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
            habitat.setDeveloperLifetime(settings.developerLifetime);
            habitat.setManagerLifetime(settings.managerLifetime);
        } else if (!habitat.isRunning()) {
            Habitat.resetInstance();
            habitat = Habitat.getInstance(canvas.getWidth(), canvas.getHeight(),
                    settings.n1, settings.n2, settings.p1, settings.kPercent);
            habitat.setDeveloperLifetime(settings.developerLifetime);
            habitat.setManagerLifetime(settings.managerLifetime);
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
            // habitat уже на паузе, просто завершаем
            startbtn.setDisable(false);
            stopbtn.setDisable(true);
            simulationEnded = true;
            dialogStage.close();
        });

        Button cancelButton = new Button("Отмена");
        cancelButton.setPrefWidth(80);
        cancelButton.setOnAction(e -> {
            // Возобновляем симуляцию
            habitat.resume();
            timer.start();
            waitForOkCancel = false;
            dialogStage.close();
        });

        buttonBox.getChildren().addAll(okButton, cancelButton);
        vbox.getChildren().addAll(titleLabel, textArea, buttonBox);

        Scene scene = new Scene(vbox);
        dialogStage.setScene(scene);
        dialogStage.setResizable(false);

        // Приостанавливаем симуляцию на время диалога
        if (habitat != null && habitat.isRunning()) {
            habitat.pause();
        }
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
        // Просто переключаем группу, слушатель сделает всё остальное
        Toggle current = toggleGroup.getSelectedToggle();
        Toggle target = (current == timeon) ? timeoff : timeon;
        toggleGroup.selectToggle(target);

        // updateAndRender() и requestFocus() уже в слушателе!
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
        long developerLifetime;
        long managerLifetime;

        SimulationSettings(int n1, int n2, double p1, int kPercent, long developerLifetime, long managerLifetime) {
            this.n1 = n1;
            this.n2 = n2;
            this.p1 = p1;
            this.kPercent = kPercent;
            this.developerLifetime = developerLifetime;
            this.managerLifetime = managerLifetime;
        }
    }
}
