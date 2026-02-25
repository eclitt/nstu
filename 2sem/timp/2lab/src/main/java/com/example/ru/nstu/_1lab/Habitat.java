package com.example.ru.nstu._1lab;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Класс среды обитания объектов (компании).
 * Хранит список сотрудников, управляет их генерацией и обновлением.
 */
public class Habitat {
    private List<Employee> employees = new ArrayList<>();
    
    // Параметры симуляции
    private int n1 = 2;           // интервал генерации разработчика (сек)
    private int n2 = 3;           // интервал генерации менеджера (сек)
    private double p1 = 0.7;      // вероятность появления разработчика
    private int kPercent = 50;    // максимальный процент менеджеров от разработчиков
    
    // Размеры рабочей области
    private double width;
    private double height;
    
    // Состояние симуляции
    private long startTime = 0;
    private long lastDeveloperTime = 0;
    private long lastManagerTime = 0;
    private boolean isRunning = false;
    
    private Random random = new Random();
    
    // Счётчики
    private int developerCount = 0;
    private int managerCount = 0;

    public Habitat(double width, double height) {
        this.width = width;
        this.height = height;
    }

    public Habitat(double width, double height, int n1, int n2, double p1, int kPercent) {
        this.width = width;
        this.height = height;
        this.n1 = n1;
        this.n2 = n2;
        this.p1 = p1;
        this.kPercent = kPercent;
    }

    /**
     * Метод обновления состояния среды.
     * Вызывается по таймеру, генерирует новые объекты.
     * 
     * @param elapsedTime время от начала симуляции (в мс)
     */
    public synchronized void update(long elapsedTime) {
        if (!isRunning) {
            return;
        }

        // Генерация разработчика
        if (elapsedTime - lastDeveloperTime >= n1 * 1000L) {
            if (random.nextDouble() < p1) {
                addDeveloper();
            }
            lastDeveloperTime = elapsedTime;
        }

        // Генерация менеджера
        if (elapsedTime - lastManagerTime >= n2 * 1000L) {
            if (canAddManager()) {
                addManager();
            }
            lastManagerTime = elapsedTime;
        }

        // Обновление всех объектов
        for (Employee employee : employees) {
            employee.update(elapsedTime);
        }
    }

    private void addDeveloper() {
        Developer developer = new Developer();
        placeEmployee(developer);
        employees.add(developer);
        developerCount++;
    }

    private void addManager() {
        Manager manager = new Manager();
        placeEmployee(manager);
        employees.add(manager);
        managerCount++;
    }

    /**
     * Размещает сотрудника в случайном месте рабочей области.
     */
    private void placeEmployee(Employee employee) {
        double x = random.nextDouble() * (width - employee.getWidth());
        double y = random.nextDouble() * (height - employee.getHeight());
        employee.setX(x);
        employee.setY(y);
    }

    private boolean canAddManager() {
        if (developerCount == 0) {
            return false;
        }
        double currentPercent = (managerCount * 100.0) / developerCount;
        return currentPercent < kPercent;
    }

    /**
     * Запускает симуляцию.
     */
    public synchronized void start() {
        isRunning = true;
        startTime = System.currentTimeMillis();
        lastDeveloperTime = 0;
        lastManagerTime = 0;
    }

    /**
     * Останавливает симуляцию и очищает список объектов.
     */
    public synchronized void stop() {
        isRunning = false;
        employees.clear();
    }

    /**
     * Проверяет, запущена ли симуляция.
     */
    public boolean isRunning() {
        return isRunning;
    }

    /**
     * Возвращает время симуляции в миллисекундах.
     */
    public long getElapsedTime() {
        if (!isRunning && startTime == 0) {
            return 0;
        }
        return System.currentTimeMillis() - startTime;
    }

    /**
     * Возвращает список сотрудников.
     */
    public List<Employee> getEmployees() {
        return new ArrayList<>(employees);
    }

    /**
     * Возвращает количество разработчиков.
     */
    public int getDeveloperCount() {
        return developerCount;
    }

    /**
     * Возвращает количество менеджеров.
     */
    public int getManagerCount() {
        return managerCount;
    }

    /**
     * Возвращает общее количество сгенерированных объектов.
     */
    public int getTotalCount() {
        return developerCount + managerCount;
    }

    /**
     * Возвращает ширину рабочей области.
     */
    public double getWidth() {
        return width;
    }

    /**
     * Возвращает высоту рабочей области.
     */
    public double getHeight() {
        return height;
    }
}
