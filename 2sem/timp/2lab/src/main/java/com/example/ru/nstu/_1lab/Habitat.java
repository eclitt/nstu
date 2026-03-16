package com.example.ru.nstu._1lab;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Habitat {
    private static Habitat instance;

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
    private long elapsedTime = 0;
    private long lastDeveloperTime = 0;
    private long lastManagerTime = 0;
    private boolean isRunning = false;

    private Random random = new Random();

    // Счётчики
    private int developerCount = 0;
    private int managerCount = 0;

    /**
     * Приватный конструктор для предотвращения создания экземпляров извне.
     */
    private Habitat() {
    }

    /**
     * Возвращает единственный экземпляр Habitat (Singleton).
     * @param width ширина рабочей области
     * @param height высота рабочей области
     * @param n1 интервал генерации разработчика (сек)
     * @param n2 интервал генерации менеджера (сек)
     * @param p1 вероятность появления разработчика
     * @param kPercent максимальный процент менеджеров от разработчиков
     * @return экземпляр Habitat
     */
    public static synchronized Habitat getInstance(double width, double height, int n1, int n2, double p1, int kPercent) {
        if (instance == null) {
            instance = new Habitat();
        }
        instance.width = width;
        instance.height = height;
        instance.n1 = n1;
        instance.n2 = n2;
        instance.p1 = p1;
        instance.kPercent = kPercent;
        return instance;
    }

    /**
     * Возвращает единственный экземпляр Habitat (Singleton) без параметров.
     * Используется для получения доступа к существующему экземпляру.
     * @return экземпляр Habitat
     */
    public static synchronized Habitat getInstance() {
        if (instance == null) {
            instance = new Habitat();
        }
        return instance;
    }

    /**
     * Сбрасывает экземпляр Singleton. Вызывается при полном перезапуске симуляции.
     */
    public static synchronized void resetInstance() {
        instance = null;
    }

    /**
     * Применяет новые параметры симуляции.
     * @param n1 интервал генерации разработчика (сек)
     * @param n2 интервал генерации менеджера (сек)
     * @param p1 вероятность появления разработчика
     * @param kPercent максимальный процент менеджеров от разработчиков
     */
    public synchronized void applySettings(int n1, int n2, double p1, int kPercent) {
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
        elapsedTime = 0;
        lastDeveloperTime = 0;
        lastManagerTime = 0;
        developerCount = 0;
        managerCount = 0;
        employees.clear();
    }

    /**
     * Останавливает симуляцию и очищает список объектов.
     */
    public synchronized void stop() {
        if (isRunning) {
            elapsedTime = System.currentTimeMillis() - startTime;
            isRunning = false;
            employees.clear();
        }
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
        if (!isRunning) {
            return elapsedTime;
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
