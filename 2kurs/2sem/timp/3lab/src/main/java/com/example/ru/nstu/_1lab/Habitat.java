package com.example.ru.nstu._1lab;

import java.util.*;

public class Habitat {
    private static Habitat instance;

    // Ссылка на синглтон коллекций
    private final CollectionsStorage storage;

    // Параметры симуляции
    private int n1 = 2;           // интервал генерации разработчика (сек)
    private int n2 = 3;           // интервал генерации менеджера (сек)
    private double p1 = 0.7;      // вероятность появления разработчика
    private int kPercent = 50;    // максимальный процент менеджеров от разработчиков

    // Время жизни для разных типов объектов  (мс)
    private long developerLifetime = 30000;  // 30 секунд по умолчанию
    private long managerLifetime = 25000;    // 25 секунд по умолчанию

    // Размеры рабочей области
    private double width;
    private double height;

    // Состояние симуляции
    private long startTime = 0;
    private long elapsedTime = 0;
    private long pauseTime = 0;
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
        this.storage = CollectionsStorage.getInstance();
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
     */
    public synchronized void applySettings(int n1, int n2, double p1, int kPercent) {
        this.n1 = n1;
        this.n2 = n2;
        this.p1 = p1;
        this.kPercent = kPercent;
    }

    public void setDeveloperLifetime(long lifetime) {
        this.developerLifetime = lifetime;
    }

    public void setManagerLifetime(long lifetime) {
        this.managerLifetime = lifetime;
    }

    public long getDeveloperLifetime() {
        return developerLifetime;
    }

    public long getManagerLifetime() {
        return managerLifetime;
    }

    /**
     * Метод обновления состояния среды.
     * Вызывается по таймеру, генерирует новые объекты и удаляет истёкшие.
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
                addDeveloper(elapsedTime);
            }
            lastDeveloperTime = elapsedTime;
        }

        // Генерация менеджера
        if (elapsedTime - lastManagerTime >= n2 * 1000L) {
            if (canAddManager()) {
                addManager(elapsedTime);
            }
            lastManagerTime = elapsedTime;
        }

        // Обновление всех объектов (получаем копию списка из хранилища)
        for (Employee employee : storage.getEmployees()) {
            employee.update(elapsedTime);
        }

        // Удаление объектов с истёкшим временем жизни
        removeExpiredEmployees(elapsedTime);
    }

    /**
     * Удаляет объекты с истёкшим временем жизни через CollectionsStorage.
     * @param currentTime текущее время симуляции
     */
    private void removeExpiredEmployees(long currentTime) {
        List<Employee> removed = storage.removeExpired(currentTime);

        for (Employee employee : removed) {
            if (employee instanceof Developer) {
                developerCount--;
            } else if (employee instanceof Manager) {
                managerCount--;
            }
        }
    }

    private void addDeveloper(long currentTime) {
        Developer developer = new Developer();
        developer.setLifetime(developerLifetime);
        developer.setBirthTime(currentTime);
        placeEmployee(developer);

        // Добавление через хранилище
        storage.addEmployee(developer, currentTime);
        developerCount++;
    }

    private void addManager(long currentTime) {
        Manager manager = new Manager();
        manager.setLifetime(managerLifetime);
        manager.setBirthTime(currentTime);
        placeEmployee(manager);

        // Добавление через хранилище
        storage.addEmployee(manager, currentTime);
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
        return currentPercent <= kPercent;
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

        // Очистка через хранилище
        storage.clear();
        // Очистка статического набора в классе Employee (как было в оригинале)
        if (Employee.idSet != null) {
            Employee.idSet.clear();
        }
    }

    /**
     * Останавливает симуляцию и очищает список объектов.
     */
    public synchronized void stop() {
        if (isRunning) {
            elapsedTime = System.currentTimeMillis() - startTime;
            isRunning = false;
            storage.clear();
        }
    }

    /**
     * Приостанавливает симуляцию без очистки данных.
     */
    public synchronized void pause() {
        if (isRunning) {
            pauseTime = System.currentTimeMillis() - startTime;
            isRunning = false;
        }
    }

    /**
     * Возобновляет симуляцию после паузы.
     */
    public synchronized void resume() {
        if (!isRunning) {
            startTime = System.currentTimeMillis() - pauseTime;
            isRunning = true;
        }
    }

    public boolean isRunning() {
        return isRunning;
    }

    public long getElapsedTime() {
        if (!isRunning) {
            return elapsedTime;
        }
        return System.currentTimeMillis() - startTime;
    }

    /**
     * Возвращает список сотрудников (через хранилище).
     */
    public List<Employee> getEmployees() {
        return storage.getEmployees();
    }

    /**
     * Возвращает LinkedList с сотрудниками.
     */
    public LinkedList<Employee> getEmployeesLinkedList() {
        return storage.getEmployeesLinkedList();
    }

    /**
     * Возвращает HashMap с временем рождения объектов.
     */
    public HashMap<Integer, Long> getBirthTimeMap() {
        return storage.getBirthTimeMap();
    }

    public int getDeveloperCount() {
        return developerCount;
    }

    public int getManagerCount() {
        return managerCount;
    }

    public int getTotalCount() {
        return developerCount + managerCount;
    }

    public double getWidth() {
        return width;
    }

    public double getHeight() {
        return height;
    }
}