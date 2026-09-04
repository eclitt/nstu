package com.example.ru.nstu._1lab;

import java.util.*;

public class Habitat {
    private static Habitat instance;

    // Ссылка на синглтон коллекций
    private final CollectionsStorage storage;

    // AI потоки для управления движением объектов
    private DeveloperAI developerAI;
    private ManagerAI managerAI;

    // Параметры симуляции
    private int n1 = 2;           // интервал генерации разработчика (сек)
    private int n2 = 3;           // интервал генерации менеджера (сек)
    private double p1 = 0.7;      // вероятность появления разработчика
    private int kPercent = 50;    // максимальный процент менеджеров от разработчиков

    // Время жизни для разных типов объектов  (мс)
    private long developerLifetime = 30000;  // 30 секунд по умолчанию
    private long managerLifetime = 25000;    // 25 секунд по умолчанию

    // Параметры движения
    private int developerDirectionChangeInterval = 3; // секунды между сменой направления
    private double managerOrbitRadius = 80.0;         // радиус орбиты менеджера

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
        Employee.clearIdSet();
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
     * Вызывается по таймеру (в основном потоке), генерирует новые объекты и удаляет истёкшие.
     * Движением объектов управляют AI потоки отдельно.
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

    @SuppressWarnings("unchecked")
    private synchronized void addDeveloper(long currentTime) {
        Developer developer = new Developer();
        developer.setLifetime(developerLifetime);
        developer.setBirthTime(currentTime);
        placeEmployee(developer);

        // Добавление через хранилище
        storage.addEmployee(developer, currentTime);
        developerCount++;

        // Регистрация в AI (если AI запущен)
        if (developerAI != null) {
            synchronized (developerAI.employees) {
                List<Developer> devList = (List<Developer>) developerAI.employees;
                devList.add(developer);
            }
        }
    }

    @SuppressWarnings("unchecked")
    private synchronized void addManager(long currentTime) {
        Manager manager = new Manager();
        manager.setLifetime(managerLifetime);
        manager.setBirthTime(currentTime);
        placeEmployee(manager);

        // Добавление через хранилище
        storage.addEmployee(manager, currentTime);
        managerCount++;

        // Регистрация в AI (если AI запущен)
        if (managerAI != null) {
            synchronized (managerAI.employees) {
                List<Manager> managerList = (List<Manager>) managerAI.employees;
                managerList.add(manager);
            }
        }
    }

    /**
     * Удаляет всех менеджеров из симуляции.
     * @return количество удаленных менеджеров
     */
    public synchronized int fireAllManagers() {
        List<Employee> allEmployees = storage.getEmployees();
        int removedCount = 0;
        for (Employee employee : allEmployees) {
            if (employee instanceof Manager) {
                storage.removeEmployee(employee);
                managerCount--;
                removedCount++;
                
                // Удаление из AI
                if (managerAI != null) {
                    synchronized (managerAI.employees) {
                        managerAI.employees.remove(employee);
                    }
                }
            }
        }
        return removedCount;
    }

    /**
     * Генерирует N новых менеджеров.
     * @param n количество новых менеджеров
     */
    public synchronized void hireManagers(int n) {
        for (int i = 0; i < n; i++) {
            addManager(elapsedTime);
        }
    }

    /**
     * Сбрасывает счетчики объектов.
     */
    public synchronized void resetCounts() {
        developerCount = 0;
        managerCount = 0;
    }

    /**
     * Добавляет загруженный объект в симуляцию.
     * @param employee объект
     */
    @SuppressWarnings("unchecked")
    public synchronized void addLoadedEmployee(Employee employee) {
        storage.addEmployee(employee, employee.getBirthTime());
        if (employee instanceof Developer) {
            developerCount++;
            if (developerAI != null) {
                synchronized (developerAI.employees) {
                    ((List<Developer>) developerAI.employees).add((Developer) employee);
                }
            }
        } else if (employee instanceof Manager) {
            managerCount++;
            if (managerAI != null) {
                synchronized (managerAI.employees) {
                    ((List<Manager>) managerAI.employees).add((Manager) employee);
                }
            }
        }
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
     * Запускает симуляцию и AI потоки.
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
        // Очистка статического набора в классе Employee
        Employee.clearIdSet();

        // Инициализация и запуск AI потоков
        initAndStartAI();
    }

    /**
     * Инициализирует и запускает AI потоки.
     */
    private void initAndStartAI() {
        // Создаём списки для AI (будут пополняться при спавне объектов)
        List<Developer> devList = Collections.synchronizedList(new ArrayList<>());
        List<Manager> mgrList = Collections.synchronizedList(new ArrayList<>());

        developerAI = new DeveloperAI(devList, width, height, developerDirectionChangeInterval);
        managerAI = new ManagerAI(mgrList, width, height, managerOrbitRadius);

        developerAI.start();
        managerAI.start();
    }

    /**
     * Останавливает симуляцию, AI потоки и очищает список объектов.
     */
    public synchronized void stop() {
        if (isRunning) {
            elapsedTime = System.currentTimeMillis() - startTime;
            isRunning = false;

            // Остановка AI потоков
            if (developerAI != null) {
                developerAI.stop();
                developerAI = null;
            }
            if (managerAI != null) {
                managerAI.stop();
                managerAI = null;
            }

            storage.clear();
        }
    }

    /**
     * Приостанавливает симуляцию и AI потоки без очистки данных.
     */
    public synchronized void pause() {
        if (isRunning) {
            pauseTime = System.currentTimeMillis() - startTime;
            isRunning = false;

            // Пауза AI потоков
            if (developerAI != null) {
                developerAI.pause();
            }
            if (managerAI != null) {
                managerAI.pause();
            }
        }
    }

    /**
     * Возобновляет симуляцию и AI потоки после паузы.
     */
    public synchronized void resume() {
        if (!isRunning) {
            startTime = System.currentTimeMillis() - pauseTime;
            isRunning = true;

            // Возобновление AI потоков
            if (developerAI != null) {
                developerAI.resume();
            }
            if (managerAI != null) {
                managerAI.resume();
            }
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

    /**
     * Возвращает AI для разработчиков.
     */
    public DeveloperAI getDeveloperAI() {
        return developerAI;
    }

    /**
     * Возвращает AI для менеджеров.
     */
    public ManagerAI getManagerAI() {
        return managerAI;
    }

    /**
     * Возвращает интервал смены направления разработчиков.
     */
    public int getDeveloperDirectionChangeInterval() {
        return developerDirectionChangeInterval;
    }

    /**
     * Устанавливает интервал смены направления разработчиков.
     */
    public void setDeveloperDirectionChangeInterval(int seconds) {
        this.developerDirectionChangeInterval = seconds;
    }

    /**
     * Возвращает радиус орбиты менеджеров.
     */
    public double getManagerOrbitRadius() {
        return managerOrbitRadius;
    }

    /**
     * Устанавливает радиус орбиты менеджеров.
     */
    public void setManagerOrbitRadius(double radius) {
        this.managerOrbitRadius = radius;
    }
}