package com.example.ru.nstu._1lab;

import java.util.*;

public class Habitat {
    private static Habitat instance;

    // Коллекция для хранения объектов (LinkedList по варианту 6)
    private LinkedList<Employee> employees = new LinkedList<>();

    // Коллекция для хранения и поиска уникальных идентификаторов (TreeSet по варианту 6)
    private TreeSet<Integer> existingIds = new TreeSet<>();

    // Коллекция для хранения времени рождения объектов (HashMap по варианту 6)
    // Ключ: ID объекта, Значение: время рождения (мс от начала симуляции)
    private HashMap<Integer, Long> birthTimeMap = new HashMap<>();

    // Параметры симуляции
    private int n1 = 2;           // интервал генерации разработчика (сек)
    private int n2 = 3;           // интервал генерации менеджера (сек)
    private double p1 = 0.7;      // вероятность появления разработчика
    private int kPercent = 50;    // максимальный процент менеджеров от разработчиков
    
    // Время жизни для разных типов объектов (мс)
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
     * Устанавливает время жизни для разработчиков.
     * @param lifetime время жизни в миллисекундах
     */
    public void setDeveloperLifetime(long lifetime) {
        this.developerLifetime = lifetime;
    }
    
    /**
     * Устанавливает время жизни для менеджеров.
     * @param lifetime время жизни в миллисекундах
     */
    public void setManagerLifetime(long lifetime) {
        this.managerLifetime = lifetime;
    }
    
    /**
     * Возвращает время жизни разработчиков.
     * @return время жизни в миллисекундах
     */
    public long getDeveloperLifetime() {
        return developerLifetime;
    }
    
    /**
     * Возвращает время жизни менеджеров.
     * @return время жизни в миллисекундах
     */
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

        // Обновление всех объектов
        for (Employee employee : employees) {
            employee.update(elapsedTime);
        }
        
        // Удаление объектов с истёкшим временем жизни
        removeExpiredEmployees(elapsedTime);
    }
    
    /**
     * Удаляет объекты с истёкшим временем жизни из всех коллекций.
     * @param currentTime текущее время симуляции
     */
    private void removeExpiredEmployees(long currentTime) {
        Iterator<Employee> iterator = employees.iterator();
        while (iterator.hasNext()) {
            Employee employee = iterator.next();
            if (employee.isExpired(currentTime)) {
                // Удаляем из LinkedList
                iterator.remove();
                // Удаляем из TreeSet идентификаторов
                existingIds.remove(employee.getId());
                // Удаляем из HashMap времени рождения
                birthTimeMap.remove(employee.getId());
                
                if (employee instanceof Developer) {
                    developerCount--;
                } else if (employee instanceof Manager) {
                    managerCount--;
                }
            }
        }
    }

    private void addDeveloper(long currentTime) {
        Developer developer = new Developer();
        developer.setLifetime(developerLifetime);
        developer.setBirthTime(currentTime);
        placeEmployee(developer);
        employees.add(developer);
        existingIds.add(developer.getId());
        birthTimeMap.put(developer.getId(), currentTime);
        developerCount++;
    }

    private void addManager(long currentTime) {
        Manager manager = new Manager();
        manager.setLifetime(managerLifetime);
        manager.setBirthTime(currentTime);
        placeEmployee(manager);
        employees.add(manager);
        existingIds.add(manager.getId());
        birthTimeMap.put(manager.getId(), currentTime);
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
        existingIds.clear();
        birthTimeMap.clear();
        Employee.idSet.clear();
    }

    /**
     * Останавливает симуляцию и очищает список объектов.
     */
    public synchronized void stop() {
        if (isRunning) {
            elapsedTime = System.currentTimeMillis() - startTime;
            isRunning = false;
            employees.clear();
            existingIds.clear();
            birthTimeMap.clear();
        }
    }

    /**
     * Приостанавливает симуляцию без очистки данных.
     * Может быть возобновлена через resume().
     */
    public synchronized void pause() {
        if (isRunning) {
            pauseTime = System.currentTimeMillis() - startTime;
            isRunning = false;
            // employees.clear() НЕ вызываем — сохраняем сотрудников
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
        return new LinkedList<>(employees);
    }
    
    /**
     * Возвращает LinkedList с сотрудниками (для передачи в диалоговое окно).
     * @return LinkedList сотрудников
     */
    public LinkedList<Employee> getEmployeesLinkedList() {
        return new LinkedList<>(employees);
    }
    
    /**
     * Возвращает HashMap с временем рождения объектов.
     * @return HashMap<ID, время рождения>
     */
    public HashMap<Integer, Long> getBirthTimeMap() {
        return new HashMap<>(birthTimeMap);
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
