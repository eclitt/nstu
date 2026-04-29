package com.example.ru.nstu._1lab;

import java.util.*;

/**
 * Singleton класс для хранения коллекций данных симуляции.
 * Инкапсулирует LinkedList, TreeSet и HashMap.
 */
public class CollectionsStorage {
    private static CollectionsStorage instance;

    // Коллекция для хранения объектов (LinkedList по варианту 6)
    private final LinkedList<Employee> employees = new LinkedList<>();

    // Коллекция для хранения и поиска уникальных идентификаторов (TreeSet по варианту 6)
    private final TreeSet<Integer> existingIds = new TreeSet<>();

    // Коллекция для хранения времени рождения объектов (HashMap по варианту 6)
    // Ключ: ID объекта, Значение: время рождения (мс от начала симуляции)
    private final HashMap<Integer, Long> birthTimeMap = new HashMap<>();

    /**
     * Приватный конструктор для предотвращения создания экземпляров извне.
     */
    private CollectionsStorage() {
    }

    /**
     * Возвращает единственный экземпляр CollectionsStorage (Singleton).
     * @return экземпляр CollectionsStorage
     */
    public static synchronized CollectionsStorage getInstance() {
        if (instance == null) {
            instance = new CollectionsStorage();
        }
        return instance;
    }

    /**
     * Добавляет сотрудника и сопутствующие данные в коллекции.
     * @param employee объект сотрудника
     * @param birthTime время рождения
     */
    public synchronized void addEmployee(Employee employee, long birthTime) {
        employees.add(employee);
        existingIds.add(employee.getId());
        birthTimeMap.put(employee.getId(), birthTime);
    }

    /**
     * Удаляет объекты с истёкшим временем жизни из всех коллекций.
     * @param currentTime текущее время симуляции
     * @return список удаленных сотрудников (для обновления счетчиков в Habitat)
     */
    public synchronized List<Employee> removeExpired(long currentTime) {
        List<Employee> removed = new ArrayList<>();
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

                removed.add(employee);
            }
        }
        return removed;
    }

    /**
     * Удаляет сотрудника из всех коллекций.
     * @param employee сотрудник для удаления
     */
    public synchronized void removeEmployee(Employee employee) {
        employees.remove(employee);
        existingIds.remove(employee.getId());
        birthTimeMap.remove(employee.getId());
    }

    /**
     * Очищает все коллекции.
     */
    public synchronized void clear() {
        employees.clear();
        existingIds.clear();
        birthTimeMap.clear();
    }

    /**
     * Возвращает список сотрудников (копию для безопасности).
     */
    public synchronized List<Employee> getEmployees() {
        return new LinkedList<>(employees);
    }

    /**
     * Возвращает LinkedList с сотрудниками.
     */
    public synchronized LinkedList<Employee> getEmployeesLinkedList() {
        return new LinkedList<>(employees);
    }

    /**
     * Возвращает HashMap с временем рождения объектов.
     */
    public synchronized HashMap<Integer, Long> getBirthTimeMap() {
        return new HashMap<>(birthTimeMap);
    }

    /**
     * Проверяет наличие ID.
     */
    public synchronized boolean containsId(int id) {
        return existingIds.contains(id);
    }

    /**
     * Возвращает количество элементов в коллекции сотрудников.
     */
    public synchronized int size() {
        return employees.size();
    }
}