package com.example.ru.nstu._1lab;

import java.util.ArrayList;
import java.util.List;      // импорт интерфейса List
public class Company {
    private List<Employee> employees = new ArrayList<>();
    private int developerCount;
    private int managerCount;

    public synchronized void addDeveloper() {
        employees.add(new Developer());
        developerCount++;
        System.out.println("Added new developer: Total " + developerCount); };

    public synchronized void addManager() {
        employees.add(new Manager());
        managerCount++;
        System.out.println("Added new manager: Total " + managerCount); };

    public synchronized boolean canAddManager(int kPercent) {
        if (developerCount == 0) return false;
        double currentPercent = (managerCount * 100.0) / developerCount;
        return currentPercent < kPercent; };

    public void printStatistic() {
        System.out.println("Statistic:\nDevelopers: " + developerCount + "\n" + "Managers: " + managerCount);
    };
}
