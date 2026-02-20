package com.example.ru.nstu._1lab;

import java.util.Random;

public class EmployeeGenerator implements Runnable{
    private Company company;
    private int n1;
    private int n2;
    private double p1;
    private int kPercent;
    private volatile boolean running = true;

    public EmployeeGenerator(Company company, int n1,int n2,double p1, int kPercent) {
        this.company = company;
        this.n1 = n1;
        this.n2 = n2;
        this.p1 = p1;
        this.kPercent = kPercent; }

    @Override
    public void run() {
        long lastDeveloperTime = System.currentTimeMillis();
        long lastManagerTime = System.currentTimeMillis();
        Random random = new Random();

        while (running) {
            long currentTime = System.currentTimeMillis();

            // Генерация разработчика
            if (currentTime - lastDeveloperTime >= n1 * 1000L) {
                if (random.nextDouble() < p1) {
                    company.addDeveloper();
                }
                lastDeveloperTime = currentTime;
            }

            // Генерация менеджера
            if (currentTime - lastManagerTime >= n2 * 1000L) {
                if (company.canAddManager(kPercent)) {
                    company.addManager();
                } else { System.out.println("Cannot add manager: limit reached (" + kPercent + "% of developers)"); }
                lastManagerTime = currentTime;
            }

            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                break;
            }
        }
    }

    public void stop() { running = false; }
}
