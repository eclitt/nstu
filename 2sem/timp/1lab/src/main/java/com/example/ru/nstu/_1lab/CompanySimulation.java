package com.example.ru.nstu._1lab;


public class CompanySimulation {
    public static void  main(String[] args) throws InterruptedException {
        Company company = new Company();

        EmployeeGenerator generator = new EmployeeGenerator(company,2,3,0.7, 50);

        Thread generatorThread = new Thread(generator);
        generatorThread.start();

        for (int i = 0; i < 100; i++) {
            Thread.sleep(100);
            company.printStatistic();
        }
        generator.stop();
        generatorThread.join();

    }
}
