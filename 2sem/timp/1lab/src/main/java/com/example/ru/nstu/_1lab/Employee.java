package com.example.ru.nstu._1lab;
public abstract class Employee {
    private static int counter = 0;
    protected int id;
    protected long CreationTime;

    public Employee() {
        this.id = ++counter;
        this.CreationTime = System.currentTimeMillis();
    }

    protected int GetId() { return id; }
    protected long GetCreationTime(){ return CreationTime; }
}
