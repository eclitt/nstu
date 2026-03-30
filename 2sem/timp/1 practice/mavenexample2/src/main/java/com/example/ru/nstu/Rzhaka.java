package com.example.ru.nstu;

public class Rzhaka {
    public static void main(String[] args) {
        System.out.println("Количество аргументов: " + args.length);

        for (int i = 0; i < args.length; i++) {
            System.out.println("Аргумент " + i + ": " + args[i]);
        }
    }
}
