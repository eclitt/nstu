#ifndef STRING_H
#define STRING_H

#include <iostream>

class String {
private:
    char* str;          // Указатель на строку
    int length;         // Длина строки
    static int count;   // Статический счетчик объектов

public:
    // Конструкторы и деструктор
    String();                           // Конструктор по умолчанию
    String(const char* s);              // Конструктор с параметром
    String(const String& other);        // Конструктор копирования
    ~String();                          // Деструктор

    // Методы работы со строками
    void setString(const char* s);      // Изменение строки
    void print() const;                 // Вывод строки
    int findSubstring(const char* substr) const; // Поиск подстроки
    const char* getString() const { return str ? str : ""; } // Получение строки
    int getLength() const { return length; } // Получение длины

    // Перегруженные операторы (методы класса)
    String operator+(const String& other) const;    // Сложение двух объектов
    String operator+(const char* other) const;      // Сложение с char*
    String& operator=(const String& other);         // Присваивание
    bool operator==(const String& other) const;     // Сравнение ==
    bool operator!=(const String& other) const;     // Сравнение !=
    bool operator<(const String& other) const;      // Сравнение <
    bool operator>(const String& other) const;      // Сравнение >
    bool operator<=(const String& other) const;     // Сравнение <=
    bool operator>=(const String& other) const;     // Сравнение >=

    // Дружественные функции
    friend String operator+(const char* lhs, const String& rhs); // char* + String
    friend std::ostream& operator<<(std::ostream& os, const String& obj); // Вывод
    friend std::istream& operator>>(std::istream& is, String& obj);       // Ввод

    // Статический метод
    static int getCount();              // Получение количества объектов
};

#endif