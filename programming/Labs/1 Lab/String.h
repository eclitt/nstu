#ifndef STRING_H
#define STRING_H

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
    String concat(const String& other) const;    // Объединение строк

    // Статический метод
    static int getCount();              // Получение количества объектов
};

#endif