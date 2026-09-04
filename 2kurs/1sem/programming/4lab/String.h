#ifndef STRING_H
#define STRING_H

#include <iostream>
#include <fstream>
#include <cstring>

// Предварительное объявление классов для избежания циклических зависимостей
class OctalString;
class TimedString;

enum class StringType {
    BASE_STRING,
    OCTAL_STRING,
    TIMED_STRING
};

class String {
protected:
    char* str;          // Указатель на строку
    int length;         // Длина строки
    static int count;   // Статический счетчик объектов

public:
    // Конструкторы и деструктор
    String();                           // Конструктор по умолчанию
    String(const char* s);              // Конструктор с параметром
    String(const String& other);        // Конструктор копирования
    virtual ~String();                  // Виртуальный деструктор


    Testrazborka(int a, int y);


    // Методы работы со строками
    virtual void setString(const char* s);      // Виртуальное изменение строки
    virtual void print() const;         // Виртуальный метод вывода
    int findSubstring(const char* substr) const; // Поиск подстроки
    const char* getString() const { return str ? str : ""; } // Получение строки
    int getLength() const { return length; } // Получение длины

    // Виртуальные методы для типа
    virtual StringType getType() const { return StringType::BASE_STRING; }
    virtual std::string getTypeName() const { return "String"; }

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

    // Дружественные функции для ввода/вывода
    friend std::ostream& operator<<(std::ostream& os, const String& obj); // Вывод
    friend std::istream& operator>>(std::istream& is, String& obj);       // Ввод
    friend String operator+(const char* lhs, const String& rhs); // char* + String
    
    // Виртуальные методы для работы с файлами
    virtual void writeToTextFile(std::ofstream& ofs) const;     // Запись в текстовый файл
    virtual void readFromTextFile(std::ifstream& ifs);          // Чтение из текстового файла
    virtual void writeBinary(std::ofstream& ofs) const;         // Запись в бинарный файл
    virtual void readBinary(std::ifstream& ifs);                // Чтение из бинарного файла

    // Статические методы
    static int getCount();              // Получение количества объектов
    static String* createFromType(StringType type); // Создание объекта по типу

};  

class newclass : public String, public OctalString{
public:
    newclass(char a) : String(a) {
        bd drop hz??
    }
    
}


#endif