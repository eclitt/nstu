#include "String.h"
#include <cstring>
#include <iostream>

// Инициализация статической переменной
int String::count = 0;

// Конструктор по умолчанию
String::String() : str(nullptr), length(0) {
    count++;
    std::cout << "Вызван конструктор по умолчанию. Объектов: " << count << std::endl;
}

// Конструктор с параметром
String::String(const char* s) {
    length = strlen(s);
    str = new char[length + 1];
    strcpy(str, s);
    count++;
    std::cout << "Вызван конструктор с параметром. Объектов: " << count << std::endl;
}

// Конструктор копирования
String::String(const String& other) {
    length = other.length;
    str = new char[length + 1];
    strcpy(str, other.str);
    count++;
    std::cout << "Вызван конструктор копирования. Объектов: " << count << std::endl;
}

// Деструктор
String::~String() {
    delete[] str;
    count--;
    std::cout << "Вызван деструктор. Осталось объектов: " << count << std::endl;
}

// Изменение строки
void String::setString(const char* s) {
    delete[] str;
    length = strlen(s);
    str = new char[length + 1];
    strcpy(str, s);
}

// Вывод строки
void String::print() const {
    if (str) std::cout << "Строка: " << str << " (длина: " << length << ")" << std::endl;
    else std::cout << "Строка пуста." << std::endl;
}

// Поиск подстроки
int String::findSubstring(const char* substr) const {
    if (!str || !substr) return -1;
    char* pos = strstr(str, substr);
    return pos ? pos - str : -1;
}

// Объединение строк
String String::concat(const String& other) const {
    char* newStr = new char[length + other.length + 1];
    strcpy(newStr, str);
    strcat(newStr, other.str);
    String result(newStr);
    delete[] newStr;
    return result;
}

// Получение количества объектов
int String::getCount() {
    return count;
}