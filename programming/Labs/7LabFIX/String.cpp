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
    if (s) {
        length = strlen(s);
        str = new char[length + 1];
        strcpy(str, s);
    } else {
        length = 0;
        str = nullptr;
    }
    count++;
    std::cout << "Вызван конструктор с параметром. Объектов: " << count << std::endl;
}

// Конструктор копирования
String::String(const String& other) {
    if (other.str) {
        length = other.length;
        str = new char[length + 1];
        strcpy(str, other.str);
    } else {
        length = 0;
        str = nullptr;
    }
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
    if (s) {
        length = strlen(s);
        str = new char[length + 1];
        strcpy(str, s);
    } else {
        length = 0;
        str = nullptr;
    }
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

// Оператор сложения двух объектов (String + String)
String String::operator+(const String& other) const {
    if (!str && !other.str) return String();
    if (!str) return String(other.str);
    if (!other.str) return String(str);
    
    char* newStr = new char[length + other.length + 1];
    strcpy(newStr, str);
    strcat(newStr, other.str);
    String result(newStr);
    delete[] newStr;
    return result;
}

// Оператор сложения с char* (String + char*)
String String::operator+(const char* other) const {
    if (!str && !other) return String();
    if (!str) return String(other);
    if (!other) return String(str);
    
    int otherLen = strlen(other);
    char* newStr = new char[length + otherLen + 1];
    strcpy(newStr, str);
    strcat(newStr, other);
    String result(newStr);
    delete[] newStr;
    return result;
}

// Оператор присваивания
String& String::operator=(const String& other) {
    if (this != &other) {
        delete[] str;
        if (other.str) {
            length = other.length;
            str = new char[length + 1];
            strcpy(str, other.str);
        } else {
            length = 0;
            str = nullptr;
        }
    }
    std::cout << "Вызван оператор присваивания." << std::endl;
    return *this;
}

// Операторы сравнения
bool String::operator==(const String& other) const {
    if (!str && !other.str) return true;
    if (!str || !other.str) return false;
    return strcmp(str, other.str) == 0;
}

bool String::operator!=(const String& other) const {
    return !(*this == other);
}

bool String::operator<(const String& other) const {
    if (!str && !other.str) return false;
    if (!str) return true;
    if (!other.str) return false;
    return strcmp(str, other.str) < 0;
}

bool String::operator>(const String& other) const {
    if (!str && !other.str) return false;
    if (!str) return false;
    if (!other.str) return true;
    return strcmp(str, other.str) > 0;
}

bool String::operator<=(const String& other) const {
    return !(*this > other);
}

bool String::operator>=(const String& other) const {
    return !(*this < other);
}

// Дружественная функция: char* + String
String operator+(const char* lhs, const String& rhs) {
    if (!lhs && !rhs.str) return String();
    if (!lhs) return String(rhs.str);
    if (!rhs.str) return String(lhs);
    
    int lhsLen = strlen(lhs);
    char* newStr = new char[lhsLen + rhs.length + 1];
    strcpy(newStr, lhs);
    strcat(newStr, rhs.str);
    String result(newStr);
    delete[] newStr;
    return result;
}

// Дружественная функция: вывод
std::ostream& operator<<(std::ostream& os, const String& obj) {
    if (obj.str) os << obj.str;
    else os << "";
    return os;
}

// Дружественная функция: ввод
std::istream& operator>>(std::istream& is, String& obj) {
    char buffer[1000];
    is.getline(buffer, 1000);
    obj.setString(buffer);
    return is;
}

// Получение количества объектов
int String::getCount() {
    return count;
}