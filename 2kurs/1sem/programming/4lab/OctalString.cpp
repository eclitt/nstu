#include "OctalString.h"
#include <cstring>
#include <iostream>
#include <string>

// Проверка, является ли строка восьмеричным числом
bool OctalString::isValidOctal(const char* s) const {
    if (!s || strlen(s) == 0) return false;
    
    // Восьмеричное число может содержать только цифры 0-7
    for (int i = 0; s[i] != '\0'; i++) {
        if (s[i] < '0' || s[i] > '7') {
            return false;
        }
    }
    return true;
}

// Перевод из восьмеричной в десятичную систему
long long OctalString::octalToDecimal(const char* octal) const {
    if (!octal || !isValidOctal(octal)) return 0;
    
    long long decimal = 0;
    int len = strlen(octal);
    
    for (int i = 0; i < len; i++) {
        decimal = decimal * 8 + (octal[i] - '0');
    }
    
    return decimal;
}

// Перевод из десятичной в восьмеричную систему
std::string OctalString::decimalToOctal(long long decimal) const {
    if (decimal == 0) return "0";
    
    std::string octal;
    long long temp = decimal < 0 ? -decimal : decimal;
    
    while (temp > 0) {
        octal = char('0' + (temp % 8)) + octal;
        temp /= 8;
    }
    
    if (decimal < 0) {
        octal = "-" + octal;
    }
    
    return octal;
}

// Конструктор по умолчанию
OctalString::OctalString() : String(), number(0) {
}

// Конструктор с параметром (строка)
OctalString::OctalString(const char* s) : String() {
    setString(s);
}

// Конструктор с параметром (число)
OctalString::OctalString(long long num) : String() {
    setNumber(num);
}

// Конструктор копирования
OctalString::OctalString(const OctalString& other) : String(other) {
    number = other.number;
}

// Конструктор преобразования
OctalString::OctalString(const String& other) : String(other) {
    if (isValidOctal(other.getString())) {
        number = octalToDecimal(other.getString());
    } else {
        number = 0;
    }
}

// Деструктор
OctalString::~OctalString() {
}

// Вывод строки с числовым значением
void OctalString::print() const {
    if (str) {
        std::cout << "OctalString: \"" << str << "\" (длина: " << length 
                  << ", десятичное значение: " << number << ")" << std::endl;
    } else {
        std::cout << "OctalString пуста." << std::endl;
    }
}

// Изменение строки с проверкой (переопределенная версия)
void OctalString::setString(const char* s) {
    if (!s) {
        String::setString(s);
        number = 0;
        return;
    }
    
    if (isValidOctal(s)) {
        String::setString(s);
        number = octalToDecimal(s);
    } else {
        throw std::invalid_argument("Строка '" + std::string(s) + "' не является восьмеричным числом");
    }
}

// Установка числового значения
void OctalString::setNumber(long long num) {
    number = num;
    std::string octalStr = decimalToOctal(num);
    String::setString(octalStr.c_str());
}

// Запись в текстовый файл
void OctalString::writeToTextFile(std::ofstream& ofs) const {
    // Сначала записываем тип объекта
    ofs << static_cast<int>(getType()) << " ";
    
    // Затем длину строки и саму строку
    ofs << length << " ";
    if (length > 0) {
        ofs.write(str, length);
    }
    
    // И числовое значение
    ofs << " " << number;
}

// Чтение из текстового файла
void OctalString::readFromTextFile(std::ifstream& ifs) {
    // Пропускаем тип (он уже считан фабричным методом)
    // Сначала читаем базовую часть (длину и строку)
    int len;
    ifs >> len;
    ifs.get(); // Пропускаем пробел
    
    if (len > 0) {
        char* buffer = new char[len + 1];
        ifs.read(buffer, len);
        buffer[len] = '\0';
        
        // Используем setString для проверки корректности
        try {
            this->setString(buffer);
        } catch (const std::exception& e) {
            delete[] buffer;
            throw; // Перебрасываем исключение дальше
        }
        delete[] buffer;
    } else {
        this->setString("");
        number = 0;
    }
    
    // Считываем числовое значение (если есть пробел перед ним)
    ifs >> number;
}

// Запись в бинарный файл
void OctalString::writeBinary(std::ofstream& ofs) const {
    // Сначала вызываем метод базового класса
    String::writeBinary(ofs);
    
    // Затем записываем числовое значение
    ofs.write(reinterpret_cast<const char*>(&number), sizeof(number));
}

// Чтение из бинарного файла
void OctalString::readBinary(std::ifstream& ifs) {
    // Сначала считываем базовую часть
    String::readBinary(ifs);
    
    // Затем считываем числовое значение
    ifs.read(reinterpret_cast<char*>(&number), sizeof(number));
}

// Оператор сложения
OctalString OctalString::operator+(const OctalString& other) const {
    long long sum = number + other.number;
    return OctalString(sum);
}

// Оператор присваивания (OctalString)
OctalString& OctalString::operator=(const OctalString& other) {
    if (this != &other) {
        String::operator=(other);
        number = other.number;
    }
    return *this;
}

// Оператор присваивания (String)
OctalString& OctalString::operator=(const String& other) {
    String::operator=(other);
    if (isValidOctal(other.getString())) {
        number = octalToDecimal(other.getString());
    } else {
        number = 0;
    }
    return *this;
}

// Сложение с числом
OctalString OctalString::operator+(long long num) const {
    long long sum = number + num;
    return OctalString(sum);
}

// Вычитание числа
OctalString OctalString::operator-(long long num) const {
    long long diff = number - num;
    return OctalString(diff);
}

// Сравнение
bool OctalString::operator==(const OctalString& other) const {
    return number == other.number;
}