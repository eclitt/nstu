#include "TimedString.h"
#include <cstring>
#include <iostream>

// Конструктор по умолчанию
TimedString::TimedString() : String() {
    creationTime = std::time(nullptr);
}

// Конструктор с параметром
TimedString::TimedString(const char* s) : String(s) {
    creationTime = std::time(nullptr);
}

// Конструктор копирования
TimedString::TimedString(const TimedString& other) : String(other) {
    creationTime = other.creationTime;
}

// Конструктор преобразования
TimedString::TimedString(const String& other) : String(other) {
    creationTime = std::time(nullptr);
}

// Деструктор
TimedString::~TimedString() {
}

// Вывод строки с временем создания
void TimedString::print() const {
    if (str) {
        std::cout << "TimedString: \"" << str << "\" (длина: " << length 
                  << ", создан: " << getCreationTimeString() << ")" << std::endl;
    } else {
        std::cout << "TimedString пуста. Создан: " << getCreationTimeString() << std::endl;
    }
}

// Получение строки с временем создания
std::string TimedString::getCreationTimeString() const {
    std::tm* timeinfo = std::localtime(&creationTime);
    char buffer[80];
    std::strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", timeinfo);
    return std::string(buffer);
}

// Запись в текстовый файл
void TimedString::writeToTextFile(std::ofstream& ofs) const {
    // Сначала записываем тип объекта
    ofs << static_cast<int>(getType()) << " ";
    
    // Затем длину строки и саму строку
    ofs << length << " ";
    if (length > 0) {
        ofs.write(str, length);
    }
    
    // И время создания
    ofs << " " << creationTime;
}

// Чтение из текстового файла
void TimedString::readFromTextFile(std::ifstream& ifs) {
    // Считываем длину строки
    ifs >> length;
    ifs.get(); // Пропускаем пробел
    
    if (length > 0) {
        char* buffer = new char[length + 1];
        ifs.read(buffer, length);
        buffer[length] = '\0';
        this->setString(buffer);
        delete[] buffer;
    } else {
        this->setString("");
    }
    
    // Считываем время создания
    ifs >> creationTime;
}

// Запись в бинарный файл
void TimedString::writeBinary(std::ofstream& ofs) const {
    // Сначала вызываем метод базового класса
    String::writeBinary(ofs);
    
    // Затем записываем время создания
    ofs.write(reinterpret_cast<const char*>(&creationTime), sizeof(creationTime));
}

// Чтение из бинарного файла
void TimedString::readBinary(std::ifstream& ifs) {
    // Сначала считываем базовую часть
    String::readBinary(ifs);
    
    // Затем считываем время создания
    ifs.read(reinterpret_cast<char*>(&creationTime), sizeof(creationTime));
}

// Оператор присваивания (TimedString)
TimedString& TimedString::operator=(const TimedString& other) {
    if (this != &other) {
        String::operator=(other);
        creationTime = other.creationTime;
    }
    return *this;
}

// Оператор присваивания (String)
TimedString& TimedString::operator=(const String& other) {
    String::operator=(other);
    creationTime = std::time(nullptr);
    return *this;
}