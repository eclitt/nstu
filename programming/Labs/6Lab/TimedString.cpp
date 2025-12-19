#include "TimedString.h"
#include "Exceptions.h"
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
    if (!ofs.is_open() || ofs.fail()) {
        throw FileException("Ошибка записи TimedString в файл: файл не открыт");
    }
    
    // Сначала записываем тип объекта
    ofs << static_cast<int>(getType()) << " ";
    if (ofs.fail() || ofs.bad()) {
        throw FileException("Ошибка записи типа TimedString в файл");
    }
    
    // Затем длину строки и саму строку
    ofs << length << " ";
    if (ofs.fail() || ofs.bad()) {
        throw FileException("Ошибка записи длины TimedString в файл");
    }
    
    if (length > 0) {
        ofs.write(str, length);
        if (ofs.fail() || ofs.bad()) {
            throw FileException("Ошибка записи данных TimedString в файл");
        }
    }
    
    // И время создания
    ofs << " " << creationTime;
    if (ofs.fail() || ofs.bad()) {
        throw FileException("Ошибка записи времени создания TimedString в файл");
    }
}

// Чтение из текстового файла
void TimedString::readFromTextFile(std::ifstream& ifs) {
    if (!ifs.is_open() || ifs.fail()) {
        throw FileException("Ошибка чтения TimedString из файла: файл не открыт или поврежден");
    }
    
    // Считываем длину строки
    ifs >> length;
    if (ifs.fail() || ifs.bad()) {
        throw FileException("Ошибка чтения длины TimedString из файла");
    }
    
    if (length < 0 || length > 1000000) {
        throw OutOfRangeException("Некорректная длина TimedString в файле: " + std::to_string(length));
    }
    
    ifs.get(); // Пропускаем пробел
    
    if (length > 0) {
        char* buffer = nullptr;
        try {
            buffer = new (std::nothrow) char[length + 1];
            if (!buffer) {
                throw MemoryException("Не удалось выделить память для чтения TimedString из файла");
            }
            ifs.read(buffer, length);
            if (ifs.fail() || ifs.bad()) {
                delete[] buffer;
                throw FileException("Ошибка чтения данных TimedString из файла");
            }
            buffer[length] = '\0';
            this->setString(buffer);
            delete[] buffer;
        } catch (const std::bad_alloc&) {
            delete[] buffer;
            throw MemoryException("Недостаточно памяти для чтения TimedString из файла");
        }
    } else {
        this->setString("");
    }
    
    // Считываем время создания
    ifs >> creationTime;
    if (ifs.fail() || ifs.bad()) {
        throw FileException("Ошибка чтения времени создания TimedString из файла");
    }
}

// Запись в бинарный файл
void TimedString::writeBinary(std::ofstream& ofs) const {
    // Сначала вызываем метод базового класса
    String::writeBinary(ofs);
    
    // Затем записываем время создания
    ofs.write(reinterpret_cast<const char*>(&creationTime), sizeof(creationTime));
    if (ofs.fail() || ofs.bad()) {
        throw FileException("Ошибка записи времени создания TimedString в бинарный файл");
    }
}

// Чтение из бинарного файла
void TimedString::readBinary(std::ifstream& ifs) {
    // Сначала считываем базовую часть
    String::readBinary(ifs);
    
    // Затем считываем время создания
    ifs.read(reinterpret_cast<char*>(&creationTime), sizeof(creationTime));
    if (ifs.fail() || ifs.bad()) {
        throw FileException("Ошибка чтения времени создания TimedString из бинарного файла");
    }
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