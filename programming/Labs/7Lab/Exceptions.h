#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <stdexcept>
#include <string>

// Базовый класс для исключений программы
class StringException : public std::exception {
protected:
    std::string message;
public:
    explicit StringException(const std::string& msg) : message(msg) {}
    virtual const char* what() const noexcept override {
        return message.c_str();
    }
};

// Исключение для ошибок выделения памяти
class MemoryException : public StringException {
public:
    explicit MemoryException(const std::string& msg = "Ошибка выделения памяти") 
        : StringException(msg) {}
};

// Исключение для выхода за пределы диапазона
class OutOfRangeException : public StringException {
public:
    explicit OutOfRangeException(const std::string& msg = "Выход за пределы допустимого диапазона") 
        : StringException(msg) {}
};

// Исключение для деления на ноль
class DivisionByZeroException : public StringException {
public:
    explicit DivisionByZeroException(const std::string& msg = "Деление на ноль") 
        : StringException(msg) {}
};

// Исключение для ошибок работы с файлами
class FileException : public StringException {
public:
    explicit FileException(const std::string& msg = "Ошибка работы с файлом") 
        : StringException(msg) {}
};

// Исключение для некорректных аргументов
class InvalidArgumentException : public StringException {
public:
    explicit InvalidArgumentException(const std::string& msg = "Некорректный аргумент") 
        : StringException(msg) {}
};

#endif

