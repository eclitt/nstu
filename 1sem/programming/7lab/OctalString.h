#ifndef OCTALSTRING_H
#define OCTALSTRING_H

#include "String.h"
#include <stdexcept>
#include <string>

class OctalString : public String {
private:
    long long number;   // Числовое значение в восьмеричной системе
    
    // Вспомогательные методы
    bool isValidOctal(const char* s) const;  // Проверка, является ли строка восьмеричным числом
    long long octalToDecimal(const char* octal) const;  // Перевод из восьмеричной в десятичную
    std::string decimalToOctal(long long decimal) const; // Перевод из десятичной в восьмеричную
    
public:
    // Конструкторы
    OctalString();                                  // Конструктор по умолчанию
    OctalString(const char* s);                     // Конструктор с параметром (строки)
    OctalString(long long num);                     // Конструктор с параметром (числа)
    OctalString(const OctalString& other);          // Конструктор копирования
    OctalString(const String& other);               // Конструктор преобразования
    virtual ~OctalString();                         // Деструктор
    
    // Переопределенные методы
    virtual void print() const override;            // Вывод строки с числовым значением
    virtual StringType getType() const override { return StringType::OCTAL_STRING; }
    virtual std::string getTypeName() const override { return "OctalString"; }
    virtual void setString(const char* s) override; // Изменение строки с проверкой
    
    // Новые методы
    long long getNumber() const { return number; }  // Получение числового значения
    void setNumber(long long num);                  // Установка числового значения
    
    // Методы для работы с файлами
    virtual void writeToTextFile(std::ofstream& ofs) const override;
    virtual void readFromTextFile(std::ifstream& ifs) override;
    virtual void writeBinary(std::ofstream& ofs) const override;
    virtual void readBinary(std::ifstream& ifs) override;
    
    // Перегруженные операторы
    OctalString operator+(const OctalString& other) const;
    OctalString& operator=(const OctalString& other);
    OctalString& operator=(const String& other);
    
    // Операции с числами
    OctalString operator+(long long num) const;
    OctalString operator-(long long num) const;
    OctalString operator/(long long num) const;  // Деление (может вызвать исключение при делении на ноль)
    OctalString operator%(long long num) const;  // Остаток от деления (может вызвать исключение при делении на ноль)
    bool operator==(const OctalString& other) const;
};

#endif