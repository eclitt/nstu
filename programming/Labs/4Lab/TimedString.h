#ifndef TIMEDSTRING_H
#define TIMEDSTRING_H

#include "String.h"
#include <ctime>
#include <iomanip>
#include <sstream>

class TimedString : public String {
private:
    std::time_t creationTime;  // Время создания объекта
    
public:
    // Конструкторы
    TimedString();                            // Конструктор по умолчанию
    TimedString(const char* s);               // Конструктор с параметром
    TimedString(const TimedString& other);    // Конструктор копирования
    TimedString(const String& other);         // Конструктор преобразования
    virtual ~TimedString();                   // Деструктор
    
    // Переопределенные методы
    virtual void print() const override;      // Вывод строки с временем создания
    virtual StringType getType() const override { return StringType::TIMED_STRING; }
    virtual std::string getTypeName() const override { return "TimedString"; }
    
    // Новые методы
    std::time_t getCreationTime() const { return creationTime; }
    std::string getCreationTimeString() const;
    
    // Методы для работы с файлами
    virtual void writeToTextFile(std::ofstream& ofs) const override;
    virtual void readFromTextFile(std::ifstream& ifs) override;
    virtual void writeBinary(std::ofstream& ofs) const override;
    virtual void readBinary(std::ifstream& ifs) override;
    
    // Перегруженные операторы
    TimedString& operator=(const TimedString& other);
    TimedString& operator=(const String& other);
};

#endif