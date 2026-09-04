#include "String.h"
#include "OctalString.h"
#include "TimedString.h"
#include "Exceptions.h"
#include <iostream>
#include <new>

// Инициализация статической переменной
int String::count = 0;

// Конструктор по умолчанию
String::String() : str(nullptr), length(0) {
    count++;
}

// Конструктор с параметром
String::String(const char* s) {
    if (s) {
        length = strlen(s);
        if (length < 0) {
            throw InvalidArgumentException("Отрицательная длина строки");
        }
        try {
            str = new (std::nothrow) char[length + 1];
            if (!str) {
                throw MemoryException("Не удалось выделить память для строки длиной " + std::to_string(length));
            }
        strcpy(str, s);
        } catch (const std::bad_alloc&) {
            throw MemoryException("Недостаточно памяти для создания строки");
        }
    } else {
        length = 0;
        str = nullptr;
    }
    count++;
}

// Конструктор копирования
String::String(const String& other) {
    if (other.str) {
        length = other.length;
        try {
            str = new (std::nothrow) char[length + 1];
            if (!str) {
                throw MemoryException("Не удалось выделить память при копировании строки");
            }
        strcpy(str, other.str);
        } catch (const std::bad_alloc&) {
            throw MemoryException("Недостаточно памяти при копировании строки");
        }
    } else {
        length = 0;
        str = nullptr;
    }
    count++;
}

// Деструктор
String::~String() {
    delete[] str;
    count--;
}

// Изменение строки (теперь виртуальное)
void String::setString(const char* s) {
    delete[] str;
    if (s) {
        length = strlen(s);
        if (length < 0) {
            throw InvalidArgumentException("Отрицательная длина строки");
        }
        try {
            str = new (std::nothrow) char[length + 1];
            if (!str) {
                throw MemoryException("Не удалось выделить память для установки строки");
            }
        strcpy(str, s);
        } catch (const std::bad_alloc&) {
            throw MemoryException("Недостаточно памяти для установки строки");
        }
    } else {
        length = 0;
        str = nullptr;
    }
}

// Вывод строки
void String::print() const {
    if (str) {
        std::cout << getTypeName() << ": \"" << str << "\" (длина: " << length << ")" << std::endl;
    } else {
        std::cout << getTypeName() << " пуста." << std::endl;
    }
}

// Поиск подстроки
int String::findSubstring(const char* substr) const {
    if (!str || !substr) return -1;
    char* pos = strstr(str, substr);
    return pos ? pos - str : -1;
}

// Оператор сложения двух объектов
String String::operator+(const String& other) const {
    if (!str && !other.str) return String();
    if (!str) return String(other.str);
    if (!other.str) return String(str);
    
    int newLength = length + other.length;
    if (newLength < 0) {
        throw OutOfRangeException("Переполнение при сложении строк");
    }
    
    char* newStr = nullptr;
    try {
        newStr = new (std::nothrow) char[newLength + 1];
        if (!newStr) {
            throw MemoryException("Не удалось выделить память для сложения строк");
        }
    strcpy(newStr, str);
    strcat(newStr, other.str);
    String result(newStr);
    delete[] newStr;
    return result;
    } catch (const std::bad_alloc&) {
        delete[] newStr;
        throw MemoryException("Недостаточно памяти для сложения строк");
    }
}

// Оператор сложения с char*
String String::operator+(const char* other) const {
    if (!str && !other) return String();
    if (!str) return String(other);
    if (!other) return String(str);
    
    int otherLen = strlen(other);
    int newLength = length + otherLen;
    if (newLength < 0) {
        throw OutOfRangeException("Переполнение при сложении строк");
    }
    
    char* newStr = nullptr;
    try {
        newStr = new (std::nothrow) char[newLength + 1];
        if (!newStr) {
            throw MemoryException("Не удалось выделить память для сложения строк");
        }
    strcpy(newStr, str);
    strcat(newStr, other);
    String result(newStr);
    delete[] newStr;
    return result;
    } catch (const std::bad_alloc&) {
        delete[] newStr;
        throw MemoryException("Недостаточно памяти для сложения строк");
    }
}

// Оператор присваивания
String& String::operator=(const String& other) {
    if (this != &other) {
        delete[] str;
        if (other.str) {
            length = other.length;
            try {
                str = new (std::nothrow) char[length + 1];
                if (!str) {
                    throw MemoryException("Не удалось выделить память при присваивании");
                }
            strcpy(str, other.str);
            } catch (const std::bad_alloc&) {
                throw MemoryException("Недостаточно памяти при присваивании");
            }
        } else {
            length = 0;
            str = nullptr;
        }
    }
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
    int newLength = lhsLen + rhs.length;
    if (newLength < 0) {
        throw OutOfRangeException("Переполнение при сложении строк");
    }
    
    char* newStr = nullptr;
    try {
        newStr = new (std::nothrow) char[newLength + 1];
        if (!newStr) {
            throw MemoryException("Не удалось выделить память для сложения строк");
        }
    strcpy(newStr, lhs);
    strcat(newStr, rhs.str);
    String result(newStr);
    delete[] newStr;
    return result;
    } catch (const std::bad_alloc&) {
        delete[] newStr;
        throw MemoryException("Недостаточно памяти для сложения строк");
    }
}

// Дружественная функция: вывод в поток
std::ostream& operator<<(std::ostream& os, const String& obj) {
    if (obj.str) os << obj.str;
    else os << "";
    return os;
}

// Дружественная функция: ввод из потока
std::istream& operator>>(std::istream& is, String& obj) {
    char buffer[1000];
    is.getline(buffer, 1000);
    obj.setString(buffer);
    return is;
}

// Запись в текстовый файл
void String::writeToTextFile(std::ofstream& ofs) const {
    if (!ofs.is_open() || ofs.fail()) {
        throw FileException("Ошибка записи в файл: файл не открыт");
    }
    
    // Сначала записываем тип объекта
    ofs << static_cast<int>(getType()) << " ";
    
    if (ofs.fail() || ofs.bad()) {
        throw FileException("Ошибка записи типа в файл");
    }
    
    // Затем длину и строку
    if (str) {
        ofs << length << " ";
        if (ofs.fail() || ofs.bad()) {
            throw FileException("Ошибка записи длины в файл");
        }
        ofs.write(str, length);
        if (ofs.fail() || ofs.bad()) {
            throw FileException("Ошибка записи данных строки в файл");
        }
    } else {
        ofs << "0 ";
        if (ofs.fail() || ofs.bad()) {
            throw FileException("Ошибка записи в файл");
        }
    }
}

// Чтение из текстового файла
void String::readFromTextFile(std::ifstream& ifs) {
    if (!ifs.is_open() || ifs.fail()) {
        throw FileException("Ошибка чтения из файла: файл не открыт или поврежден");
    }
    
    // Пропускаем тип (он уже считан фабричным методом)
    int len;
    ifs >> len;
    
    if (ifs.fail() || ifs.bad()) {
        throw FileException("Ошибка чтения длины строки из файла");
    }
    
    if (len < 0) {
        throw InvalidArgumentException("Отрицательная длина строки в файле");
    }
    
    if (len > 1000000) { // Защита от слишком больших значений
        throw OutOfRangeException("Слишком большая длина строки в файле: " + std::to_string(len));
    }
    
    ifs.get(); // Пропускаем пробел
    
    if (len > 0) {
        char* buffer = nullptr;
        try {
            buffer = new (std::nothrow) char[len + 1];
            if (!buffer) {
                throw MemoryException("Не удалось выделить память для чтения строки из файла");
            }
        ifs.read(buffer, len);
            if (ifs.fail() || ifs.bad()) {
                delete[] buffer;
                throw FileException("Ошибка чтения данных строки из файла");
            }
        buffer[len] = '\0';
        this->setString(buffer);
        delete[] buffer;
        } catch (const std::bad_alloc&) {
            delete[] buffer;
            throw MemoryException("Недостаточно памяти для чтения строки из файла");
        }
    } else {
        this->setString("");
    }
}

// Запись в бинарный файл
void String::writeBinary(std::ofstream& ofs) const {
    if (!ofs.is_open() || ofs.fail()) {
        throw FileException("Ошибка записи в бинарный файл: файл не открыт");
    }
    
    // Сначала тип
    StringType type = getType();
    ofs.write(reinterpret_cast<const char*>(&type), sizeof(type));
    
    if (ofs.fail() || ofs.bad()) {
        throw FileException("Ошибка записи типа в бинарный файл");
    }
    
    // Затем длину и данные
    ofs.write(reinterpret_cast<const char*>(&length), sizeof(length));
    if (ofs.fail() || ofs.bad()) {
        throw FileException("Ошибка записи длины в бинарный файл");
    }
    
    if (length > 0) {
        ofs.write(str, length);
        if (ofs.fail() || ofs.bad()) {
            throw FileException("Ошибка записи данных строки в бинарный файл");
        }
    }
}

// Чтение из бинарного файла
void String::readBinary(std::ifstream& ifs) {
    if (!ifs.is_open() || ifs.fail()) {
        throw FileException("Ошибка чтения из бинарного файла: файл не открыт или поврежден");
    }
    
    // Читаем тип (для согласованности с writeBinary)
    StringType type;
    ifs.read(reinterpret_cast<char*>(&type), sizeof(type));
    
    if (ifs.fail() || ifs.bad()) {
        throw FileException("Ошибка чтения типа из бинарного файла");
    }
    
    // Читаем длину и данные
    delete[] str;
    ifs.read(reinterpret_cast<char*>(&length), sizeof(length));
    
    if (ifs.fail() || ifs.bad()) {
        throw FileException("Ошибка чтения длины из бинарного файла");
    }
    
    if (length < 0) {
        throw InvalidArgumentException("Отрицательная длина строки в бинарном файле");
    }
    
    if (length > 1000000) { // Защита от слишком больших значений
        throw OutOfRangeException("Слишком большая длина строки в бинарном файле: " + std::to_string(length));
    }
    
    if (length > 0) {
        try {
            str = new (std::nothrow) char[length + 1];
            if (!str) {
                throw MemoryException("Не удалось выделить память для чтения строки из бинарного файла");
            }
        ifs.read(str, length);
            if (ifs.fail() || ifs.bad()) {
                delete[] str;
                str = nullptr;
                throw FileException("Ошибка чтения данных строки из бинарного файла");
            }
        str[length] = '\0';
        } catch (const std::bad_alloc&) {
            throw MemoryException("Недостаточно памяти для чтения строки из бинарного файла");
        }
    } else {
        str = nullptr;
    }
}

// Получение количества объектов
int String::getCount() {
    return count;
}

// Создание объекта по типу
String* String::createFromType(StringType type) {
    switch (type) {
        case StringType::BASE_STRING:
            return new String();
        case StringType::OCTAL_STRING:
            return new OctalString();
        case StringType::TIMED_STRING:
            return new TimedString();
        default:
            return nullptr;
    }
}