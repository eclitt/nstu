#include "String.h"
#include "OctalString.h"
#include "TimedString.h"
#include <iostream>

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
        str = new char[length + 1];
        strcpy(str, s);
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
        str = new char[length + 1];
        strcpy(str, other.str);
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
        str = new char[length + 1];
        strcpy(str, s);
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
    
    char* newStr = new char[length + other.length + 1];
    strcpy(newStr, str);
    strcat(newStr, other.str);
    String result(newStr);
    delete[] newStr;
    return result;
}

// Оператор сложения с char*
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
    // Сначала записываем тип объекта
    ofs << static_cast<int>(getType()) << " ";
    
    // Затем длину и строку
    if (str) {
        ofs << length << " ";
        ofs.write(str, length);
    } else {
        ofs << "0 ";
    }
}

// Чтение из текстового файла
void String::readFromTextFile(std::ifstream& ifs) {
    // Пропускаем тип (он уже считан фабричным методом)
    int len;
    ifs >> len;
    ifs.get(); // Пропускаем пробел
    
    if (len > 0) {
        char* buffer = new char[len + 1];
        ifs.read(buffer, len);
        buffer[len] = '\0';
        this->setString(buffer);
        delete[] buffer;
    } else {
        this->setString("");
    }
}

// Запись в бинарный файл
void String::writeBinary(std::ofstream& ofs) const {
    // Сначала тип
    StringType type = getType();
    ofs.write(reinterpret_cast<const char*>(&type), sizeof(type));
    
    // Затем длину и данные
    ofs.write(reinterpret_cast<const char*>(&length), sizeof(length));
    if (length > 0) {
        ofs.write(str, length);
    }
}

// Чтение из бинарного файла
void String::readBinary(std::ifstream& ifs) {
    // Читаем тип (для согласованности с writeBinary)
    StringType type;
    ifs.read(reinterpret_cast<char*>(&type), sizeof(type));
    
    // Читаем длину и данные
    delete[] str;
    ifs.read(reinterpret_cast<char*>(&length), sizeof(length));
    if (length > 0) {
        str = new char[length + 1];
        ifs.read(str, length);
        str[length] = '\0';
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