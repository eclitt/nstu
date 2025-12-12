#ifndef DATE_STUDENT_H
#define DATE_STUDENT_H

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <ctime>

// ===================== Класс Date =====================
class Date {
private:
    int day;
    int month;
    int year;

    bool isValidDate(int d, int m, int y) const {
        if (y < 1900 || y > 2100) return false;
        if (m < 1 || m > 12) return false;
        
        int daysInMonth[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
        
        if (m == 2) {
            bool isLeap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
            if (isLeap) daysInMonth[1] = 29;
        }
        
        return d >= 1 && d <= daysInMonth[m - 1];
    }

public:
    Date() : day(1), month(1), year(2000) {}
    
    Date(int d, int m, int y) {
        if (isValidDate(d, m, y)) {
            day = d;
            month = m;
            year = y;
        } else {
            day = 1;
            month = 1;
            year = 2000;
        }
    }
    
    int getDay() const { return day; }
    int getMonth() const { return month; }
    int getYear() const { return year; }
    
    void setDay(int d) { if (d >= 1 && d <= 31) day = d; }
    void setMonth(int m) { if (m >= 1 && m <= 12) month = m; }
    void setYear(int y) { if (y >= 1900 && y <= 2100) year = y; }
    
    // Операторы сравнения
    bool operator<(const Date& other) const {
        if (year != other.year) return year < other.year;
        if (month != other.month) return month < other.month;
        return day < other.day;
    }
    
    bool operator>(const Date& other) const { return other < *this; }
    
    bool operator==(const Date& other) const {
        return day == other.day && month == other.month && year == other.year;
    }
    
    bool operator!=(const Date& other) const { return !(*this == other); }
    
    bool operator<=(const Date& other) const {
        return *this < other || *this == other;
    }
    
    bool operator>=(const Date& other) const {
        return *this > other || *this == other;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Date& date) {
        os << std::setw(2) << std::setfill('0') << date.day << "."
           << std::setw(2) << std::setfill('0') << date.month << "."
           << date.year;
        return os;
    }
    
    friend std::istream& operator>>(std::istream& is, Date& date) {
        std::string input;
        is >> input;
        std::istringstream iss(input);
        char delimiter;
        int d, m, y;
        
        if (iss >> d >> delimiter >> m >> delimiter >> y) {
            if (date.isValidDate(d, m, y)) {
                date.day = d;
                date.month = m;
                date.year = y;
            }
        }
        return is;
    }
    
    void serialize(std::ostream& os) const {
        os.write(reinterpret_cast<const char*>(&day), sizeof(day));
        os.write(reinterpret_cast<const char*>(&month), sizeof(month));
        os.write(reinterpret_cast<const char*>(&year), sizeof(year));
    }
    
    void deserialize(std::istream& is) {
        is.read(reinterpret_cast<char*>(&day), sizeof(day));
        is.read(reinterpret_cast<char*>(&month), sizeof(month));
        is.read(reinterpret_cast<char*>(&year), sizeof(year));
    }
    
    static Date getCurrentDate() {
        std::time_t t = std::time(nullptr);
        std::tm* now = std::localtime(&t);
        return Date(now->tm_mday, now->tm_mon + 1, now->tm_year + 1900);
    }
    
    int getAge() const {
        Date current = getCurrentDate();
        int age = current.year - year;
        if (current.month < month || (current.month == month && current.day < day)) {
            age--;
        }
        return age;
    }
};

// ===================== Класс Student =====================
class Student {
private:
    std::string lastName;
    Date birthDate;
    Date admissionDate;
    Date expulsionDate;
    std::string address;
    std::string group;
    
public:
    Student() : lastName(""), birthDate(1, 1, 2000), 
                admissionDate(1, 9, 2020), expulsionDate(30, 6, 2024),
                address(""), group("") {}
    
    Student(const std::string& ln, const Date& bd, const Date& ad, 
            const Date& ed, const std::string& addr, const std::string& gr)
        : lastName(ln), birthDate(bd), admissionDate(ad), 
          expulsionDate(ed), address(addr), group(gr) {}
    
    std::string getLastName() const { return lastName; }
    Date getBirthDate() const { return birthDate; }
    Date getAdmissionDate() const { return admissionDate; }
    Date getExpulsionDate() const { return expulsionDate; }
    std::string getAddress() const { return address; }
    std::string getGroup() const { return group; }
    
    void setLastName(const std::string& ln) { lastName = ln; }
    void setBirthDate(const Date& bd) { birthDate = bd; }
    void setAdmissionDate(const Date& ad) { admissionDate = ad; }
    void setExpulsionDate(const Date& ed) { expulsionDate = ed; }
    void setAddress(const std::string& addr) { address = addr; }
    void setGroup(const std::string& gr) { group = gr; }
    
    bool operator<(const Student& other) const { return lastName < other.lastName; }
    bool operator>(const Student& other) const { return lastName > other.lastName; }
    bool operator==(const Student& other) const {
        return lastName == other.lastName && 
               birthDate == other.birthDate &&
               group == other.group;
    }
    
    bool compareByBirthDate(const Student& other) const { return birthDate < other.birthDate; }
    bool compareByAdmissionDate(const Student& other) const { return admissionDate < other.admissionDate; }
    bool compareByGroup(const Student& other) const {
        if (group != other.group) return group < other.group;
        return lastName < other.lastName;
    }
    
    void serialize(std::ostream& os) const {
        size_t len = lastName.length();
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(lastName.c_str(), len);
        birthDate.serialize(os);
        admissionDate.serialize(os);
        expulsionDate.serialize(os);
        
        len = address.length();
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(address.c_str(), len);
        
        len = group.length();
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(group.c_str(), len);
    }
    
    void deserialize(std::istream& is) {
        size_t len;
        
        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        char* buffer = new char[len + 1];
        is.read(buffer, len);
        buffer[len] = '\0';
        lastName = buffer;
        delete[] buffer;
        
        birthDate.deserialize(is);
        admissionDate.deserialize(is);
        expulsionDate.deserialize(is);
        
        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        buffer = new char[len + 1];
        is.read(buffer, len);
        buffer[len] = '\0';
        address = buffer;
        delete[] buffer;
        
        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        buffer = new char[len + 1];
        is.read(buffer, len);
        buffer[len] = '\0';
        group = buffer;
        delete[] buffer;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Student& student) {
        os << "Фамилия: " << student.lastName << "\n"
           << "Дата рождения: " << student.birthDate 
           << " (возраст: " << student.birthDate.getAge() << " лет)\n"
           << "Дата поступления: " << student.admissionDate << "\n"
           << "Дата отчисления: " << student.expulsionDate << "\n"
           << "Адрес: " << student.address << "\n"
           << "Группа: " << student.group << "\n";
        
        Date current = Date::getCurrentDate();
        int course = current.getYear() - student.admissionDate.getYear();
        if (current.getMonth() < 9) course--;
        if (course < 1) course = 1;
        if (course > 5) course = 5;
        
        os << "Курс: " << course << "\n"
           << "Статус: " << (student.isCurrentlyStudying() ? "обучается" : "отчислен") << "\n"
           << "--------------------------------------------------\n";
        return os;
    }
    
    bool isCurrentlyStudying() const {
        Date current = Date::getCurrentDate();
        return current >= admissionDate && current <= expulsionDate;
    }
};

#endif // DATE_STUDENT_H