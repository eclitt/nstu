#include <iostream>
#include <string>
using namespace std;

class Date {
    private:
        int day;
        int month;
    public:
        Date(): day(0), month(0) {}
        Date(int d, int m): day(d), month(m) {}

        void print_date() const {  // Переименовал для ясности
            cout << "Date is " << day << "/" << month << " "; 
        }
        
        // Геттеры для получения значений (если нужны)
        int getDay() const { return day; }
        int getMonth() const { return month; }
};

class Time {
    private:
        int hour;
        int minute;
        int second;
    public:
        Time(): hour(0), minute(0), second(0) {}
        Time(int h, int m, int s): hour(h), minute(m), second(s) {}

        void print_time() const {
            cout << "Duration is " << hour << ":" << minute << ":" << second << " "; 
        }
        
        // Геттеры
        int getHour() const { return hour; }
        int getMinute() const { return minute; }
        int getSecond() const { return second; }
};

class Call {
    private:
        bool is_out;
        Date date;
        Time duration;  // Было time, исправил на duration
        string number;
    public:
        // Конструктор по умолчанию
        Call(): is_out(false), date(), duration(), number("0") {}  // False -> false
        
        // Конструктор с параметрами
        Call(bool io, const Date& d, const Time& t, string num): 
            is_out(io), date(d), duration(t), number(num) {}  // Исправил time на duration, sting на string

        void print_call() const {
            cout << "Call " << number << " ";
            date.print_date();  // print_date, а не get_date
            cout << " ";
            duration.print_time();  // print_time, а не get_time
            cout << " ";
            cout << "Is out: " << (is_out ? "Yes" : "No") << endl;  // Добавил endl
        }
};

class Journal {
    private:
        Call* calls;
        int capacity;
        int count;
    
        void extend_capacity() {
            int new_capacity = capacity * 2;  // Добавил тип int
            Call* newCalls = new Call[new_capacity];  // Добавил точку с запятой

            for (int i = 0; i < count; i++) {
                newCalls[i] = calls[i];
            }
            delete[] calls;
            calls = newCalls;  // Добавил точку с запятой
            capacity = new_capacity;
            cout << "Expand capacity to " << capacity << endl;  // Исправил x на ;
        }
        
    public:
        Journal(int initial_capacity = 10): capacity(initial_capacity), count(0) {
            calls = new Call[capacity];
        }

        ~Journal() {
            delete[] calls;
        }

        void add_call(const Call& call) {
            if (count >= capacity) {
                extend_capacity();
            }
            calls[count] = call;
            count++;
            cout << "New call added: ";
            call.print_call();
        }

        void print_calls() {
            if (count == 0) {
                cout << "No calls in journal" << endl;
                return;
            }
            
            cout << "\n=== JOURNAL CALLS ===" << endl;
            for (int i = 0; i < count; i++) {
                cout << "Call #" << i + 1 << ": ";
                calls[i].print_call();  // Исправил -> на . (calls - массив объектов)
            }
            cout << "=====================" << endl;
        }
};

int main() {
    // ПРАВИЛЬНОЕ создание объектов:
    
    // 1. Создаем объекты Date и Time
    Date d1(15, 3);        // 15 марта
    Time t1(12, 30, 45);   // 12:30:45
    
    // 2. Создаем объект Call
    Call myCall(false, d1, t1, "+7-912-345-67-89");
    
    // 3. Создаем журнал
    Journal myJournal;  // Убрал скобки! Journal myJournal(); - это объявление функции!
    
    // 4. Добавляем вызов в журнал
    myJournal.add_call(myCall);
    
    // 5. Добавляем еще несколько вызовов напрямую
    myJournal.add_call(Call(true, Date(16, 3), Time(14, 20, 0), "+7-923-456-78-90"));
    myJournal.add_call(Call(false, Date(16, 3), Time(9, 15, 30), "+7-934-567-89-01"));
    
    // 6. Выводим все вызовы
    myJournal.print_calls();
    
    return 0;
}