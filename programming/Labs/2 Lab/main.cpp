#include "String.h"
#include <iostream>

using namespace std;

int main() {
    cout << "=== Демонстрация перегруженных операций для класса String ===" << endl << endl;

    // Создание объектов
    cout << "1. Создание объектов:" << endl;
    String s1("Hello");
    String s2("World");
    String s3;
    String s4 = s1;  // Конструктор копирования
    
    cout << endl << "2. Оператор присваивания:" << endl;
    s3 = s2;  // Оператор присваивания
    cout << "s3 = s2: " << s3 << endl;

    cout << endl << "3. Операции сложения:" << endl;
    // String + String
    String s5 = s1 + s2;
    cout << "s1 + s2: " << s5 << endl;
    
    // String + char*
    String s6 = s1 + " C++";
    cout << "s1 + \" C++\": " << s6 << endl;
    
    // char* + String
    String s7 = "C++ " + s2;
    cout << "\"C++ \" + s2: " << s7 << endl;

    cout << endl << "4. Операции сравнения:" << endl;
    cout << "s1 == s4: " << (s1 == s4 ? "true" : "false") << endl;
    cout << "s1 == s2: " << (s1 == s2 ? "true" : "false") << endl;
    cout << "s1 != s2: " << (s1 != s2 ? "true" : "false") << endl;
    cout << "s1 < s2: " << (s1 < s2 ? "true" : "false") << endl;
    cout << "s1 > s2: " << (s1 > s2 ? "true" : "false") << endl;
    cout << "s1 <= s4: " << (s1 <= s4 ? "true" : "false") << endl;
    cout << "s1 >= s4: " << (s1 >= s4 ? "true" : "false") << endl;

    cout << endl << "5. Комбинированные операции:" << endl;
    String s8 = s1 + " " + s2 + "!";
    cout << "s1 + \" \" + s2 + \"!\": " << s8 << endl;

    cout << endl << "6. Операторы ввода/вывода:" << endl;
    String s9;
    cout << "Введите строку для s9: ";
    cin >> s9;
    cout << "Вы ввели: " << s9 << endl;

    cout << endl << "7. Статический счетчик объектов:" << endl;
    cout << "Текущее количество объектов: " << String::getCount() << endl;

    cout << endl << "8. Демонстрация остальных методов:" << endl;
    cout << "Длина s1: " << s1.getLength() << endl;
    cout << "Поиск 'ell' в s1: позиция " << s1.findSubstring("ell") << endl;

    cout << endl << "=== Завершение программы ===" << endl;
    return 0;
}