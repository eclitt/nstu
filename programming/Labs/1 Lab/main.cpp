#include "String.h"
#include <iostream>

int main() {
    // Создание объектов разными конструкторами
    String s1;                     // Конструктор по умолчанию
    s1.print();

    String s2("Hello");            // Конструктор с параметром
    s2.print();

    String s3 = s2;                // Конструктор копирования
    s3.print();

    // Изменение строки
    s1.setString("World");
    s1.print();

    // Поиск подстроки
    int index = s2.findSubstring("ell");
    std::cout << "Подстрока 'ell' найдена в позиции: " << index << std::endl;

    // Объединение строк
    String s4 = s2.concat(s1);
    s4.print();

    // Использование статического метода
    std::cout << "Текущее количество объектов: " << String::getCount() << std::endl;

    return 0;
}