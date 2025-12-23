#include "StringTree.h"
#include "String.h"
#include "OctalString.h"
#include "TimedString.h"
#include <iostream>

int main() {
    std::cout << "=== Демонстрация работы с полиморфизмом в бинарном дереве ===" << std::endl;
    
    // Создаем дерево
    StringTree tree;
    
    // Создаем объекты разных типов (полиморфизм)
    std::cout << "\n1. Создание объектов разных типов:" << std::endl;
    String* str1 = new String("Hello");
    String* str2 = new OctalString("123");  // Восьмеричное число
    String* str3 = new TimedString("World");
    String* str4 = new OctalString("777");
    String* str5 = new TimedString("Test");
    
    // Добавляем в дерево (все как String*, но сохраняют свой тип)
    std::cout << "\n2. Добавление объектов в дерево:" << std::endl;
    tree.add(str1);
    tree.add(str2);
    tree.add(str3);
    tree.add(str4);
    tree.add(str5);
    
    std::cout << "Размер дерева: " << tree.size() << std::endl;
    
    // Демонстрация полиморфизма
    std::cout << "\n3. Демонстрация полиморфизма:" << std::endl;
    tree.demonstratePolymorphism();
    
    // Вывод всех элементов
    std::cout << "\n4. Вывод всех элементов дерева:" << std::endl;
    tree.printAll();
    
    // Сохранение в файлы
    std::cout << "\n5. Сохранение дерева в файлы:" << std::endl;
    tree.saveToTextFile("tree.txt");
    tree.saveToBinaryFile("tree.bin");
    
    // Создаем новое дерево и загружаем из файла
    std::cout << "\n6. Загрузка дерева из файлов:" << std::endl;
    StringTree tree2;
    tree2.loadFromTextFile("tree.txt");
    std::cout << "\nЗагруженное дерево (из текстового файла):" << std::endl;
    tree2.printAll();
    
    StringTree tree3;
    tree3.loadFromBinaryFile("tree.bin");
    std::cout << "\nЗагруженное дерево (из бинарного файла):" << std::endl;
    tree3.printAll();
    
    // Поиск элементов
    std::cout << "\n7. Поиск элементов:" << std::endl;
    int idx = tree.indexOf("Hello");
    if (idx != -1) {
        std::cout << "Найден элемент 'Hello' на позиции: " << idx << std::endl;
    }
    
    String* found = tree.find("123");
    if (found) {
        std::cout << "Найден элемент '123': ";
        found->print();
    }
    
    std::cout << "\n=== Программа завершена ===" << std::endl;
    return 0;
}

