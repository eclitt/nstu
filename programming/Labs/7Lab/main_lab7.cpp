#include "Tree.h"
#include "TemplateFunctions.h"
#include "String.h"
#include "OctalString.h"
#include "TimedString.h"
#include "Exceptions.h"
#include <iostream>
#include <iomanip>

using namespace std;

// Функции-предикаты для шаблонных функций
bool isEven(const int& value) {
    return value % 2 == 0;
}

bool isPositive(const float& value) {
    return value > 0.0f;
}

bool isLongString(const String& str) {
    return str.getLength() > 5;
}

// Тестирование дерева с типом int
void testIntTree() {
    cout << "\n========================================" << endl;
    cout << "ТЕСТИРОВАНИЕ ДЕРЕВА С ТИПОМ int" << endl;
    cout << "========================================" << endl;
    
    Tree<int> intTree;
    
    // Добавление элементов
    cout << "\n1. Добавление элементов:" << endl;
    intTree.add(10);
    intTree.add(5);
    intTree.add(15);
    intTree.add(3);
    intTree.add(7);
    intTree.add(12);
    intTree.add(20);
    
    cout << "Размер дерева: " << intTree.size() << endl;
    intTree.printAll();
    
    // Вставка по индексу
    cout << "\n2. Вставка элемента 8 на позицию 3:" << endl;
    intTree.insertAt(3, 8);
    intTree.printAll();
    
    // Поиск элемента
    cout << "\n3. Поиск элемента 15:" << endl;
    int index = intTree.indexOf(15);
    if (index != -1) {
        cout << "Элемент 15 найден на позиции " << index << endl;
    } else {
        cout << "Элемент 15 не найден" << endl;
    }
    
    // Использование шаблонных функций
    cout << "\n4. Использование шаблонных функций:" << endl;
    try {
        int maxVal = findMax(intTree);
        int minVal = findMin(intTree);
        cout << "Максимальный элемент: " << maxVal << endl;
        cout << "Минимальный элемент: " << minVal << endl;
        
        int evenCount = countIf(intTree, isEven);
        cout << "Количество четных элементов: " << evenCount << endl;
    } catch (const exception& e) {
        cout << "Ошибка: " << e.what() << endl;
    }
    
    // Удаление элемента
    cout << "\n5. Удаление элемента на позиции 2:" << endl;
    intTree.removeAt(2);
    intTree.printAll();
    
    // Получение элемента по индексу
    cout << "\n6. Получение элемента по индексу 3:" << endl;
    try {
        int value = intTree.getAt(3);
        cout << "Элемент на позиции 3: " << value << endl;
    } catch (const exception& e) {
        cout << "Ошибка: " << e.what() << endl;
    }
    
    // Копирование дерева
    cout << "\n7. Копирование дерева:" << endl;
    Tree<int> copiedTree = copyTree(intTree);
    cout << "Оригинальное дерево: ";
    printTreeFormatted(intTree);
    cout << "Скопированное дерево: ";
    printTreeFormatted(copiedTree);
}

// Тестирование дерева с типом float
void testFloatTree() {
    cout << "\n========================================" << endl;
    cout << "ТЕСТИРОВАНИЕ ДЕРЕВА С ТИПОМ float" << endl;
    cout << "========================================" << endl;
    
    Tree<float> floatTree;
    
    // Добавление элементов
    cout << "\n1. Добавление элементов:" << endl;
    floatTree.add(3.14f);
    floatTree.add(2.71f);
    floatTree.add(1.41f);
    floatTree.add(1.73f);
    floatTree.add(0.577f);
    
    cout << "Размер дерева: " << floatTree.size() << endl;
    floatTree.printAll();
    
    // Вставка по индексу
    cout << "\n2. Вставка элемента 2.5 на позицию 2:" << endl;
    floatTree.insertAt(2, 2.5f);
    floatTree.printAll();
    
    // Поиск элемента
    cout << "\n3. Поиск элемента 3.14:" << endl;
    int index = floatTree.indexOf(3.14f);
    if (index != -1) {
        cout << "Элемент 3.14 найден на позиции " << index << endl;
    } else {
        cout << "Элемент 3.14 не найден" << endl;
    }
    
    // Использование шаблонных функций
    cout << "\n4. Использование шаблонных функций:" << endl;
    try {
        float maxVal = findMax(floatTree);
        float minVal = findMin(floatTree);
        cout << fixed << setprecision(3);
        cout << "Максимальный элемент: " << maxVal << endl;
        cout << "Минимальный элемент: " << minVal << endl;
        
        int positiveCount = countIf(floatTree, isPositive);
        cout << "Количество положительных элементов: " << positiveCount << endl;
    } catch (const exception& e) {
        cout << "Ошибка: " << e.what() << endl;
    }
    
    // Объединение деревьев
    cout << "\n5. Объединение двух деревьев:" << endl;
    Tree<float> floatTree2;
    floatTree2.add(10.5f);
    floatTree2.add(20.3f);
    
    Tree<float> mergedTree = mergeTrees(floatTree, floatTree2);
    cout << "Первое дерево: ";
    printTreeFormatted(floatTree);
    cout << "Второе дерево: ";
    printTreeFormatted(floatTree2);
    cout << "Объединенное дерево: ";
    printTreeFormatted(mergedTree);
}

// Тестирование дерева с классом String из ЛР1
void testStringTree() {
    cout << "\n========================================" << endl;
    cout << "ТЕСТИРОВАНИЕ ДЕРЕВА С КЛАССОМ String" << endl;
    cout << "========================================" << endl;
    
    Tree<String> stringTree;
    
    // Добавление элементов
    cout << "\n1. Добавление элементов:" << endl;
    stringTree.add(String("Первый"));
    stringTree.add(String("Второй"));
    stringTree.add(String("Третий"));
    stringTree.add(String("Четвертый"));
    stringTree.add(String("Пятый"));
    
    cout << "Размер дерева: " << stringTree.size() << endl;
    stringTree.printAll();
    
    // Вставка по индексу
    cout << "\n2. Вставка элемента на позицию 2:" << endl;
    stringTree.insertAt(2, String("Вставленный"));
    stringTree.printAll();
    
    // Поиск элемента
    cout << "\n3. Поиск элемента \"Третий\":" << endl;
    String searchStr("Третий");
    int index = stringTree.indexOf(searchStr);
    if (index != -1) {
        cout << "Элемент \"Третий\" найден на позиции " << index << endl;
        String* found = stringTree.find(searchStr);
        if (found) {
            cout << "Найденный элемент: " << *found << endl;
        }
    } else {
        cout << "Элемент \"Третий\" не найден" << endl;
    }
    
    // Использование шаблонных функций
    cout << "\n4. Использование шаблонных функций:" << endl;
    try {
        String maxStr = findMax(stringTree);
        String minStr = findMin(stringTree);
        cout << "Максимальный элемент (лексикографически): " << maxStr << endl;
        cout << "Минимальный элемент (лексикографически): " << minStr << endl;
        
        int longStrCount = countIf(stringTree, isLongString);
        cout << "Количество длинных строк (>5 символов): " << longStrCount << endl;
    } catch (const exception& e) {
        cout << "Ошибка: " << e.what() << endl;
    }
    
    // Работа с производными классами через базовый класс
    cout << "\n5. Работа с производными классами (через объекты):" << endl;
    Tree<String> derivedTree;
    
    // Создаем объекты разных типов и копируем их в дерево
    String baseStr("Базовая строка");
    OctalString octStr("777");
    TimedString timedStr("Строка с временем");
    OctalString octStr2(64);
    
    derivedTree.add(baseStr);
    derivedTree.add(String(octStr.getString()));  // Преобразуем в базовый класс
    derivedTree.add(String(timedStr.getString()));
    derivedTree.add(String(octStr2.getString()));
    
    cout << "Дерево с объектами String:" << endl;
    derivedTree.printAll();
}

// Демонстрация обработки исключений
void testExceptions() {
    cout << "\n========================================" << endl;
    cout << "ТЕСТИРОВАНИЕ ОБРАБОТКИ ИСКЛЮЧЕНИЙ" << endl;
    cout << "========================================" << endl;
    
    Tree<int> tree;
    tree.add(1);
    tree.add(2);
    tree.add(3);
    
    // Тест выхода за пределы диапазона
    cout << "\n1. Попытка доступа к несуществующему индексу:" << endl;
    try {
        int value = tree.getAt(10);
        cout << "Значение: " << value << endl;
    } catch (const OutOfRangeException& e) {
        cout << "Поймано исключение: " << e.what() << endl;
    }
    
    // Тест вставки с неверным индексом
    cout << "\n2. Попытка вставки с неверным индексом:" << endl;
    try {
        tree.insertAt(100, 99);
    } catch (const OutOfRangeException& e) {
        cout << "Поймано исключение: " << e.what() << endl;
    }
    
    // Тест удаления с неверным индексом
    cout << "\n3. Попытка удаления с неверным индексом:" << endl;
    try {
        tree.removeAt(-1);
    } catch (const OutOfRangeException& e) {
        cout << "Поймано исключение: " << e.what() << endl;
    }
    
    // Тест поиска максимума в пустом дереве
    cout << "\n4. Попытка поиска максимума в пустом дереве:" << endl;
    Tree<int> emptyTree;
    try {
        int maxVal = findMax(emptyTree);
        cout << "Максимум: " << maxVal << endl;
    } catch (const exception& e) {
        cout << "Поймано исключение: " << e.what() << endl;
    }
}




































int main() {
    cout << "========================================" << endl;
    cout << "ЛАБОРАТОРНАЯ РАБОТА №7" << endl;
    cout << "Универсальность. Применение шаблонов" << endl;
    cout << "функций и классов" << endl;
    cout << "========================================" << endl;
    cout << "\nЗадание:" << endl;
    cout << "Разработать шаблоны стандартных структур данных." << endl;
    cout << "В качестве структур данных взять разработанные" << endl;
    cout << "классы в лабораторной работе №5 (StringTree)." << endl;
    cout << "\nТестирование с типами: int, float, String (ЛР1)" << endl;
    cout << "========================================" << endl;
    
    try {
        // Тестирование с встроенными типами
        testIntTree();
        testFloatTree();
        
        // Тестирование с классом из ЛР1
        testStringTree();
        
        // Тестирование обработки исключений
        testExceptions();
        
    } catch (const StringException& e) {
        cout << "\nОшибка StringException: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "\nНеожиданная ошибка: " << e.what() << endl;
    }
    
    cout << "\n========================================" << endl;
    cout << "ТЕСТИРОВАНИЕ ЗАВЕРШЕНО" << endl;
    cout << "========================================" << endl;
    
    return 0;
}
