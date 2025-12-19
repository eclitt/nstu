#include "String.h"
#include "OctalString.h"
#include "TimedString.h"
#include "StringTree.h"
#include "Exceptions.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <memory>
#include <thread>
#include <chrono>
#include <climits>

using namespace std;

// Функция для паузы
void pause(int seconds) {
    this_thread::sleep_for(chrono::seconds(seconds));
}

// Демонстрация создания и копирования объектов
void demonstrateCreationAndCopying() {
    cout << "\n=== СОЗДАНИЕ И КОПИРОВАНИЕ ОБЪЕКТОВ ===" << endl;
    
    // 1. Создание объектов базового класса
    cout << "\n1. Базовый класс String:" << endl;
    String s1("Первая строка");
    String s2 = s1;  // Конструктор копирования
    String s3;
    s3 = s1;         // Оператор присваивания
    
    s1.print();
    s2.print();
    s3.print();
    
    // 2. Создание объектов OctalString
    cout << "\n2. Производный класс OctalString:" << endl;
    OctalString os1("12345");  // Восьмеричное число
    OctalString os2(42);       // Десятичное число 42 -> "52" в восьмеричной
    OctalString os3(os1);      // Конструктор копирования
    OctalString os4;
    os4 = os2;                 // Оператор присваивания
    
    os1.print();
    os2.print();
    os3.print();
    os4.print();
    
    // 3. Создание объектов TimedString
    cout << "\n3. Производный класс TimedString:" << endl;
    TimedString ts1("Строка с временем");
    cout << "Пауза 2 секунды..." << endl;
    pause(2);
    
    TimedString ts2(ts1);      // Конструктор копирования
    TimedString ts3;
    ts3 = ts1;                 // Оператор присваивания
    
    ts1.print();
    ts2.print();
    ts3.print();
    
    // 4. Преобразование типов
    cout << "\n4. Преобразование типов:" << endl;
    String base("7654321");
    OctalString os5(base);     // Конструктор преобразования
    TimedString ts5(base);     // Конструктор преобразования
    
    os5.print();
    ts5.print();
    
    // 5. Присваивание объектов разных типов
    cout << "\n5. Присваивание объектов разных типов:" << endl;
    OctalString os6;
    os6 = base;                // String -> OctalString
    os6.print();
    
    TimedString ts6;
    ts6 = base;                // String -> TimedString
    ts6.print();
}

// Демонстрация полиморфизма
void demonstratePolymorphism() {
    cout << "\n=== ДЕМОНСТРАЦИЯ ПОЛИМОРФИЗМА ===" << endl;
    
    // Создаем массив указателей на базовый класс
    vector<unique_ptr<String>> objects;
    
    // Добавляем объекты разных типов
    objects.push_back(unique_ptr<String>(new String("Простая строка")));
    objects.push_back(unique_ptr<String>(new OctalString("777")));
    objects.push_back(unique_ptr<String>(new OctalString(255)));
    objects.push_back(unique_ptr<String>(new TimedString("Строка с временем создания")));
    
    // Демонстрация виртуальных методов
    cout << "\nВывод всех объектов через виртуальные методы:" << endl;
    for (const auto& obj : objects) {
        obj->print();
    }
    
    // Демонстрация операций через базовый класс
    cout << "\nОперации через базовый класс:" << endl;
    String* s1 = objects[0].get();
    String* s2 = objects[1].get();
    String result = *s1 + " " + *s2;
    cout << "Результат сложения: " << result << endl;
}

// Демонстрация работы с файлами (исправленная)
void demonstrateFileOperations() {
    cout << "\n=== РАБОТА С ФАЙЛАМИ ===" << endl;
    
    // 1. Создаем объекты разных типов
    vector<unique_ptr<String>> objectsToSave;
    objectsToSave.push_back(unique_ptr<String>(new String("Текстовая строка")));
    objectsToSave.push_back(unique_ptr<String>(new OctalString("1234567")));
    objectsToSave.push_back(unique_ptr<String>(new OctalString(100)));
    objectsToSave.push_back(unique_ptr<String>(new TimedString("Строка с временем")));
    
    // 2. Сохраняем в текстовый файл
    cout << "\n1. Сохранение в текстовый файл..." << endl;
    ofstream textOut("objects.txt");
    if (textOut.is_open()) {
        textOut << objectsToSave.size() << endl;
        for (const auto& obj : objectsToSave) {
            obj->writeToTextFile(textOut);
            textOut << endl;
        }
        textOut.close();
        cout << "Объекты сохранены в 'objects.txt'" << endl;
    } else {
        cout << "Ошибка открытия файла для записи!" << endl;
    }
    
    // 3. Загружаем из текстового файла
    cout << "\n2. Загрузка из текстового файла..." << endl;
    vector<unique_ptr<String>> loadedObjects;
    ifstream textIn("objects.txt");
    if (textIn.is_open()) {
        int count;
        textIn >> count;
        textIn.get(); // Пропускаем перевод строки
        
        for (int i = 0; i < count; i++) {
            // Читаем тип объекта
            int typeInt;
            textIn >> typeInt;
            StringType type = static_cast<StringType>(typeInt);
            
            // Создаем объект нужного типа
            unique_ptr<String> obj(String::createFromType(type));
            if (obj) {
                try {
                    obj->readFromTextFile(textIn);
                    loadedObjects.push_back(move(obj));
                } catch (const FileException& e) {
                    cout << "Ошибка файла при чтении объекта: " << e.what() << endl;
                } catch (const MemoryException& e) {
                    cout << "Ошибка памяти при чтении объекта: " << e.what() << endl;
                } catch (const exception& e) {
                    cout << "Ошибка при чтении объекта: " << e.what() << endl;
                }
            }
            textIn.get(); // Пропускаем перевод строки
        }
        textIn.close();
        
        cout << "\nЗагруженные объекты:" << endl;
        for (const auto& obj : loadedObjects) {
            obj->print();
        }
    } else {
        cout << "Ошибка открытия файла для чтения!" << endl;
    }
    
    // 4. Бинарные файлы
    cout << "\n3. Работа с бинарными файлами..." << endl;
    
    // Сохраняем в бинарный файл
    ofstream binOut("objects.bin", ios::binary);
    if (binOut.is_open()) {
        int count = objectsToSave.size();
        binOut.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& obj : objectsToSave) {
            obj->writeBinary(binOut);
        }
        binOut.close();
        cout << "Объекты сохранены в 'objects.bin'" << endl;
    } else {
        cout << "Ошибка открытия бинарного файла для записи!" << endl;
    }
    
    // Загружаем из бинарного файла (исправленная версия)
    vector<unique_ptr<String>> binLoadedObjects;
    ifstream binIn("objects.bin", ios::binary);
    if (binIn.is_open()) {
        int count;
        binIn.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        for (int i = 0; i < count; i++) {
            // Читаем тип объекта
            StringType type;
            binIn.read(reinterpret_cast<char*>(&type), sizeof(type));
            
            // Возвращаемся назад, чтобы readBinary мог прочитать тип снова
            binIn.seekg(-static_cast<streamoff>(sizeof(type)), ios::cur);
            
            // Создаем объект нужного типа
            unique_ptr<String> obj(String::createFromType(type));
            if (obj) {
                obj->readBinary(binIn);
                binLoadedObjects.push_back(move(obj));
            }
        }
        binIn.close();
        
        cout << "\nОбъекты загружены из бинарного файла:" << endl;
        for (const auto& obj : binLoadedObjects) {
            obj->print();
        }
    } else {
        cout << "Ошибка открытия бинарного файла для чтения!" << endl;
    }
}

// Демонстрация специфических возможностей OctalString
void demonstrateOctalStringFeatures() {
    cout << "\n=== СПЕЦИФИЧЕСКИЕ ВОЗМОЖНОСТИ OCTALSTRING ===" << endl;
    
    try {
        // Создание объектов
        OctalString oct1("12345");
        OctalString oct2(42);
        
        cout << "\nСозданные объекты:" << endl;
        oct1.print();
        oct2.print();
        
        // Арифметические операции
        cout << "\nАрифметические операции:" << endl;
        OctalString sum = oct1 + oct2;
        cout << "oct1 + oct2 = ";
        sum.print();
        
        OctalString diff = oct2 - 10;
        cout << "oct2 - 10 = ";
        diff.print();
        
        // Проверка восьмеричных чисел
        cout << "\nПроверка восьмеричных чисел:" << endl;
        OctalString valid("76543210");
        valid.print();
        
        cout << "\nПопытка создать невосьмеричную строку:" << endl;
        try {
            OctalString invalid("123ABC");
            invalid.print();
        } catch (const InvalidArgumentException& e) {
            cout << "Ошибка: " << e.what() << endl;
        } catch (const exception& e) {
            cout << "Ошибка: " << e.what() << endl;
        }
        
    } catch (const InvalidArgumentException& e) {
        cout << "Исключение: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "Исключение: " << e.what() << endl;
    }
}

// Демонстрация специфических возможностей TimedString
void demonstrateTimedStringFeatures() {
    cout << "\n=== СПЕЦИФИЧЕСКИЕ ВОЗМОЖНОСТИ TIMEDSTRING ===" << endl;
    
    TimedString ts1("Первая строка");
    cout << "Пауза 3 секунды..." << endl;
    pause(3);
    
    TimedString ts2("Вторая строка");
    
    cout << "\nОбъекты с разным временем создания:" << endl;
    ts1.print();
    ts2.print();
    
    cout << "\nВремя создания ts1: " << ts1.getCreationTimeString() << endl;
    cout << "Время создания ts2: " << ts2.getCreationTimeString() << endl;
}

// Демонстрация бинарного дерева из полиморфных объектов
void demonstrateStringTree() {
    cout << "\n=== ДИНАМИЧЕСКИЙ СПИСОК ОБЪЕКТОВ (БИНАРНОЕ ДЕРЕВО) ===" << endl;

    StringTree tree;

    // Добавление разнообразных объектов (через ссылки на базовый класс)
    tree.add(new String("Базовая строка"));
    tree.add(new OctalString("777"));
    tree.add(new TimedString("Строка с временем"));
    tree.add(new OctalString(64));
    tree.add(new TimedString("Ещё одна timed-строка"));

    cout << "\n1. Исходное дерево после добавления элементов:" << endl;
    tree.printAll();

    // Вставка по номеру
    cout << "\n2. Вставка по номеру:" << endl;
    cout << "Вставляем TimedString на позицию 2..." << endl;
    tree.insertAt(2, new TimedString("Вставленная по номеру 2"));
    tree.printAll();

    // Поиск по значению
    cout << "\n3. Поиск по значению строки:" << endl;
    const char* valueToFind = "777";
    int index = tree.indexOf(valueToFind);
    if (index != -1) {
        cout << "Строка \"" << valueToFind << "\" найдена на позиции " << index << endl;
        String* found = tree.find(valueToFind);
        if (found) {
            cout << "Полиморфный вывод найденного объекта:" << endl;
            found->print();
        }
    } else {
        cout << "Строка \"" << valueToFind << "\" не найдена в дереве." << endl;
    }

    // Удаление по номеру
    cout << "\n4. Удаление по номеру:" << endl;
    cout << "Удаляем элемент с номером 1..." << endl;
    tree.removeAt(1);
    tree.printAll();

    cout << "\n5. Финальное количество элементов в дереве: " << tree.size() << endl;
}

// Демонстрация обработки исключений
void demonstrateExceptionHandling() {
    cout << "\n=== ДЕМОНСТРАЦИЯ ОБРАБОТКИ ИСКЛЮЧЕНИЙ ===" << endl;
    
    // 1. Деление на ноль
    cout << "\n1. Демонстрация деления на ноль:" << endl;
    try {
        OctalString oct1(100);
        OctalString result = oct1 / 0;
        result.print();
    } catch (const DivisionByZeroException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "   Неожиданное исключение: " << e.what() << endl;
    }
    
    // 2. Остаток от деления на ноль
    cout << "\n2. Демонстрация остатка от деления на ноль:" << endl;
    try {
        OctalString oct2(100);
        OctalString result = oct2 % 0;
        result.print();
    } catch (const DivisionByZeroException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    }
    
    // 3. Некорректная восьмеричная строка
    cout << "\n3. Демонстрация некорректной восьмеричной строки:" << endl;
    try {
        OctalString invalid("123ABC");
        invalid.print();
    } catch (const InvalidArgumentException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    }
    
    // 4. Выход за пределы диапазона в StringTree
    cout << "\n4. Демонстрация выхода за пределы диапазона в StringTree:" << endl;
    try {
        StringTree tree;
        tree.add(new String("Первый"));
        tree.add(new String("Второй"));
        tree.insertAt(10, new String("Некорректный индекс")); // Индекс больше размера
    } catch (const OutOfRangeException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    }
    
    // 5. Отрицательный индекс в StringTree
    cout << "\n5. Демонстрация отрицательного индекса в StringTree:" << endl;
    try {
        StringTree tree;
        tree.add(new String("Первый"));
        tree.removeAt(-1); // Отрицательный индекс
    } catch (const OutOfRangeException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    }
    
    // 6. Переполнение при сложении
    cout << "\n6. Демонстрация переполнения при сложении:" << endl;
    try {
        OctalString oct3(LLONG_MAX - 10);
        OctalString oct4(20);
        OctalString result = oct3 + oct4; // Переполнение
        result.print();
    } catch (const OutOfRangeException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    }
    
    // 7. Ошибка работы с файлом (несуществующий файл)
    cout << "\n7. Демонстрация ошибки чтения из несуществующего файла:" << endl;
    try {
        ifstream ifs("несуществующий_файл.txt");
        if (!ifs.is_open()) {
            throw FileException("Файл 'несуществующий_файл.txt' не найден");
        }
        String s;
        s.readFromTextFile(ifs);
        ifs.close();
    } catch (const FileException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    }
    
    // 8. Попытка добавления пустого указателя
    cout << "\n8. Демонстрация добавления пустого указателя в дерево:" << endl;
    try {
        StringTree tree;
        tree.add(nullptr);
    } catch (const InvalidArgumentException& e) {
        cout << "   Поймано исключение: " << e.what() << endl;
    }
    
    // 9. Множественные исключения в одном блоке try-catch
    cout << "\n9. Демонстрация обработки разных типов исключений:" << endl;
    try {
        OctalString oct5("777");
        OctalString result1 = oct5 / 0;
        OctalString result2 = oct5 + LLONG_MAX;
    } catch (const DivisionByZeroException& e) {
        cout << "   Поймано исключение деления на ноль: " << e.what() << endl;
    } catch (const OutOfRangeException& e) {
        cout << "   Поймано исключение выхода за пределы: " << e.what() << endl;
    } catch (const StringException& e) {
        cout << "   Поймано общее исключение StringException: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "   Поймано стандартное исключение: " << e.what() << endl;
    }
    
    cout << "\n=== ДЕМОНСТРАЦИЯ ИСКЛЮЧЕНИЙ ЗАВЕРШЕНА ===" << endl;
}

int main() {
    cout << "=== ЛАБОРАТОРНАЯ РАБОТА 6: ОБРАБОТКА ИСКЛЮЧИТЕЛЬНЫХ СИТУАЦИЙ ===" << endl;
    cout << "Вариант 10: Иерархия классов для работы со строками" << endl;
    
    cout << "\nНачальное количество объектов String: " << String::getCount() << endl;
    
    // Демонстрация обработки исключений
    demonstrateExceptionHandling();
    
    // Демонстрация различных аспектов (с обработкой исключений)
    try {
        demonstrateCreationAndCopying();
        demonstratePolymorphism();
        demonstrateOctalStringFeatures();
        demonstrateTimedStringFeatures();
        demonstrateStringTree();
        demonstrateFileOperations();
    } catch (const StringException& e) {
        cout << "\nОшибка в демонстрационных функциях: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "\nНеожиданная ошибка: " << e.what() << endl;
    }
    
    // Массив разнотипных объектов
    cout << "\n=== МАССИВ РАЗНОТИПНЫХ ОБЪЕКТОВ ===" << endl;
    String* array[5];
    
    try {
        array[0] = new String("String object");
        array[1] = new OctalString("7654321");
        array[2] = new OctalString(255);
        array[3] = new TimedString("Timed object");
        array[4] = new TimedString("Another timed");
        
        for (int i = 0; i < 5; i++) {
            array[i]->print();
        }
    } catch (const MemoryException& e) {
        cout << "Ошибка выделения памяти: " << e.what() << endl;
    } catch (const exception& e) {
        cout << "Ошибка при создании объектов: " << e.what() << endl;
    }
    
    // Очистка памяти
    for (int i = 0; i < 5; i++) {
        if (array[i]) {
            delete array[i];
        }
    }
    
    cout << "\nФинальное количество объектов String: " << String::getCount() << endl;
    cout << "\n=== ПРОГРАММА УСПЕШНО ЗАВЕРШЕНА ===" << endl;
    
    return 0;
}