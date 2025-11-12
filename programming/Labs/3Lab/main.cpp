#include "String.h"
#include <iostream>
#include <fstream>
#include <vector>

using namespace std;

// Функция для демонстрации работы с текстовыми файлами
void demonstrateTextFiles() {
    cout << "\n=== РАБОТА С ТЕКСТОВЫМИ ФАЙЛАМИ ===" << endl;
    
    // Создаем несколько объектов для записи
    vector<String> stringsToWrite;
    stringsToWrite.push_back(String("Hello"));
    stringsToWrite.push_back(String("World"));
    stringsToWrite.push_back(String("C++ Programming"));
    stringsToWrite.push_back(String("Text File Demo"));
    
    // Запись в текстовый файл
    ofstream textOut("strings.txt");
    if (textOut.is_open()) {
        textOut << stringsToWrite.size() << endl; // Сохраняем количество объектов
        for (const auto& str : stringsToWrite) {
            str.writeToTextFile(textOut);
            textOut << endl; // Разделитель между объектами
        }
        textOut.close();
        cout << "Объекты успешно записаны в текстовый файл 'strings.txt'" << endl;
    } else {
        cout << "Ошибка открытия файла для записи!" << endl;
    }
    
    // Чтение из текстового файла
    vector<String> stringsFromText;
    ifstream textIn("strings.txt");
    if (textIn.is_open()) {
        int count;
        textIn >> count;
        textIn.get(); // Пропускаем символ новой строки
        
        for (int i = 0; i < count; i++) {
            String str;
            str.readFromTextFile(textIn);
            stringsFromText.push_back(str);
            textIn.get(); // Пропускаем символ новой строки
        }
        textIn.close();
        
        cout << "Объекты успешно загружены из текстового файла:" << endl;
        for (size_t i = 0; i < stringsFromText.size(); i++) {
            cout << "  " << i + 1 << ": " << stringsFromText[i] << endl;
        }
    } else {
        cout << "Ошибка открытия файла для чтения!" << endl;
    }
}

// Функция для демонстрации работы с бинарными файлами
void demonstrateBinaryFiles() {
    cout << "\n=== РАБОТА С БИНАРНЫМИ ФАЙЛАМИ ===" << endl;
    
    // Создаем несколько объектов для записи
    vector<String> stringsToWrite;
    stringsToWrite.push_back(String("Binary"));
    stringsToWrite.push_back(String("File"));
    stringsToWrite.push_back(String("Demo"));
    stringsToWrite.push_back(String("C++ Rocks!"));
    
    // Запись в бинарный файл
    ofstream binOut("strings.bin", ios::binary);
    if (binOut.is_open()) {
        int count = stringsToWrite.size();
        binOut.write(reinterpret_cast<const char*>(&count), sizeof(count));
        
        for (const auto& str : stringsToWrite) {
            str.writeBinary(binOut);
        }
        binOut.close();
        cout << "Объекты успешно записаны в бинарный файл 'strings.bin'" << endl;
    } else {
        cout << "Ошибка открытия бинарного файла для записи!" << endl;
    }
    
    // Чтение из бинарного файла
    vector<String> stringsFromBinary;
    ifstream binIn("strings.bin", ios::binary);
    if (binIn.is_open()) {
        int count;
        binIn.read(reinterpret_cast<char*>(&count), sizeof(count));
        
        for (int i = 0; i < count; i++) {
            String str;
            str.readBinary(binIn);
            stringsFromBinary.push_back(str);
        }
        binIn.close();
        
        cout << "Объекты успешно загружены из бинарного файла:" << endl;
        for (size_t i = 0; i < stringsFromBinary.size(); i++) {
            cout << "  " << i + 1 << ": " << stringsFromBinary[i] << endl;
        }
    } else {
        cout << "Ошибка открытия бинарного файла для чтения!" << endl;
    }
}

// Функция для интерактивного ввода данных
void interactiveDemo() {
    cout << "\n=== ИНТЕРАКТИВНЫЙ РЕЖИМ ===" << endl;
    
    vector<String> userStrings;
    int choice;
    
    do {
        cout << "\nМеню:" << endl;
        cout << "1. Добавить строку" << endl;
        cout << "2. Показать все строки" << endl;
        cout << "3. Сохранить в текстовый файл" << endl;
        cout << "4. Загрузить из текстового файла" << endl;
        cout << "5. Сохранить в бинарный файл" << endl;
        cout << "6. Загрузить из бинарного файла" << endl;
        cout << "0. Выход" << endl;
        cout << "Выберите действие: ";
        cin >> choice;
        cin.ignore(); // Очищаем буфер
        
        switch (choice) {
            case 1: {
                String newStr;
                cout << "Введите строку: ";
                cin >> newStr;
                userStrings.push_back(newStr);
                cout << "Строка добавлена!" << endl;
                break;
            }
            case 2: {
                if (userStrings.empty()) {
                    cout << "Список строк пуст!" << endl;
                } else {
                    cout << "Текущие строки:" << endl;
                    for (size_t i = 0; i < userStrings.size(); i++) {
                        cout << "  " << i + 1 << ": " << userStrings[i] << endl;
                    }
                }
                break;
            }
            case 3: {
                ofstream out("user_strings.txt");
                if (out.is_open()) {
                    out << userStrings.size() << endl;
                    for (const auto& str : userStrings) {
                        str.writeToTextFile(out);
                        out << endl;
                    }
                    out.close();
                    cout << "Строки сохранены в 'user_strings.txt'" << endl;
                } else {
                    cout << "Ошибка сохранения!" << endl;
                }
                break;
            }
            case 4: {
                ifstream in("user_strings.txt");
                if (in.is_open()) {
                    userStrings.clear();
                    int count;
                    in >> count;
                    in.get();
                    
                    for (int i = 0; i < count; i++) {
                        String str;
                        str.readFromTextFile(in);
                        userStrings.push_back(str);
                        in.get();
                    }
                    in.close();
                    cout << "Строки загружены из 'user_strings.txt'" << endl;
                } else {
                    cout << "Файл не найден!" << endl;
                }
                break;
            }
            case 5: {
                ofstream out("user_strings.bin", ios::binary);
                if (out.is_open()) {
                    int count = userStrings.size();
                    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
                    for (const auto& str : userStrings) {
                        str.writeBinary(out);
                    }
                    out.close();
                    cout << "Строки сохранены в 'user_strings.bin'" << endl;
                } else {
                    cout << "Ошибка сохранения!" << endl;
                }
                break;
            }
            case 6: {
                ifstream in("user_strings.bin", ios::binary);
                if (in.is_open()) {
                    userStrings.clear();
                    int count;
                    in.read(reinterpret_cast<char*>(&count), sizeof(count));
                    
                    for (int i = 0; i < count; i++) {
                        String str;
                        str.readBinary(in);
                        userStrings.push_back(str);
                    }
                    in.close();
                    cout << "Строки загружены из 'user_strings.bin'" << endl;
                } else {
                    cout << "Файл не найден!" << endl;
                }
                break;
            }
            case 0:
                cout << "Выход из интерактивного режима." << endl;
                break;
            default:
                cout << "Неверный выбор!" << endl;
        }
    } while (choice != 0);
}

int main() {
    cout << "=== ДЕМОНСТРАЦИЯ РАБОТЫ С ФАЙЛАМИ ДЛЯ КЛАССА STRING ===" << endl;
    
    // Демонстрация базового функционала
    cout << "\n=== БАЗОВЫЙ ФУНКЦИОНАЛ ===" << endl;
    String s1("Hello");
    String s2("World");
    String s3;
    
    cout << "s1: " << s1 << endl;
    cout << "s2: " << s2 << endl;
    
    s3 = s1 + " " + s2;
    cout << "s1 + \" \" + s2: " << s3 << endl;
    
    // Демонстрация работы с файлами
    demonstrateTextFiles();
    demonstrateBinaryFiles();
    
    // Интерактивный режим
    interactiveDemo();
    
    cout << "\n=== ЗАВЕРШЕНИЕ ПРОГРАММЫ ===" << endl;
    cout << "Общее количество объектов: " << String::getCount() << endl;
    
    return 0;
}