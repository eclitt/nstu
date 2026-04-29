#include "StringTree.h"
#include "Exceptions.h"
#include <iostream>
#include <cstring>
#include <fstream>
#include <new>

StringTree::StringTree() : root(nullptr), nodeCount(0) {}

StringTree::~StringTree() {
    clear();
}

void StringTree::deleteTree(Node* node) {
    if (!node) return;
    deleteTree(node->left);
    deleteTree(node->right);
    delete node->data;   // дерево владеет объектами
    delete node;
}

void StringTree::deleteNodesOnly(Node* node) {
    if (!node) return;
    deleteNodesOnly(node->left);
    deleteNodesOnly(node->right);
    // НЕ удаляем node->data, только сам узел
    delete node;
}

void StringTree::clear() {
    deleteTree(root);
    root = nullptr;
    nodeCount = 0;
}

void StringTree::inorderCollect(Node* node, std::vector<String*>& arr) const {
    if (!node) return;
    inorderCollect(node->left, arr);
    arr.push_back(node->data);
    inorderCollect(node->right, arr);
}

StringTree::Node* StringTree::buildBalanced(std::vector<String*>& arr, int l, int r) {
    if (l > r) return nullptr;
    int m = (l + r) / 2;
    
    Node* node = nullptr;
    try {
        node = new (std::nothrow) Node(arr[m]);
        if (!node) {
            throw MemoryException("Не удалось выделить память для узла дерева");
        }
        node->left  = buildBalanced(arr, l, m - 1);
        node->right = buildBalanced(arr, m + 1, r);
    } catch (const std::bad_alloc&) {
        delete node;
        throw MemoryException("Недостаточно памяти для создания узла дерева");
    }
    return node;
}

void StringTree::add(String* obj) {
    if (!obj) {
        throw InvalidArgumentException("Попытка добавления пустого указателя в дерево");
    }
    // Добавление "в конец" как вставка по номеру size()
    insertAt(nodeCount, obj);
}

void StringTree::insertAt(int index, String* obj) {
    if (!obj) {
        throw InvalidArgumentException("Попытка вставки пустого указателя в дерево");
    }
    
    if (index < 0 || index > nodeCount) {
        delete obj;
        throw OutOfRangeException("Индекс вставки " + std::to_string(index) + 
                                   " выходит за пределы допустимого диапазона [0, " + 
                                   std::to_string(nodeCount) + "]");
    }

    // Собираем все элементы в вектор (только указатели, не удаляем объекты)
    std::vector<String*> arr;
    arr.reserve(nodeCount + 1);
    inorderCollect(root, arr);

    // Вставляем новый элемент в нужную позицию (логика «списка»)
    arr.insert(arr.begin() + index, obj);

    // Удаляем только узлы дерева, но НЕ объекты данных
    // (объекты остаются в векторе и будут использованы при перестройке)
    deleteNodesOnly(root);
    root = nullptr;

    // Перестраиваем сбалансированное дерево из массива
    root = buildBalanced(arr, 0, static_cast<int>(arr.size()) - 1);
    nodeCount = static_cast<int>(arr.size());
}

bool StringTree::removeAt(int index) {
    if (index < 0 || index >= nodeCount) {
        throw OutOfRangeException("Индекс удаления " + std::to_string(index) + 
                                  " выходит за пределы допустимого диапазона [0, " + 
                                  std::to_string(nodeCount - 1) + "]");
    }

    // Собираем все элементы
    std::vector<String*> arr;
    arr.reserve(nodeCount);
    inorderCollect(root, arr);

    // Освобождаем удаляемый объект
    delete arr[index];
    arr.erase(arr.begin() + index);

    // Удаляем только узлы дерева, но НЕ объекты данных
    // (оставшиеся объекты остаются в векторе и будут использованы при перестройке)
    deleteNodesOnly(root);
    root = nullptr;

    // Перестраиваем дерево
    if (arr.empty()) {
        root = nullptr;
        nodeCount = 0;
    } else {
        root = buildBalanced(arr, 0, static_cast<int>(arr.size()) - 1);
        nodeCount = static_cast<int>(arr.size());
    }
    return true;
}

int StringTree::indexOfNode(Node* node, const char* value, int& currentIndex) const {
    if (!node) return -1;

    int idx = indexOfNode(node->left, value, currentIndex);
    if (idx != -1) return idx;

    // Текущий элемент
    if (node->data && std::strcmp(node->data->getString(), value) == 0) {
        return currentIndex;
    }
    ++currentIndex;

    return indexOfNode(node->right, value, currentIndex);
}

int StringTree::indexOf(const char* value) const {
    int currentIndex = 0;
    return indexOfNode(root, value, currentIndex);
}

String* StringTree::findInNode(Node* node, const char* value) const {
    if (!node) return nullptr;
    String* leftRes = findInNode(node->left, value);
    if (leftRes) return leftRes;
    if (node->data && std::strcmp(node->data->getString(), value) == 0) {
        return node->data;
    }
    return findInNode(node->right, value);
}

String* StringTree::find(const char* value) const {
    return findInNode(root, value);
}

void StringTree::printNode(Node* node) const {
    if (!node) return;
    printNode(node->left);
    if (node->data) {
        // Полиморфный вызов: будет вызываться переопределённый print()
        node->data->print();
    }
    printNode(node->right);
}

void StringTree::saveNodeToText(Node* node, std::ofstream& ofs) const {
    if (!node) return;
    saveNodeToText(node->left, ofs);
    if (node->data) {
        // ПОЛИМОРФИЗМ: вызывается writeToTextFile() соответствующего типа
        node->data->writeToTextFile(ofs);
        ofs << std::endl;
    }
    saveNodeToText(node->right, ofs);
}

void StringTree::saveNodeToBinary(Node* node, std::ofstream& ofs) const {
    if (!node) return;
    saveNodeToBinary(node->left, ofs);
    if (node->data) {
        // ПОЛИМОРФИЗМ: вызывается writeBinary() соответствующего типа
        node->data->writeBinary(ofs);
    }
    saveNodeToBinary(node->right, ofs);
}

void StringTree::printAll() const {
    if (!root) {
        std::cout << "Дерево пусто." << std::endl;
        return;
    }
    std::cout << "\nСодержимое дерева (симметричный обход):" << std::endl;
    printNode(root);
}

// Вспомогательная функция для демонстрации полиморфизма
void StringTree::demonstratePolymorphismNode(Node* node, int& index) const {
    if (!node) return;
    demonstratePolymorphismNode(node->left, index);
    if (node->data) {
        // ПОЛИМОРФИЗМ: через указатель на базовый класс String* вызываются
        // виртуальные функции производных классов
        std::cout << "[" << index << "] Указатель: String*, ";
        std::cout << "Реальный тип (через getTypeName()): " << node->data->getTypeName() << ", ";
        std::cout << "Тип (через getType()): " << static_cast<int>(node->data->getType()) << ", ";
        std::cout << "Содержимое: \"" << node->data->getString() << "\"" << std::endl;
        // Полиморфный вызов print() - каждый тип выводит по-своему
        std::cout << "  -> Вызов print() (полиморфный): ";
        node->data->print();
        index++;
    }
    demonstratePolymorphismNode(node->right, index);
}

void StringTree::demonstratePolymorphism() const {
    if (!root) {
        std::cout << "Дерево пусто. Нет объектов для демонстрации полиморфизма." << std::endl;
        return;
    }
    std::cout << "\n=== ДЕМОНСТРАЦИЯ ПОЛИМОРФИЗМА ===" << std::endl;
    std::cout << "Все объекты хранятся как String*, но каждый сохраняет свой реальный тип!" << std::endl;
    std::cout << "При вызове виртуальных функций вызывается версия производного класса.\n" << std::endl;
    
    int index = 0;
    demonstratePolymorphismNode(root, index);
    
    std::cout << "\n=== КОНЕЦ ДЕМОНСТРАЦИИ ПОЛИМОРФИЗМА ===" << std::endl;
}

void StringTree::saveToTextFile(const char* filename) const {
    std::ofstream ofs(filename, std::ios::out);
    if (!ofs.is_open()) {
        std::cerr << "Ошибка открытия файла для записи: " << filename << std::endl;
        return;
    }
    
    // Записываем количество элементов
    ofs << nodeCount << std::endl;
    
    // Используем полиморфизм: для каждого объекта вызывается его виртуальный метод writeToTextFile
    saveNodeToText(root, ofs);
    ofs.close();
    std::cout << "Дерево сохранено в текстовый файл: " << filename << std::endl;
}

void StringTree::loadFromTextFile(const char* filename) {
    std::ifstream ifs(filename, std::ios::in);
    if (!ifs.is_open()) {
        std::cerr << "Ошибка открытия файла для чтения: " << filename << std::endl;
        return;
    }
    
    // Очищаем текущее дерево
    clear();
    
    int count;
    ifs >> count;
    ifs.get(); // Пропускаем перевод строки
    
    // Используем полиморфизм: создаем объекты правильного типа через фабричный метод
    for (int i = 0; i < count; i++) {
        int typeInt;
        ifs >> typeInt;
        StringType type = static_cast<StringType>(typeInt);
        
        // ПОЛИМОРФИЗМ: создаем объект нужного типа через фабричный метод
        String* obj = String::createFromType(type);
        if (obj) {
            // ПОЛИМОРФИЗМ: вызывается readFromTextFile() соответствующего типа
            obj->readFromTextFile(ifs);
            add(obj);
        }
    }
    
    ifs.close();
    std::cout << "Дерево загружено из текстового файла: " << filename << std::endl;
}

void StringTree::saveToBinaryFile(const char* filename) const {
    std::ofstream ofs(filename, std::ios::out | std::ios::binary);
    if (!ofs.is_open()) {
        std::cerr << "Ошибка открытия файла для записи: " << filename << std::endl;
        return;
    }
    
    // Записываем количество элементов
    ofs.write(reinterpret_cast<const char*>(&nodeCount), sizeof(nodeCount));
    
    // Используем полиморфизм: для каждого объекта вызывается его виртуальный метод writeBinary
    saveNodeToBinary(root, ofs);
    ofs.close();
    std::cout << "Дерево сохранено в бинарный файл: " << filename << std::endl;
}

void StringTree::loadFromBinaryFile(const char* filename) {
    std::ifstream ifs(filename, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        std::cerr << "Ошибка открытия файла для чтения: " << filename << std::endl;
        return;
    }
    
    // Очищаем текущее дерево
    clear();
    
    int count;
    ifs.read(reinterpret_cast<char*>(&count), sizeof(count));
    
    // Используем полиморфизм: создаем объекты правильного типа
    for (int i = 0; i < count; i++) {
        // Сохраняем текущую позицию в файле
        std::streampos pos = ifs.tellg();
        
        // Читаем тип объекта (он записан первым в writeBinary)
        StringType type;
        ifs.read(reinterpret_cast<char*>(&type), sizeof(type));
        
        // ПОЛИМОРФИЗМ: создаем объект нужного типа через фабричный метод
        String* obj = String::createFromType(type);
        if (obj) {
            // Возвращаемся к позиции перед типом, так как readBinary сам прочитает тип
            ifs.seekg(pos);
            
            // ПОЛИМОРФИЗМ: вызывается readBinary() соответствующего типа
            // readBinary прочитает тип и данные объекта
            obj->readBinary(ifs);
            add(obj);
        }
    }
    
    ifs.close();
    std::cout << "Дерево загружено из бинарного файла: " << filename << std::endl;
}


