#ifndef STRINGTREE_H
#define STRINGTREE_H

#include "String.h"
#include <vector>
// Бинарное дерево, хранящее указатели на объекты и демонстрирующее полиморфизм
// Узлы дерева содержат указатели на базовый класс String, поэтому в дереве
// могут находиться объекты String, OctalString, TimedString и любых потомков.

class StringTree {
private:
    struct Node {
        String* data;   // Полиморфный объект
        Node* left;
        Node* right;

        explicit Node(String* d) : data(d), left(nullptr), right(nullptr) {}
    };

    Node* root;
    int   nodeCount;

    // Вспомогательные методы
    void deleteTree(Node* node);
    void deleteNodesOnly(Node* node);  // Удаляет только узлы, не объекты данных
    void inorderCollect(Node* node, std::vector<String*>& arr) const;
    Node* buildBalanced(std::vector<String*>& arr, int l, int r);
    void printNode(Node* node) const;
    int indexOfNode(Node* node, const char* value, int& currentIndex) const;
    String* findInNode(Node* node, const char* value) const;
    void demonstratePolymorphismNode(Node* node, int& index) const;
    void saveNodeToText(Node* node, std::ofstream& ofs) const;
    void saveNodeToBinary(Node* node, std::ofstream& ofs) const;

public:
    StringTree();
    ~StringTree();

    int size() const { return nodeCount; }
    bool empty() const { return nodeCount == 0; }

    // Полное удаление дерева
    void clear();

    // Добавление в конец (эквивалент вставки по номеру = size())
    void add(String* obj);

    // Вставка по номеру (0 <= index <= size()).
    // Номер понимается как позиция в симметричном (in-order) обходе дерева.
    void insertAt(int index, String* obj);

    // Удаление по номеру (0 <= index < size()).
    // Возвращает true, если элемент был удалён.
    bool removeAt(int index);

    // Поиск по значению строки (точное совпадение getString()).
    // Возвращает индекс в обходе или -1, если не найдено.
    int indexOf(const char* value) const;

    // Поиск по значению строки, возвращает указатель на объект
    // (объект остаётся во владении дерева).
    String* find(const char* value) const;

    // Просмотр всей структуры (симметричный обход)
    void printAll() const;
    
    // Демонстрация полиморфизма: вывод информации о типах объектов
    void demonstratePolymorphism() const;
    
    // Методы для работы с файлами (используют полиморфизм)
    void saveToTextFile(const char* filename) const;
    void loadFromTextFile(const char* filename);
    void saveToBinaryFile(const char* filename) const;
    void loadFromBinaryFile(const char* filename);
};

#endif


