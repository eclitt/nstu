#ifndef TREE_H
#define TREE_H

#include "Exceptions.h"
#include <vector>
#include <iostream>
#include <cstring>

// Шаблонный класс бинарного дерева для хранения объектов любого типа
template<typename T>
class Tree {
private:
    struct Node {
        T data;        // Данные узла
        Node* left;
        Node* right;

        explicit Node(const T& d) : data(d), left(nullptr), right(nullptr) {}
    };

    Node* root;
    int nodeCount;

    // Вспомогательные методы
    void deleteTree(Node* node);
    void deleteTreeNodesOnly(Node* node);
    void inorderCollect(Node* node, std::vector<T>& arr) const;
    Node* buildBalanced(std::vector<T>& arr, int l, int r);
    void printNode(Node* node) const;
    int indexOfNode(Node* node, const T& value, int& currentIndex) const;
    T* findInNode(Node* node, const T& value) const;

public:
    Tree();
    ~Tree();

    int size() const { return nodeCount; }
    bool empty() const { return nodeCount == 0; }

    // Полное удаление дерева
    void clear();

    // Добавление в конец
    void add(const T& obj);

    // Вставка по номеру (0 <= index <= size())
    void insertAt(int index, const T& obj);

    // Удаление по номеру (0 <= index < size())
    bool removeAt(int index);

    // Поиск по значению, возвращает индекс или -1
    int indexOf(const T& value) const;

    // Поиск по значению, возвращает указатель на объект
    T* find(const T& value) const;

    // Просмотр всей структуры (симметричный обход)
    void printAll() const;

    // Получение элемента по индексу
    T getAt(int index) const;
};

// Реализация шаблонного класса

template<typename T>
Tree<T>::Tree() : root(nullptr), nodeCount(0) {}

template<typename T>
Tree<T>::~Tree() {
    clear();
}

template<typename T>
void Tree<T>::deleteTree(Node* node) {
    if (!node) return;
    deleteTree(node->left);
    deleteTree(node->right);
    delete node;
}

template<typename T>
void Tree<T>::deleteTreeNodesOnly(Node* node) {
    if (!node) return;
    deleteTreeNodesOnly(node->left);
    deleteTreeNodesOnly(node->right);
    delete node;
}

template<typename T>
void Tree<T>::clear() {
    deleteTree(root);
    root = nullptr;
    nodeCount = 0;
}

template<typename T>
void Tree<T>::inorderCollect(Node* node, std::vector<T>& arr) const {
    if (!node) return;
    inorderCollect(node->left, arr);
    arr.push_back(node->data);
    inorderCollect(node->right, arr);
}

template<typename T>
typename Tree<T>::Node* Tree<T>::buildBalanced(std::vector<T>& arr, int l, int r) {
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

template<typename T>
void Tree<T>::add(const T& obj) {
    insertAt(nodeCount, obj);
}

template<typename T>
void Tree<T>::insertAt(int index, const T& obj) {
    if (index < 0 || index > nodeCount) {
        throw OutOfRangeException("Индекс вставки " + std::to_string(index) + 
                                   " выходит за пределы допустимого диапазона [0, " + 
                                   std::to_string(nodeCount) + "]");
    }

    // Собираем все элементы в вектор
    std::vector<T> arr;
    arr.reserve(nodeCount + 1);
    inorderCollect(root, arr);

    // Вставляем новый элемент в нужную позицию
    arr.insert(arr.begin() + index, obj);

    // Сохраняем старое дерево
    Node* oldRoot = root;
    
    // Удаляем старые узлы
    if (oldRoot) {
        deleteTreeNodesOnly(oldRoot);
    }
    root = nullptr;
    nodeCount = 0;
    
    try {
        // Перестраиваем сбалансированное дерево из массива
        root = buildBalanced(arr, 0, static_cast<int>(arr.size()) - 1);
        nodeCount = static_cast<int>(arr.size());
    } catch (...) {
        // При исключении удаляем частично созданное дерево
        if (root) {
            deleteTree(root);
            root = nullptr;
        }
        nodeCount = 0;
        throw;
    }
}

template<typename T>
bool Tree<T>::removeAt(int index) {
    if (index < 0 || index >= nodeCount) {
        throw OutOfRangeException("Индекс удаления " + std::to_string(index) + 
                                  " выходит за пределы допустимого диапазона [0, " + 
                                  std::to_string(nodeCount - 1) + "]");
    }

    // Собираем все элементы
    std::vector<T> arr;
    arr.reserve(nodeCount);
    inorderCollect(root, arr);

    // Удаляем элемент из вектора
    arr.erase(arr.begin() + index);

    // Сохраняем старое дерево
    Node* oldRoot = root;
    
    // Удаляем старые узлы
    if (oldRoot) {
        deleteTreeNodesOnly(oldRoot);
    }
    root = nullptr;
    nodeCount = 0;
    
    try {
        // Перестраиваем дерево без удаленного элемента
        if (arr.empty()) {
            root = nullptr;
            nodeCount = 0;
        } else {
            root = buildBalanced(arr, 0, static_cast<int>(arr.size()) - 1);
            nodeCount = static_cast<int>(arr.size());
        }
    } catch (...) {
        // При исключении удаляем частично созданное дерево
        if (root) {
            deleteTreeNodesOnly(root);
            root = nullptr;
        }
        nodeCount = 0;
        throw;
    }
    
    return true;
}

template<typename T>
int Tree<T>::indexOfNode(Node* node, const T& value, int& currentIndex) const {
    if (!node) return -1;

    int idx = indexOfNode(node->left, value, currentIndex);
    if (idx != -1) return idx;

    // Текущий элемент
    if (node->data == value) {
        return currentIndex;
    }
    ++currentIndex;

    return indexOfNode(node->right, value, currentIndex);
}

template<typename T>
int Tree<T>::indexOf(const T& value) const {
    int currentIndex = 0;
    return indexOfNode(root, value, currentIndex);
}

template<typename T>
T* Tree<T>::findInNode(Node* node, const T& value) const {
    if (!node) return nullptr;
    T* leftRes = findInNode(node->left, value);
    if (leftRes) return leftRes;
    if (node->data == value) {
        return &(node->data);
    }
    return findInNode(node->right, value);
}

template<typename T>
T* Tree<T>::find(const T& value) const {
    return findInNode(root, value);
}

template<typename T>
void Tree<T>::printNode(Node* node) const {
    if (!node) return;
    printNode(node->left);
    std::cout << node->data << " ";
    printNode(node->right);
}

template<typename T>
void Tree<T>::printAll() const {
    if (!root) {
        std::cout << "Дерево пусто." << std::endl;
        return;
    }
    std::cout << "\nСодержимое дерева (симметричный обход): ";
    printNode(root);
    std::cout << std::endl;
}

template<typename T>
T Tree<T>::getAt(int index) const {
    if (index < 0 || index >= nodeCount) {
        throw OutOfRangeException("Индекс " + std::to_string(index) + 
                                  " выходит за пределы допустимого диапазона [0, " + 
                                  std::to_string(nodeCount - 1) + "]");
    }
    
    std::vector<T> arr;
    arr.reserve(nodeCount);
    inorderCollect(root, arr);
    return arr[index];
}

#endif // TREE_H

