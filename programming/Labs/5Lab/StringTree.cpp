#include "StringTree.h"
#include <iostream>
#include <cstring>

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
    Node* node = new Node(arr[m]);
    node->left  = buildBalanced(arr, l, m - 1);
    node->right = buildBalanced(arr, m + 1, r);
    return node;
}

void StringTree::add(String* obj) {
    // Добавление "в конец" как вставка по номеру size()
    insertAt(nodeCount, obj);
}

void StringTree::insertAt(int index, String* obj) {
    if (index < 0 || index > nodeCount) {
        std::cout << "Некорректный индекс вставки: " << index << std::endl;
        delete obj;
        return;
    }

    // Собираем все элементы в вектор
    std::vector<String*> arr;
    arr.reserve(nodeCount + 1);
    inorderCollect(root, arr);

    // Вставляем новый элемент в нужную позицию (логика «списка»)
    arr.insert(arr.begin() + index, obj);

    // Перестраиваем сбалансированное дерево из массива
    deleteTree(root);
    root = buildBalanced(arr, 0, static_cast<int>(arr.size()) - 1);
    nodeCount = static_cast<int>(arr.size());
}

bool StringTree::removeAt(int index) {
    if (index < 0 || index >= nodeCount) {
        std::cout << "Некорректный индекс удаления: " << index << std::endl;
        return false;
    }

    // Собираем все элементы
    std::vector<String*> arr;
    arr.reserve(nodeCount);
    inorderCollect(root, arr);

    // Освобождаем удаляемый объект
    delete arr[index];
    arr.erase(arr.begin() + index);

    // Перестраиваем дерево
    deleteTree(root);
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

void StringTree::printAll() const {
    if (!root) {
        std::cout << "Дерево пусто." << std::endl;
        return;
    }
    std::cout << "\nСодержимое дерева (симметричный обход):" << std::endl;
    printNode(root);
}


