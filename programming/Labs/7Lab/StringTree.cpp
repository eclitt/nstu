#include "StringTree.h"
#include "Exceptions.h"
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

// Удаляет только узлы, но не объекты (для безопасной перестройки дерева)
void StringTree::deleteTreeNodesOnly(Node* node) {
    if (!node) return;
    deleteTreeNodesOnly(node->left);
    deleteTreeNodesOnly(node->right);
    delete node;  // Удаляем только узел, объект остается
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

    // Собираем все элементы в вектор
    std::vector<String*> arr;
    arr.reserve(nodeCount + 1);
    inorderCollect(root, arr);

    // Вставляем новый элемент в нужную позицию (логика «списка»)
    arr.insert(arr.begin() + index, obj);

    // Сохраняем старое дерево на случай исключения
    Node* oldRoot = root;
    
    // Удаляем старые узлы (объекты уже собраны в arr)
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
        // При исключении удаляем частично созданное дерево и объекты
        if (root) {
            deleteTree(root);  // Удаляем узлы и объекты частично созданного дерева
            root = nullptr;
        } else {
            // Если дерево не было создано, удаляем объект, который не удалось вставить
            delete obj;
        }
        nodeCount = 0;
        throw;
    }
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

    // Сохраняем указатель на удаляемый объект
    String* objToDelete = arr[index];
    
    // Удаляем элемент из вектора (но не удаляем сам объект пока)
    arr.erase(arr.begin() + index);

    // Сохраняем старое дерево
    Node* oldRoot = root;
    
    // Удаляем старые узлы (объекты уже собраны в arr)
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
        // При исключении удаляем все объекты из arr (кроме objToDelete, который будет удален отдельно)
        // и частично созданное дерево
        if (root) {
            // Удаляем только узлы частично созданного дерева, объекты уже в arr
            deleteTreeNodesOnly(root);
            root = nullptr;
        }
        // Удаляем все объекты из arr, так как дерево не было создано
        for (String* obj : arr) {
            if (obj != objToDelete) {
                delete obj;
            }
        }
        // Удаляем объект, который должен был быть удален
        delete objToDelete;
        nodeCount = 0;
        throw;
    }
    
    // Если все успешно, удаляем объект
    delete objToDelete;
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


