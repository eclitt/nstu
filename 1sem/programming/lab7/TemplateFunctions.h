#ifndef TEMPLATE_FUNCTIONS_H
#define TEMPLATE_FUNCTIONS_H

#include "Tree.h"
#include <iostream>
#include <algorithm>
#include <cstring>

// Шаблонная функция для поиска максимального элемента в дереве
template<typename T>
T findMax(const Tree<T>& tree) {
    if (tree.empty()) {
        throw std::runtime_error("Дерево пусто");
    }
    
    T maxVal = tree.getAt(0);
    for (int i = 1; i < tree.size(); ++i) {
        T current = tree.getAt(i);
        if (current > maxVal) {
            maxVal = current;
        }
    }
    return maxVal;
}

// Шаблонная функция для поиска минимального элемента в дереве
template<typename T>
T findMin(const Tree<T>& tree) {
    if (tree.empty()) {
        throw std::runtime_error("Дерево пусто");
    }
    
    T minVal = tree.getAt(0);
    for (int i = 1; i < tree.size(); ++i) {
        T current = tree.getAt(i);
        if (current < minVal) {
            minVal = current;
        }
    }
    return minVal;
}

// Шаблонная функция для подсчета элементов, удовлетворяющих условию
template<typename T>
int countIf(const Tree<T>& tree, bool (*predicate)(const T&)) {
    int count = 0;
    for (int i = 0; i < tree.size(); ++i) {
        if (predicate(tree.getAt(i))) {
            count++;
        }
    }
    return count;
}

// Шаблонная функция для вывода всех элементов дерева с форматированием
template<typename T>
void printTreeFormatted(const Tree<T>& tree, const char* label = "") {
    if (label && strlen(label) > 0) {
        std::cout << label << ": ";
    }
    if (tree.empty()) {
        std::cout << "Дерево пусто" << std::endl;
        return;
    }
    std::cout << "[";
    for (int i = 0; i < tree.size(); ++i) {
        std::cout << tree.getAt(i);
        if (i < tree.size() - 1) {
            std::cout << ", ";
        }
    }
    std::cout << "]" << std::endl;
}

// Шаблонная функция для копирования дерева
template<typename T>
Tree<T> copyTree(const Tree<T>& source) {
    Tree<T> result;
    for (int i = 0; i < source.size(); ++i) {
        result.add(source.getAt(i));
    }
    return result;
}

// Шаблонная функция для объединения двух деревьев
template<typename T>
Tree<T> mergeTrees(const Tree<T>& tree1, const Tree<T>& tree2) {
    Tree<T> result;
    
    // Добавляем элементы из первого дерева
    for (int i = 0; i < tree1.size(); ++i) {
        result.add(tree1.getAt(i));
    }
    
    // Добавляем элементы из второго дерева
    for (int i = 0; i < tree2.size(); ++i) {
        result.add(tree2.getAt(i));
    }
    
    return result;
}

#endif // TEMPLATE_FUNCTIONS_H

