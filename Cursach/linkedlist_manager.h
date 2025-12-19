#ifndef LINKEDLIST_MANAGER_H
#define LINKEDLIST_MANAGER_H

#include "date_student.h"
#include <fstream>
#include <functional>
#include <algorithm>
#include <vector>
#include <random>
#include <stdexcept>
#include <chrono>
#include <iomanip>

// ===================== Шаблонный класс Node =====================
template <typename T>
class Node {
public:
    T data;
    Node<T>* next;
    
    Node(const T& data) : data(data), next(nullptr) {}
};

// ===================== Абстрактный класс Container =====================
template <typename T>
class Container {
public:
    virtual ~Container() {}
    virtual void add(const T& item) = 0;
    virtual void insert(int index, const T& item) = 0;
    virtual void remove(int index) = 0;
    virtual void remove(const T& item) = 0;
    virtual T get(int index) const = 0;
    virtual void set(int index, const T& item) = 0;
    virtual int getSize() const = 0;
    virtual bool isEmpty() const = 0;
    virtual void clear() = 0;
    virtual int find(const T& item) const = 0;
    virtual bool contains(const T& item) const = 0;
    virtual void sort(bool ascending = true) = 0;
    virtual void sort(std::function<bool(const T&, const T&)> comparator) = 0;
    virtual bool saveToFile(const std::string& filename) const = 0;
    virtual bool loadFromFile(const std::string& filename) = 0;
    virtual void print() const = 0;
};

// ===================== Класс LinkedList =====================
template <typename T>
class LinkedList : public Container<T> {
private:
    Node<T>* head;
    int size;
    
    Node<T>* getNode(int index) const {
        if (index < 0 || index >= size) return nullptr;
        Node<T>* current = head;
        for (int i = 0; i < index; i++) current = current->next;
        return current;
    }
    
    void swapNodes(int i, int j) {
        if (i < 0 || i >= size || j < 0 || j >= size || i == j) return;
        T temp = get(i);
        set(i, get(j));
        set(j, temp);
    }
    
    void quickSort(int left, int right, std::function<bool(const T&, const T&)> comparator) {
        if (left >= right) return;
        T pivot = get((left + right) / 2);
        int i = left, j = right;
        
        while (i <= j) {
            while (comparator(get(i), pivot)) i++;
            while (comparator(pivot, get(j))) j--;
            if (i <= j) {
                swapNodes(i, j);
                i++; j--;
            }
        }
        
        if (left < j) quickSort(left, j, comparator);
        if (i < right) quickSort(i, right, comparator);
    }
    
public:
    LinkedList() : head(nullptr), size(0) {}
    
    LinkedList(const LinkedList<T>& other) : head(nullptr), size(0) {
        Node<T>* current = other.head;
        while (current) {
            add(current->data);
            current = current->next;
        }
    }
    
    ~LinkedList() { clear(); }
    
    LinkedList<T>& operator=(const LinkedList<T>& other) {
        if (this != &other) {
            clear();
            Node<T>* current = other.head;
            while (current) {
                add(current->data);
                current = current->next;
            }
        }
        return *this;
    }
    
    virtual void add(const T& item) override {
        Node<T>* newNode = new Node<T>(item);
        if (!head) {
            head = newNode;
        } else {
            Node<T>* current = head;
            while (current->next) current = current->next;
            current->next = newNode;
        }
        size++;
    }
    
    virtual void insert(int index, const T& item) override {
        if (index < 0 || index > size) throw std::out_of_range("Index out of range");
        
        if (index == 0) {
            Node<T>* newNode = new Node<T>(item);
            newNode->next = head;
            head = newNode;
        } else if (index == size) {
            add(item);
            return;
        } else {
            Node<T>* prev = getNode(index - 1);
            Node<T>* newNode = new Node<T>(item);
            newNode->next = prev->next;
            prev->next = newNode;
        }
        size++;
    }
    
    virtual void remove(int index) override {
        if (index < 0 || index >= size) throw std::out_of_range("Index out of range");
        
        if (index == 0) {
            Node<T>* temp = head;
            head = head->next;
            delete temp;
        } else {
            Node<T>* prev = getNode(index - 1);
            Node<T>* toDelete = prev->next;
            prev->next = toDelete->next;
            delete toDelete;
        }
        size--;
    }
    
    virtual void remove(const T& item) override {
        int index = find(item);
        if (index != -1) remove(index);
    }
    
    virtual T get(int index) const override {
        Node<T>* node = getNode(index);
        if (!node) throw std::out_of_range("Index out of range");
        return node->data;
    }
    
    virtual void set(int index, const T& item) override {
        Node<T>* node = getNode(index);
        if (!node) throw std::out_of_range("Index out of range");
        node->data = item;
    }
    
    virtual int getSize() const override { return size; }
    virtual bool isEmpty() const override { return size == 0; }
    
    virtual void clear() override {
        while (head) {
            Node<T>* temp = head;
            head = head->next;
            delete temp;
        }
        size = 0;
    }
    
    virtual int find(const T& item) const override {
        Node<T>* current = head;
        int index = 0;
        while (current) {
            if (current->data == item) return index;
            current = current->next;
            index++;
        }
        return -1;
    }
    
    virtual bool contains(const T& item) const override {
        return find(item) != -1;
    }
    
    virtual void sort(bool ascending = true) override {
        sort([ascending](const T& a, const T& b) {
            return ascending ? a < b : b < a;
        });
    }
    
    virtual void sort(std::function<bool(const T&, const T&)> comparator) override {
        if (size <= 1) return;
        quickSort(0, size - 1, comparator);
    }
    
    virtual bool saveToFile(const std::string& filename) const override {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;
        
        file.write(reinterpret_cast<const char*>(&size), sizeof(size));
        Node<T>* current = head;
        while (current) {
            current->data.serialize(file);
            current = current->next;
        }
        
        file.close();
        return true;
    }
    
    virtual bool loadFromFile(const std::string& filename) override {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) return false;
        
        clear();
        int fileSize;
        file.read(reinterpret_cast<char*>(&fileSize), sizeof(fileSize));
        
        for (int i = 0; i < fileSize; i++) {
            T item;
            item.deserialize(file);
            add(item);
        }
        
        file.close();
        return true;
    }
    
    virtual void print() const override {
        if (isEmpty()) {
            std::cout << "Список пуст.\n";
            return;
        }
        
        // Просто выводим данные без заголовков
        Node<T>* current = head;
        int counter = 1;
        
        while (current) {
            const Student& student = current->data;
            
            // Выводим строку в формате: Фамилия | дата | дата | дата | адрес | группа
            std::cout << counter << ". "
                      << student.getLastName() << " | "
                      << student.getBirthDate() << " | "
                      << student.getAdmissionDate() << " | "
                      << student.getExpulsionDate() << " | "
                      << student.getAddress() << " | "
                      << student.getGroup();
            
            // Добавляем статус в скобках
            if (student.isCurrentlyStudying()) {
                std::cout << " (обучается)";
            } else {
                std::cout << " (отчислен)";
            }
            
            std::cout << "\n";
            current = current->next;
            counter++;
        }
    }
    
    void addSorted(const T& item, bool ascending = true) {
        if (isEmpty()) {
            add(item);
            return;
        }
        
        int index = 0;
        Node<T>* current = head;
        while (current) {
            if ((ascending && item < current->data) || 
                (!ascending && current->data < item)) {
                insert(index, item);
                return;
            }
            current = current->next;
            index++;
        }
        add(item);
    }
    
    std::vector<T> findByPredicate(std::function<bool(const T&)> predicate) const {
        std::vector<T> results;
        Node<T>* current = head;
        while (current) {
            if (predicate(current->data)) results.push_back(current->data);
            current = current->next;
        }
        return results;
    }
    
    T& operator[](int index) {
        Node<T>* node = getNode(index);
        if (!node) throw std::out_of_range("Index out of range");
        return node->data;
    }
    
    const T& operator[](int index) const {
        Node<T>* node = getNode(index);
        if (!node) throw std::out_of_range("Index out of range");
        return node->data;
    }
};

// ===================== Класс StudentManager =====================
class StudentManager {
private:
    LinkedList<Student> students;
    
public:
    void addStudent(const Student& student) { students.add(student); }
    void insertStudent(int index, const Student& student) { students.insert(index, student); }
    void removeStudent(int index) { students.remove(index); }
    
    void removeStudentByLastName(const std::string& lastName) {
        for (int i = 0; i < students.getSize(); i++) {
            if (students.get(i).getLastName() == lastName) {
                students.remove(i);
                return;
            }
        }
    }
    
    Student getStudent(int index) const { return students.get(index); }
    void updateStudent(int index, const Student& student) { students.set(index, student); }
    int getStudentCount() const { return students.getSize(); }
    bool isEmpty() const { return students.isEmpty(); }
    void clearAll() { students.clear(); }
    
    std::vector<Student> searchByGroup(const std::string& group) const {
        return students.findByPredicate([&group](const Student& s) {
            return s.getGroup() == group;
        });
    }
    
    std::vector<Student> searchByLastName(const std::string& lastName) const {
        return students.findByPredicate([&lastName](const Student& s) {
            return s.getLastName() == lastName;
        });
    }
    
    std::vector<Student> searchByLastNamePattern(const std::string& pattern) const {
        return students.findByPredicate([&pattern](const Student& s) {
            return s.getLastName().find(pattern) != std::string::npos;
        });
    }
    
    std::vector<Student> searchCurrentlyStudying() const {
        return students.findByPredicate([](const Student& s) {
            return s.isCurrentlyStudying();
        });
    }
    
    void sortByLastName(bool ascending = true) {
        students.sort([ascending](const Student& a, const Student& b) {
            return ascending ? a < b : b < a;
        });
    }
    
    void sortByBirthDate(bool ascending = true) {
        students.sort([ascending](const Student& a, const Student& b) {
            return ascending ? a.compareByBirthDate(b) : b.compareByBirthDate(a);
        });
    }
    
    void sortByAdmissionDate(bool ascending = true) {
        students.sort([ascending](const Student& a, const Student& b) {
            return ascending ? a.compareByAdmissionDate(b) : b.compareByAdmissionDate(a);
        });
    }
    
    void sortByGroup(bool ascending = true) {
        students.sort([ascending](const Student& a, const Student& b) {
            return ascending ? a.compareByGroup(b) : b.compareByGroup(a);
        });
    }
    
    void addStudentSorted(const Student& student, bool ascending = true) {
        students.addSorted(student, ascending);
    }
    
    void printStatistics() const {
        if (students.isEmpty()) {
            std::cout << "Нет данных для статистики.\n";
            return;
        }
        
        int total = students.getSize();
        int currentlyStudying = searchCurrentlyStudying().size();
        
        Student oldest = students.get(0);
        Student youngest = students.get(0);
        for (int i = 1; i < students.getSize(); i++) {
            Student current = students.get(i);
            if (current.getBirthDate() < oldest.getBirthDate()) oldest = current;
            if (current.getBirthDate() > youngest.getBirthDate()) youngest = current;
        }
        
        std::cout << "=== Статистика ===\n"
                  << "Всего студентов: " << total << "\n"
                  << "Обучается сейчас: " << currentlyStudying << "\n"
                  << "Самый старший: " << oldest.getLastName() 
                  << " (" << oldest.getBirthDate().getAge() << " лет)\n"
                  << "Самый младший: " << youngest.getLastName() 
                  << " (" << youngest.getBirthDate().getAge() << " лет)\n"
                  << "==================\n";
    }
    
    void editByGroupAndPattern(const std::string& group, const std::string& pattern,
                               const Student& newData) {
        for (int i = 0; i < students.getSize(); i++) {
            Student& student = students[i];
            if (student.getGroup() == group && 
                student.getLastName().find(pattern) != std::string::npos) {
                student = newData;
            }
        }
    }
    
    void removeByGroup(const std::string& group) {
        for (int i = students.getSize() - 1; i >= 0; i--) {
            if (students.get(i).getGroup() == group) {
                students.remove(i);
            }
        }
    }
    
    bool saveToFile(const std::string& filename) const { return students.saveToFile(filename); }
    bool loadFromFile(const std::string& filename) { return students.loadFromFile(filename); }
    
    void printAllStudents() const {
        std::cout << "=== Список студентов (" << students.getSize() << " шт.) ===\n";
        students.print();
        std::cout << "===============================\n";
    }
    
    void generateTestData(int count) {
        std::vector<std::string> lastNames = {"Иванов", "Петров", "Сидоров", "Кузнецов", 
                                             "Смирнов", "Попов", "Васильев", "Павлов"};
        std::vector<std::string> groups = {"ИВТ-101", "ИВТ-102", "ПМИ-201", "ПМИ-202", "ФИИТ-301"};
        std::vector<std::string> addresses = {"ул. Ленина, 10", "ул. Советская, 25", 
                                              "ул. Мира, 15", "ул. Пушкина, 7"};
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> lastNameDist(0, lastNames.size() - 1);
        std::uniform_int_distribution<> groupDist(0, groups.size() - 1);
        std::uniform_int_distribution<> addressDist(0, addresses.size() - 1);
        std::uniform_int_distribution<> yearDist(1995, 2005);
        std::uniform_int_distribution<> monthDist(1, 12);
        std::uniform_int_distribution<> dayDist(1, 28);
        
        for (int i = 0; i < count; i++) {
            std::string lastName = lastNames[lastNameDist(gen)];
            int birthYear = yearDist(gen);
            Date birthDate(dayDist(gen), monthDist(gen), birthYear);
            Date admissionDate(1, 9, birthYear + 18);
            Date expulsionDate(30, 6, birthYear + 22);
            std::string address = addresses[addressDist(gen)];
            std::string group = groups[groupDist(gen)];
            
            students.add(Student(lastName, birthDate, admissionDate, 
                                expulsionDate, address, group));
        }
    }
    
    void testPerformance() {
        std::cout << "=== Тестирование производительности ===\n";
        
        if (students.isEmpty()) {
            std::cout << "Генерирую тестовые данные...\n";
            generateTestData(1000);
        }
        
        int n = students.getSize();
        std::cout << "Элементов: " << n << "\n";
        
        auto start = std::chrono::high_resolution_clock::now();
        sortByLastName();
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Сортировка: " << duration.count() << " мс\n";
        
        start = std::chrono::high_resolution_clock::now();
        auto results = searchByGroup("ИВТ-101");
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Поиск: " << duration.count() << " мс (найдено: " 
                  << results.size() << ")\n";
        
        start = std::chrono::high_resolution_clock::now();
        Student testStudent("Тестовый", Date(1, 1, 2000), Date(1, 9, 2020),
                           Date(30, 6, 2024), "ул. Тестовая, 1", "ТЕСТ-001");
        students.add(testStudent);
        end = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        std::cout << "Добавление: " << duration.count() << " мс\n";
        
        std::cout << "=======================================\n";
    }
};

#endif // LINKEDLIST_MANAGER_H