#include "linkedlist_manager.h"
#include <limits>
#include <climits>

// ===================== Класс Menu =====================
class Menu {
private:
    StudentManager manager;
    
    void clearInput() {
        std::cin.clear();
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    
    int getIntInput(const std::string& prompt, int min = INT_MIN, int max = INT_MAX) {
        int value;
        while (true) {
            std::cout << prompt;
            if (std::cin >> value) {
                if (value >= min && value <= max) {
                    clearInput();
                    return value;
                }
            }
            std::cout << "Ошибка! Введите число от " << min << " до " << max << ".\n";
            clearInput();
        }
    }
    
    std::string getStringInput(const std::string& prompt) {
        std::string value;
        std::cout << prompt;
        std::getline(std::cin, value);
        return value;
    }
    
    Student inputStudent() {
        std::cout << "=== Ввод данных студента ===\n";
        std::string lastName = getStringInput("Фамилия: ");
        
        std::cout << "Дата рождения (ДД ММ ГГГГ): ";
        int d, m, y;
        std::cin >> d >> m >> y;
        clearInput();
        Date birthDate(d, m, y);
        
        std::cout << "Дата поступления (ДД ММ ГГГГ): ";
        std::cin >> d >> m >> y;
        clearInput();
        Date admissionDate(d, m, y);
        
        std::cout << "Дата отчисления (ДД ММ ГГГГ): ";
        std::cin >> d >> m >> y;
        clearInput();
        Date expulsionDate(d, m, y);
        
        std::string address = getStringInput("Адрес: ");
        std::string group = getStringInput("Группа: ");
        
        return Student(lastName, birthDate, admissionDate, expulsionDate, address, group);
    }
    
    void printStudentList() {
        if (manager.isEmpty()) {
            std::cout << "Список студентов пуст.\n";
            return;
        }
        
        for (int i = 0; i < manager.getStudentCount(); i++) {
            std::cout << "[" << i + 1 << "] " << manager.getStudent(i).getLastName() 
                      << " (" << manager.getStudent(i).getGroup() << ")\n";
        }
    }
    
    void addStudentMenu() {
        Student student = inputStudent();
        manager.addStudent(student);
        std::cout << "Студент добавлен!\n";
    }
    
    void insertStudentMenu() {
        if (manager.isEmpty()) {
            std::cout << "Список пуст.\n";
            return;
        }
        
        printStudentList();
        int index = getIntInput("Позиция для вставки (1-" + 
                               std::to_string(manager.getStudentCount() + 1) + "): ", 
                               1, manager.getStudentCount() + 1);
        
        Student student = inputStudent();
        manager.insertStudent(index - 1, student);
        std::cout << "Студент вставлен!\n";
    }
    
    void removeStudentMenu() {
        if (manager.isEmpty()) {
            std::cout << "Список пуст.\n";
            return;
        }
        
        int choice = getIntInput("1. Удалить по номеру\n2. Удалить по фамилии\n3. Удалить по группе\nВыберите: ", 1, 3);
        
        switch (choice) {
            case 1: {
                printStudentList();
                int index = getIntInput("Номер для удаления: ", 1, manager.getStudentCount());
                manager.removeStudent(index - 1);
                std::cout << "Удалено!\n";
                break;
            }
            case 2: {
                std::string lastName = getStringInput("Фамилия: ");
                manager.removeStudentByLastName(lastName);
                std::cout << "Удалено!\n";
                break;
            }
            case 3: {
                std::string group = getStringInput("Группа: ");
                manager.removeByGroup(group);
                std::cout << "Удалено!\n";
                break;
            }
        }
    }
    
    void editStudentMenu() {
        if (manager.isEmpty()) {
            std::cout << "Список пуст.\n";
            return;
        }
        
        printStudentList();
        int index = getIntInput("Номер студента для редактирования: ", 
                               1, manager.getStudentCount());
        
        std::cout << "Текущие данные:\n";
        std::cout << manager.getStudent(index - 1).getLastName() << " | "
                  << manager.getStudent(index - 1).getBirthDate() << " | "
                  << manager.getStudent(index - 1).getAdmissionDate() << " | "
                  << manager.getStudent(index - 1).getExpulsionDate() << " | "
                  << manager.getStudent(index - 1).getAddress() << " | "
                  << manager.getStudent(index - 1).getGroup();
        
        if (manager.getStudent(index - 1).isCurrentlyStudying()) {
            std::cout << " (обучается)";
        } else {
            std::cout << " (отчислен)";
        }
        std::cout << "\n\n";
        
        Student newStudent = inputStudent();
        manager.updateStudent(index - 1, newStudent);
        std::cout << "Данные обновлены!\n";
    }
    
    void searchMenu() {
        int choice = getIntInput("1. По группе\n2. По фамилии\n3. По шаблону\n4. Обучающиеся сейчас\nВыберите: ", 1, 4);
        
        std::vector<Student> results;
        
        switch (choice) {
            case 1: {
                std::string group = getStringInput("Группа: ");
                results = manager.searchByGroup(group);
                break;
            }
            case 2: {
                std::string lastName = getStringInput("Фамилия: ");
                results = manager.searchByLastName(lastName);
                break;
            }
            case 3: {
                std::string pattern = getStringInput("Шаблон фамилии: ");
                results = manager.searchByLastNamePattern(pattern);
                break;
            }
            case 4: {
                results = manager.searchCurrentlyStudying();
                break;
            }
        }
        
        if (results.empty()) {
            std::cout << "Не найдено.\n";
            return;
        }
        
        std::cout << "Найдено " << results.size() << " студентов:\n";
        for (const auto& student : results) {
            std::cout << "• " 
                      << student.getLastName() << " | "
                      << student.getBirthDate() << " | "
                      << student.getAdmissionDate() << " | "
                      << student.getExpulsionDate() << " | "
                      << student.getAddress() << " | "
                      << student.getGroup();
            
            if (student.isCurrentlyStudying()) {
                std::cout << " (обучается)";
            } else {
                std::cout << " (отчислен)";
            }
            std::cout << "\n";
        }
    }
    
    void sortMenu() {
        int choice = getIntInput("1. По фамилии ▲\n2. По фамилии ▼\n3. По дате рождения\n4. По группе\nВыберите: ", 1, 4);
        
        switch (choice) {
            case 1: manager.sortByLastName(true); break;
            case 2: manager.sortByLastName(false); break;
            case 3: {
                bool asc = getIntInput("1. ▲\n2. ▼\nВыберите: ", 1, 2) == 1;
                manager.sortByBirthDate(asc);
                break;
            }
            case 4: {
                bool asc = getIntInput("1. ▲\n2. ▼\nВыберите: ", 1, 2) == 1;
                manager.sortByGroup(asc);
                break;
            }
        }
        
        std::cout << "Сортировка завершена!\n";
    }
    
    void fileMenu() {
        int choice = getIntInput("1. Сохранить\n2. Загрузить\nВыберите: ", 1, 2);
        std::string filename = getStringInput("Имя файла: ");
        
        if (choice == 1) {
            if (manager.saveToFile(filename)) {
                std::cout << "Сохранено!\n";
            } else {
                std::cout << "Ошибка сохранения!\n";
            }
        } else {
            if (manager.loadFromFile(filename)) {
                std::cout << "Загружено!\n";
            } else {
                std::cout << "Ошибка загрузки!\n";
            }
        }
    }
    
    void addSortedMenu() {
        Student student = inputStudent();
        int order = getIntInput("1. По возрастанию\n2. По убыванию\nВыберите: ", 1, 2);
        manager.addStudentSorted(student, order == 1);
        std::cout << "Студент добавлен с сохранением порядка!\n";
    }
    
    void groupEditMenu() {
        std::string group = getStringInput("Введите группу: ");
        std::string pattern = getStringInput("Введите шаблон фамилии: ");
        
        auto groupStudents = manager.searchByGroup(group);
        auto patternStudents = manager.searchByLastNamePattern(pattern);
        
        std::cout << "Найдено студентов в группе " << group << ": " << groupStudents.size() << "\n";
        std::cout << "Найдено студентов по шаблону '" << pattern << "': " << patternStudents.size() << "\n";
        
        std::cout << "Введите новые данные для редактирования:\n";
        Student newData = inputStudent();
        
        manager.editByGroupAndPattern(group, pattern, newData);
        std::cout << "Редактирование завершено!\n";
    }
    
public:
    void run() {
        std::cout << "=== Система управления студентами ===\n";
        
        while (true) {
            std::cout << "\n=== МЕНЮ ===\n";
            std::cout << "Студентов: " << manager.getStudentCount() << "\n";
            std::cout << "1. Добавить студента\n";
            std::cout << "2. Вставить студента\n";
            std::cout << "3. Удалить студента\n";
            std::cout << "4. Редактировать студента\n";
            std::cout << "5. Показать всех\n";
            std::cout << "6. Поиск студентов\n";
            std::cout << "7. Сортировка студентов\n";
            std::cout << "8. Статистика\n";
            std::cout << "9. Работа с файлами\n";
            std::cout << "10. Добавить с сохранением порядка\n";
            std::cout << "11. Редактировать по группе и шаблону\n";
            std::cout << "12. Тестирование производительности\n";
            std::cout << "13. Генерация тестовых данных\n";
            std::cout << "0. Выход\n";
            
            int choice = getIntInput("Выберите действие: ", 0, 13);
            
            switch (choice) {
                case 0:
                    std::cout << "Выход из программы...\n";
                    return;
                case 1:
                    addStudentMenu();
                    break;
                case 2:
                    insertStudentMenu();
                    break;
                case 3:
                    removeStudentMenu();
                    break;
                case 4:
                    editStudentMenu();
                    break;
                case 5:
                    manager.printAllStudents();
                    break;
                case 6:
                    searchMenu();
                    break;
                case 7:
                    sortMenu();
                    break;
                case 8:
                    manager.printStatistics();
                    break;
                case 9:
                    fileMenu();
                    break;
                case 10:
                    addSortedMenu();
                    break;
                case 11:
                    groupEditMenu();
                    break;
                case 12:
                    manager.testPerformance();
                    break;
                case 13: {
                    int count = getIntInput("Сколько записей сгенерировать? ", 1, 10000);
                    manager.generateTestData(count);
                    std::cout << "Сгенерировано " << count << " записей\n";
                    break;
                }
                default:
                    std::cout << "Неверный выбор!\n";
            }
        }
    }
};

// ===================== Main функция =====================
int main() {
    // Инициализация генератора случайных чисел
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    
    // Установка локали для корректного вывода кириллицы
    #ifdef _WIN32
        system("chcp 65001 > nul");
    #endif
    
    try {
        Menu menu;
        menu.run();
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "Неизвестная ошибка!\n";
        return 1;
    }
    
    return 0;
} // setf | убрать даты отчисление если учаться| шаблоны по фамилии поиск для удаления / (отчислен убрать) / то есть оставить дату отчисления если отчислен / поиск по шаблону и фамилия И группа / редактирование (поиск оп различным полям) / вставка с сохранением порядка (выбор идеального места) \ вывод определенной записи / 1 поиск по групе ()получаем список) далее поиск среди нового списка по шаблону / не нужна посторанная инфа (только выбраный студент)