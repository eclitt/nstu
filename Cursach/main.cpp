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
        
        // Спрашиваем, обучается ли студент сейчас
        std::cout << "Студент обучается сейчас? (1 - да, 0 - нет): ";
        int isStudying;
        std::cin >> isStudying;
        clearInput();
        
        Date expulsionDate;
        if (isStudying == 1) {
            // Если обучается, устанавливаем дату отчисления в будущем
            expulsionDate = Date(30, 6, admissionDate.getYear() + 4);
        } else {
            std::cout << "Дата отчисления (ДД ММ ГГГГ): ";
            std::cin >> d >> m >> y;
            clearInput();
            expulsionDate = Date(d, m, y);
        }
        
        std::string address = getStringInput("Адрес: ");
        std::string group = getStringInput("Группа: ");
        
        return Student(lastName, birthDate, admissionDate, expulsionDate, address, group);
    }
    
    void printStudentList() {
        if (manager.isEmpty()) {
            std::cout << "Список студентов пуст.\n";
            return;
        }
        
        std::cout << "Список студентов:\n";
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
        
        int choice = getIntInput("1. Удалить по номеру\n2. Удалить по фамилии\n3. Удалить по группе\n4. Удалить по шаблону фамилии и группе\nВыберите: ", 1, 4);
        
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
            case 4: {
                std::string pattern = getStringInput("Шаблон фамилии: ");
                std::string group = getStringInput("Группа: ");
                manager.removeByLastNamePatternAndGroup(pattern, group);
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
        
        int choice = getIntInput("Редактировать по:\n1. Номеру\n2. Поиску (группа + шаблон фамилии)\nВыберите: ", 1, 2);
        
        if (choice == 1) {
            // Редактирование по номеру
            printStudentList();
            int index = getIntInput("Номер студента для редактирования: ", 
                                   1, manager.getStudentCount());
            
            std::cout << "\nТекущие данные:\n";
            manager.printStudent(index - 1);
            
            std::cout << "\nВведите новые данные:\n";
            Student newStudent = inputStudent();
            manager.updateStudent(index - 1, newStudent);
            std::cout << "Данные обновлены!\n";
        } else {
            // Редактирование по поиску
            std::string group = getStringInput("Введите группу: ");
            std::string pattern = getStringInput("Введите шаблон фамилии: ");
            
            auto results = manager.searchByPatternAndGroup(pattern, group);
            if (results.empty()) {
                std::cout << "Студенты не найдены.\n";
                return;
            }
            
            std::cout << "\nНайдено " << results.size() << " студентов:\n";
            for (int i = 0; i < results.size(); i++) {
                std::cout << "[" << i + 1 << "] " << results[i].getLastName() 
                          << " (" << results[i].getGroup() << ")\n";
            }
            
            int studentIndex = getIntInput("Выберите студента для редактирования (0 - отмена): ", 
                                          0, results.size());
            if (studentIndex == 0) {
                return;
            }
            
            std::cout << "\nТекущие данные:\n";
            std::cout << "Фамилия: " << results[studentIndex - 1].getLastName() << "\n"
                      << "Дата рождения: " << results[studentIndex - 1].getBirthDate() << "\n"
                      << "Дата поступления: " << results[studentIndex - 1].getAdmissionDate() << "\n"
                      << "Дата отчисления: " << results[studentIndex - 1].getExpulsionDate() << "\n"
                      << "Адрес: " << results[studentIndex - 1].getAddress() << "\n"
                      << "Группа: " << results[studentIndex - 1].getGroup() << "\n";
            
            std::cout << "\nВведите новые данные:\n";
            Student newStudent = inputStudent();
            
            // Найти и обновить студента в основном списке
            bool updated = false;
            for (int i = 0; i < manager.getStudentCount(); i++) {
                if (manager.getStudent(i).getGroup() == group && 
                    manager.getStudent(i).getLastName().find(pattern) != std::string::npos &&
                    manager.getStudent(i).getLastName() == results[studentIndex - 1].getLastName()) {
                    manager.updateStudent(i, newStudent);
                    updated = true;
                    break;
                }
            }
            
            if (updated) {
                std::cout << "Данные обновлены!\n";
            } else {
                std::cout << "Ошибка обновления.\n";
            }
        }
    }
    
    void searchMenu() {
        int choice = getIntInput("1. По группе\n2. По фамилии\n3. По шаблону фамилии\n4. Обучающиеся сейчас\n5. По шаблону фамилии и группе\nВыберите: ", 1, 5);
        
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
            case 5: {
                std::string pattern = getStringInput("Шаблон фамилии: ");
                std::string group = getStringInput("Группа: ");
                results = manager.searchByPatternAndGroup(pattern, group);
                break;
            }
        }
        
        if (results.empty()) {
            std::cout << "Не найдено.\n";
            return;
        }
        
        std::cout << "Найдено " << results.size() << " студентов:\n";
        
        // Выводим заголовок
        const int col1 = 4;   // №
        const int col2 = 15;  // Фамилия
        const int col3 = 12;  // Дата рождения
        const int col4 = 12;  // Дата поступления
        const int col5 = 12;  // Дата отчисления
        const int col6 = 20;  // Адрес
        const int col7 = 10;  // Группа
        
        std::cout << std::left 
                  << std::setw(col1) << "№" 
                  << std::setw(col2) << "Фамилия"
                  << std::setw(col3) << "Рождение"
                  << std::setw(col4) << "Поступление"
                  << std::setw(col5) << "Отчисление"
                  << std::setw(col6) << "Адрес"
                  << std::setw(col7) << "Группа" << "\n";
        
        std::cout << std::string(80, '-') << "\n";
        
        // Выводим результаты
        for (int i = 0; i < results.size(); i++) {
            const Student& student = results[i];
            
            std::cout << std::left << std::setw(col1) << i + 1
                      << std::setw(col2) << student.getLastName()
                      << std::setw(col3) << student.getBirthDate()
                      << std::setw(col4) << student.getAdmissionDate();
            
            // Выводим дату отчисления только если студент отчислен
            if (!student.isCurrentlyStudying()) {
                std::cout << std::setw(col5) << student.getExpulsionDate();
            } else {
                std::cout << std::setw(col5) << " ";
            }
            
            std::cout << std::setw(col6) << student.getAddress() 
                      << std::setw(col7) << student.getGroup() << "\n";
        }
        
        // Предлагаем просмотреть детальную информацию
        if (!results.empty()) {
            std::cout << "\nХотите просмотреть детальную информацию о студенте? (1 - да, 0 - нет): ";
            int viewDetail;
            std::cin >> viewDetail;
            clearInput();
            
            if (viewDetail == 1) {
                int studentIndex = getIntInput("Введите номер студента из списка: ", 1, results.size());
                
                // Найти студента в основном списке
                for (int i = 0; i < manager.getStudentCount(); i++) {
                    const Student& mainStudent = manager.getStudent(i);
                    if (mainStudent.getLastName() == results[studentIndex - 1].getLastName() &&
                        mainStudent.getGroup() == results[studentIndex - 1].getGroup() &&
                        mainStudent.getBirthDate() == results[studentIndex - 1].getBirthDate()) {
                        manager.printStudent(i);
                        break;
                    }
                }
            }
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
        int sortField = getIntInput("Добавить с сохранением порядка по:\n1. Фамилии\n2. Дате рождения\n3. Группе\nВыберите: ", 1, 3);
        int order = getIntInput("1. По возрастанию\n2. По убыванию\nВыберите: ", 1, 2);
        
        switch (sortField) {
            case 1:
                manager.addStudentSortedByLastName(student, order == 1);
                break;
            case 2:
                manager.addStudentSortedByBirthDate(student, order == 1);
                break;
            case 3:
                manager.addStudentSortedByGroup(student, order == 1);
                break;
        }
        
        std::cout << "Студент добавлен с сохранением порядка!\n";
    }
    
    void viewStudentMenu() {
        if (manager.isEmpty()) {
            std::cout << "Список пуст.\n";
            return;
        }
        
        printStudentList();
        int index = getIntInput("Введите номер студента для просмотра: ", 1, manager.getStudentCount());
        manager.printStudent(index - 1);
    }
    
    void groupEditMenu() {
        std::string group = getStringInput("Введите группу: ");
        std::string pattern = getStringInput("Введите шаблон фамилии: ");
        
        auto results = manager.searchByPatternAndGroup(pattern, group);
        
        std::cout << "Найдено студентов: " << results.size() << "\n";
        
        if (results.empty()) {
            std::cout << "Редактирование не требуется.\n";
            return;
        }
        
        std::cout << "Введите новые данные для редактирования:\n";
        Student newData = inputStudent();
        
        manager.editByGroupAndPattern(group, pattern, newData);
        std::cout << "Редактирование завершено! Обновлено " << results.size() << " записей.\n";
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
            std::cout << "14. Просмотреть студента по номеру\n";
            std::cout << "0. Выход\n";
            
            int choice = getIntInput("Выберите действие: ", 0, 14);
            
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
                case 14:
                    viewStudentMenu();
                    break;
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
}