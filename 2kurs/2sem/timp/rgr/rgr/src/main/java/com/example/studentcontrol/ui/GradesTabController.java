package com.example.studentcontrol.ui;

import com.example.studentcontrol.dao.GradeDAO;
import com.example.studentcontrol.dao.StudentDAO;
import com.example.studentcontrol.dao.SubjectDAO;
import com.example.studentcontrol.model.Grade;
import com.example.studentcontrol.model.Student;
import com.example.studentcontrol.model.Subject;
import javafx.application.Platform;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import java.sql.SQLException;
import java.util.List;

public class GradesTabController {
    @FXML private ComboBox<Student> studentCombo;
    @FXML private ComboBox<Subject> subjectCombo;
    @FXML private Spinner<Integer> gradeSpinner;

    @FXML private TableView<Grade> gradesTable;
    @FXML private TableColumn<Grade, Integer> colId;
    @FXML private TableColumn<Grade, String> colSubject;
    @FXML private TableColumn<Grade, Integer> colGrade;
    @FXML private TableColumn<Grade, String> colDate;

    private StudentDAO studentDAO = new StudentDAO();
    private SubjectDAO subjectDAO = new SubjectDAO();
    private GradeDAO gradeDAO = new GradeDAO();
    private ObservableList<Grade> grades = FXCollections.observableArrayList();

    @FXML
    public void initialize() {
        System.out.println("GradesTabController initialized");

        // Настройка спиннера для оценок
        gradeSpinner.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 10, 4));

        // Настройка колонок таблицы
        colId.setCellValueFactory(cellData -> new SimpleIntegerProperty(cellData.getValue().getId()).asObject());
        colSubject.setCellValueFactory(cellData -> new SimpleStringProperty(cellData.getValue().getSubject().getName()));
        colGrade.setCellValueFactory(cellData -> new SimpleIntegerProperty(cellData.getValue().getGrade()).asObject());
        colDate.setCellValueFactory(cellData -> new SimpleStringProperty(cellData.getValue().getDateAssigned().toString()));

        gradesTable.setItems(grades);

        // Загружаем данные в фоновом потоке
        Platform.runLater(() -> {
            loadStudents();
            loadSubjects();
        });

        // Слушатель выбора студента
        studentCombo.getSelectionModel().selectedItemProperty().addListener((obs, oldVal, newVal) -> {
            if (newVal != null) {
                System.out.println("Selected student: " + newVal.getLastName());
                loadGrades(newVal.getId());
            }
        });
    }

    private void loadStudents() {
        try {
            List<Student> students = studentDAO.findAll();
            System.out.println("Loaded " + students.size() + " students");
            ObservableList<Student> studentList = FXCollections.observableArrayList(students);
            studentCombo.setItems(studentList);

            // Если есть студенты, выбираем первого
            if (!studentList.isEmpty()) {
                studentCombo.getSelectionModel().selectFirst();
            }
        } catch (SQLException e) {
            System.err.println("Error loading students: " + e.getMessage());
            e.printStackTrace();
            showError("Ошибка загрузки студентов: " + e.getMessage());
        }
    }

    private void loadSubjects() {
        try {
            List<Subject> subjects = subjectDAO.findAll();
            System.out.println("Loaded " + subjects.size() + " subjects");
            ObservableList<Subject> subjectList = FXCollections.observableArrayList(subjects);
            subjectCombo.setItems(subjectList);
        } catch (SQLException e) {
            System.err.println("Error loading subjects: " + e.getMessage());
            e.printStackTrace();
            showError("Ошибка загрузки предметов: " + e.getMessage());
        }
    }

    private void loadGrades(int studentId) {
        try {
            List<Grade> gradeList = gradeDAO.findByStudent(studentId);
            System.out.println("Loaded " + gradeList.size() + " grades for student " + studentId);
            grades.setAll(gradeList);
        } catch (SQLException e) {
            System.err.println("Error loading grades: " + e.getMessage());
            e.printStackTrace();
            showError("Ошибка загрузки оценок: " + e.getMessage());
        }
    }

    @FXML
    private void handleAdd() {
        Student student = studentCombo.getSelectionModel().getSelectedItem();
        Subject subject = subjectCombo.getSelectionModel().getSelectedItem();

        if (student == null) {
            showError("Выберите студента");
            return;
        }

        if (subject == null) {
            showError("Выберите предмет");
            return;
        }

        int grade = gradeSpinner.getValue();

        try {
            gradeDAO.insert(student.getId(), subject.getId(), grade);
            loadGrades(student.getId());

            // Показываем сообщение об успехе
            Alert alert = new Alert(Alert.AlertType.INFORMATION);
            alert.setTitle("Успех");
            alert.setHeaderText(null);
            alert.setContentText("Оценка " + grade + " добавлена студенту " + student.getLastName());
            alert.showAndWait();

        } catch (SQLException e) {
            System.err.println("Error adding grade: " + e.getMessage());
            e.printStackTrace();
            showError("Ошибка добавления оценки: " + e.getMessage());
        }
    }

    private void showError(String msg) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Ошибка");
        alert.setHeaderText(null);
        alert.setContentText(msg);
        alert.showAndWait();
    }
}