package com.example.studentcontrol.ui;

import com.example.studentcontrol.dao.StudentDAO;
import com.example.studentcontrol.model.Student;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import java.time.LocalDate;
import java.util.List;

public class StudentsTabController {
    @FXML private TableView<Student> studentsTable;
    @FXML private TableColumn<Student, Integer> colId;
    @FXML private TableColumn<Student, String> colLastName;
    @FXML private TableColumn<Student, String> colFirstName;
    @FXML private TableColumn<Student, String> colGroup;
    @FXML private TableColumn<Student, LocalDate> colDate;

    @FXML private TextField lastNameField;
    @FXML private TextField firstNameField;
    @FXML private TextField groupField;
    @FXML private TextField dateField;

    private StudentDAO studentDAO = new StudentDAO();
    private ObservableList<Student> students = FXCollections.observableArrayList();

    @FXML
    public void initialize() {
        colId.setCellValueFactory(cellData -> new SimpleIntegerProperty(cellData.getValue().getId()).asObject());
        colLastName.setCellValueFactory(cellData -> new SimpleStringProperty(cellData.getValue().getLastName()));
        colFirstName.setCellValueFactory(cellData -> new SimpleStringProperty(cellData.getValue().getFirstName()));
        colGroup.setCellValueFactory(cellData -> new SimpleStringProperty(cellData.getValue().getGroupName()));
        colDate.setCellValueFactory(cellData -> new SimpleObjectProperty<>(cellData.getValue().getEnrollmentDate()));

        studentsTable.setItems(students);
        refreshTable();
    }

    @FXML
    private void handleAdd() {
        try {
            Student s = new Student();
            s.setFirstName(firstNameField.getText());
            s.setLastName(lastNameField.getText());
            s.setGroupName(groupField.getText());
            s.setEnrollmentDate(LocalDate.parse(dateField.getText()));
            studentDAO.insert(s);
            refreshTable();
            clearFields();
        } catch (Exception e) {
            showError("Ошибка добавления: " + e.getMessage());
        }
    }

    @FXML
    private void handleUpdate() {
        Student selected = studentsTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            showError("Выберите студента для обновления");
            return;
        }
        try {
            selected.setFirstName(firstNameField.getText());
            selected.setLastName(lastNameField.getText());
            selected.setGroupName(groupField.getText());
            selected.setEnrollmentDate(LocalDate.parse(dateField.getText()));
            studentDAO.update(selected);
            refreshTable();
            clearFields();
        } catch (Exception e) {
            showError("Ошибка обновления: " + e.getMessage());
        }
    }

    @FXML
    private void handleDelete() {
        Student selected = studentsTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            showError("Выберите студента для удаления");
            return;
        }
        try {
            studentDAO.delete(selected.getId());
            refreshTable();
        } catch (Exception e) {
            showError("Ошибка удаления: " + e.getMessage());
        }
    }

    private void refreshTable() {
        try {
            List<Student> list = studentDAO.findAll();
            students.setAll(list);
        } catch (Exception e) {
            showError("Ошибка загрузки: " + e.getMessage());
        }
    }

    private void clearFields() {
        lastNameField.clear();
        firstNameField.clear();
        groupField.clear();
        dateField.clear();
    }

    private void showError(String msg) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Ошибка");
        alert.setContentText(msg);
        alert.showAndWait();
    }
}