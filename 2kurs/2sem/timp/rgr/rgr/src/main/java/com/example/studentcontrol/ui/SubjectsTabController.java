package com.example.studentcontrol.ui;

import com.example.studentcontrol.dao.SubjectDAO;
import com.example.studentcontrol.model.Subject;
import javafx.beans.property.SimpleIntegerProperty;
import javafx.beans.property.SimpleStringProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableList;
import javafx.fxml.FXML;
import javafx.scene.control.*;

import java.util.List;

public class SubjectsTabController {
    @FXML private TableView<Subject> subjectsTable;
    @FXML private TableColumn<Subject, Integer> colId;
    @FXML private TableColumn<Subject, String> colName;
    @FXML private TableColumn<Subject, Integer> colSemester;

    @FXML private TextField nameField;
    @FXML private Spinner<Integer> semesterSpinner;

    private SubjectDAO subjectDAO = new SubjectDAO();
    private ObservableList<Subject> subjects = FXCollections.observableArrayList();

    @FXML
    public void initialize() {
        colId.setCellValueFactory(cellData -> new SimpleIntegerProperty(cellData.getValue().getId()).asObject());
        colName.setCellValueFactory(cellData -> new SimpleStringProperty(cellData.getValue().getName()));
        colSemester.setCellValueFactory(cellData -> new SimpleIntegerProperty(cellData.getValue().getSemester()).asObject());

        semesterSpinner.setValueFactory(new SpinnerValueFactory.IntegerSpinnerValueFactory(1, 8, 1));

        subjectsTable.setItems(subjects);
        refreshTable();
    }

    @FXML
    private void handleAdd() {
        try {
            Subject s = new Subject();
            s.setName(nameField.getText());
            s.setSemester(semesterSpinner.getValue());
            subjectDAO.insert(s);
            refreshTable();
            clearFields();
        } catch (Exception e) {
            showError("Ошибка добавления: " + e.getMessage());
        }
    }

    @FXML
    private void handleUpdate() {
        Subject selected = subjectsTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            showError("Выберите предмет для обновления");
            return;
        }
        try {
            selected.setName(nameField.getText());
            selected.setSemester(semesterSpinner.getValue());
            subjectDAO.update(selected);
            refreshTable();
            clearFields();
        } catch (Exception e) {
            showError("Ошибка обновления: " + e.getMessage());
        }
    }

    @FXML
    private void handleDelete() {
        Subject selected = subjectsTable.getSelectionModel().getSelectedItem();
        if (selected == null) {
            showError("Выберите предмет для удаления");
            return;
        }
        try {
            subjectDAO.delete(selected.getId());
            refreshTable();
        } catch (Exception e) {
            showError("Ошибка удаления: " + e.getMessage());
        }
    }

    private void refreshTable() {
        try {
            List<Subject> list = subjectDAO.findAll();
            subjects.setAll(list);
        } catch (Exception e) {
            showError("Ошибка загрузки: " + e.getMessage());
        }
    }

    private void clearFields() {
        nameField.clear();
        semesterSpinner.getValueFactory().setValue(1);
    }

    private void showError(String msg) {
        Alert alert = new Alert(Alert.AlertType.ERROR);
        alert.setTitle("Ошибка");
        alert.setContentText(msg);
        alert.showAndWait();
    }
}