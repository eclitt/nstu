package com.example.studentcontrol.ui;

import javafx.fxml.FXML;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.scene.control.TabPane;
import javafx.stage.Stage;

public class MainController {
    @FXML private TabPane tabPane;
    private Scene scene;
    private boolean isDark = true;

    public void setScene(Scene scene) {
        this.scene = scene;
    }

    @FXML
    private void setDarkTheme() {
        if (scene != null) {
            scene.getStylesheets().clear();
            scene.getStylesheets().add(getClass().getResource("/css/styles.css").toExternalForm());
            isDark = true;
        }
    }

    @FXML
    private void setLightTheme() {
        if (scene != null) {
            scene.getStylesheets().clear();
            scene.getStylesheets().add(getClass().getResource("/css/light.css").toExternalForm());
            isDark = false;
        }
    }

    @FXML
    private void showAbout() {
        Alert alert = new Alert(Alert.AlertType.INFORMATION);
        alert.setTitle("О программе");
        alert.setHeaderText("Контроль успеваемости студентов");
        alert.setContentText("Версия 1.0\nРазработано с использованием JavaFX и SQLite");
        alert.showAndWait();
    }
}