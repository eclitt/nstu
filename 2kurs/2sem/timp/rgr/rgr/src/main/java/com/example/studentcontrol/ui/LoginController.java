package com.example.studentcontrol.ui;

import javafx.fxml.FXML;
import javafx.scene.control.*;
import javafx.stage.Stage;

public class LoginController {
    @FXML private TextField usernameField;
    @FXML private PasswordField passwordField;

    private static boolean authenticated = false;

    @FXML
    private void handleLogin() {
        String username = usernameField.getText();
        String password = passwordField.getText();

        if ("admin".equals(username) && "1234".equals(password)) {
            authenticated = true;
            ((Stage) usernameField.getScene().getWindow()).close();
        } else {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Ошибка");
            alert.setHeaderText("Неверный логин или пароль");
            alert.showAndWait();
        }
    }

    @FXML
    private void handleCancel() {
        authenticated = false;
        ((Stage) usernameField.getScene().getWindow()).close();
    }

    public static boolean isAuthenticated() {
        return authenticated;
    }
}