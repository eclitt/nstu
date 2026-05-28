package com.example.studentcontrol;

import com.example.studentcontrol.dao.DBConnection;
import com.example.studentcontrol.ui.LoginController;
import com.example.studentcontrol.ui.MainController;
import javafx.application.Application;
import javafx.fxml.FXMLLoader;
import javafx.scene.Scene;
import javafx.scene.control.Alert;
import javafx.stage.Stage;

public class MainApp extends Application {

    @Override
    public void start(Stage primaryStage) throws Exception {
        // Инициализация БД
        try {
            DBConnection.initializeDatabase();
        } catch (Exception e) {
            Alert alert = new Alert(Alert.AlertType.ERROR);
            alert.setTitle("Ошибка");
            alert.setHeaderText("Не удалось инициализировать БД");
            alert.setContentText(e.getMessage());
            alert.showAndWait();
            System.exit(1);
        }

        // Показываем окно входа
        FXMLLoader loginLoader = new FXMLLoader(getClass().getResource("/fxml/LoginView.fxml"));
        Scene loginScene = new Scene(loginLoader.load());
        Stage loginStage = new Stage();
        loginStage.setTitle("Авторизация");
        loginStage.setScene(loginScene);
        loginStage.setResizable(false);
        loginStage.showAndWait();

        if (!LoginController.isAuthenticated()) {
            System.exit(0);
        }

        // Главное окно
        FXMLLoader mainLoader = new FXMLLoader(getClass().getResource("/fxml/MainView.fxml"));
        Scene mainScene = new Scene(mainLoader.load());
        mainScene.getStylesheets().add(getClass().getResource("/css/styles.css").toExternalForm());

        MainController mainController = mainLoader.getController();
        mainController.setScene(mainScene);

        primaryStage.setTitle("Контроль успеваемости студентов");
        primaryStage.setScene(mainScene);
        primaryStage.setWidth(900);
        primaryStage.setHeight(600);
        primaryStage.show();
    }

    public static void main(String[] args) {
        launch(args);
    }
}